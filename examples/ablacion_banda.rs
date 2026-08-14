//! Pedido de GPT tras confirmar T=1,3: ¿las bandas rápida/media/lenta
//! cumplen funciones DISTINTAS, o son parámetros equivalentes con
//! distinto alpha nomás? Ablación por banda COMPLETA (todos los canales
//! de esa banda, en TODOS los bloques a la vez, apagados de un saque) --
//! no canal por canal.
//!
//! Sin op nueva ni cambio en `src/`: reimplementa el forward de bloque
//! (`EvaBlock::forward_from` en `src/model/block.rs`) a mano en este
//! ejemplo, usando sólo campos públicos, insertando una máscara sobre la
//! salida del mixer antes de sumarla al residual. Es una pasada de
//! evaluación (sin gradiente), no toca entrenamiento.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::nn::Module;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::ops;
use eva_llm_v0::tensor::Tensor;

fn antisat() -> bool { std::env::var("EVA_ALPHA_ANTISAT").is_ok() }
fn temperatura() -> f32 { std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0) }
fn alpha_de(lc: f32) -> f32 {
    if antisat() { 0.5 * (1.0 + lc / (1.0 + lc * lc).sqrt()) } else { 1.0 / (1.0 + (-lc / temperatura()).exp()) }
}

/// Igual forma que `EvaBlock::forward_from`, con una máscara sobre la
/// salida del mixer (poner en cero canales específicos antes de sumar al
/// residual). `mascara[bloque][canal] = true` significa "apagado".
fn forward_masked(model: &EvaModel, ids: &[usize], mascara: &[Vec<bool>]) -> Tensor {
    let mut x = model.embed.embed(ids);
    if let Some(pos) = &model.pos {
        let n = ids.len().min(model.cfg.seq_len);
        x = ops::add(&x, &ops::slice_rows(pos, n));
    }
    for (i, b) in model.blocks.iter().enumerate() {
        let h = ops::add(&x, &b.conv.forward(&b.norm0.forward(&x)));
        let mut m = b.mixer.forward(&b.norm1.forward(&h));
        if mascara[i].iter().any(|&v| v) {
            let (s, d) = (m.shape[0], m.shape[1]);
            let mut datos = (*m.data).clone();
            for t in 0..s {
                for c in 0..d {
                    if mascara[i][c] { datos[t * d + c] = 0.0; }
                }
            }
            m = Tensor::new(datos, vec![s, d]);
        }
        let h2 = ops::add(&h, &m);
        x = ops::add(&h2, &b.glu.forward(&b.norm2.forward(&h2)));
    }
    let x = model.norm_out.forward(&x);
    ops::matmul(&x, &model.head_w)
}

fn bpb_con_mascara(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, mascara: &[Vec<bool>]) -> f32 {
    let mut nats = 0.0f64;
    let mut n_pos = 0usize;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = forward_masked(model, &input, mascara);
        nats += ops::cross_entropy(&logits, &target).data[0] as f64 * input.len() as f64;
        n_pos += input.len();
    }
    (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7_temp13.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("no pude cargar el checkpoint");
    let n_blocks = model.blocks.len();
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    // Clasificar cada canal, en cada bloque, por banda -- mismas bandas
    // de siempre.
    let mut banda: Vec<Vec<&'static str>> = Vec::new();
    let mut conteo = std::collections::HashMap::new();
    for bi in 0..n_blocks {
        let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
        let dim = clock.log_clock.data.len();
        let mut fila = Vec::with_capacity(dim);
        for c in 0..dim {
            let a = alpha_de(clock.log_clock.data[c]);
            let b = if a < 0.05 { "rápido" } else if a < 0.3 { "medio" } else { "lento" };
            fila.push(b);
            *conteo.entry(b).or_insert(0usize) += 1;
        }
        banda.push(fila);
    }
    println!("eva ablacion_banda: {weights} | conteo total por banda: {:?}\n", conteo);

    let mascara_vacia: Vec<Vec<bool>> = banda.iter().map(|f| vec![false; f.len()]).collect();
    let base = bpb_con_mascara(&model, &ds, n_train, n, &mascara_vacia);
    println!("bpb sin ablacionar (control): {base:.4}\n");

    for objetivo in ["rápido", "medio", "lento"] {
        let mascara: Vec<Vec<bool>> = banda.iter().map(|f| f.iter().map(|&b| b == objetivo).collect()).collect();
        let n_apagados: usize = mascara.iter().map(|f| f.iter().filter(|&&x| x).count()).sum();
        let bpb = bpb_con_mascara(&model, &ds, n_train, n, &mascara);
        println!("apagar banda '{objetivo}' ({n_apagados} canales de {}): bpb {bpb:.4}  Δ={:+.4} ({:+.1}%)",
            banda.len() * banda[0].len(), bpb - base, 100.0 * (bpb - base) / base);
    }

    // Control: apagar un numero ALEATORIO de canales del mismo tamano que
    // 'medio' (la banda mas grande), para saber si el dano de una banda
    // especifica es distinto de apagar "cualquier" grupo de ese tamano.
    println!("\n=== control: mismo N de canales, elegidos sin mirar la banda ===");
    let n_medio = *conteo.get("medio").unwrap_or(&0);
    let mut rng = eva_llm_v0::rng::Rng::new(99);
    let total_canales: usize = banda.iter().map(|f| f.len()).sum();
    let mut todos: Vec<(usize, usize)> = Vec::new();
    for (bi, f) in banda.iter().enumerate() { for c in 0..f.len() { todos.push((bi, c)); } }
    let mut idx: Vec<usize> = (0..todos.len()).collect();
    rng.shuffle(&mut idx);
    let elegidos = &idx[..n_medio.min(total_canales)];
    let mut mascara_random: Vec<Vec<bool>> = banda.iter().map(|f| vec![false; f.len()]).collect();
    for &e in elegidos { let (bi, c) = todos[e]; mascara_random[bi][c] = true; }
    let bpb_random = bpb_con_mascara(&model, &ds, n_train, n, &mascara_random);
    println!("apagar {n_medio} canales AL AZAR (mismo N que 'medio'): bpb {bpb_random:.4}  Δ={:+.4} ({:+.1}%)",
        bpb_random - base, 100.0 * (bpb_random - base) / base);
}
