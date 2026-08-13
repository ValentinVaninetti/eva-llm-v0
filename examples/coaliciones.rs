//! Pedido de GPT: las 32 combinaciones posibles de bloques activos (2^5,
//! incluye el vacío y el completo), bpb de cada una, en A y B -- para
//! comparar la SUPERFICIE completa, no sólo bloques individuales o un par.
//! Reusa `forward_skips` (ya público, el mismo de `techo`), cero op nueva,
//! sin entrenar nada.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::save::load_model;

fn bpb_con_skip(model: &eva_llm_v0::model::EvaModel, ds: &TextDataset, from: usize, to: usize, skips: &[usize]) -> f32 {
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let mut nats = 0.0f64;
    let mut n_pos = 0usize;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_skips(&input, skips);
        for t in 0..seq {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&z| (z - mx).exp()).sum();
            let p = (row[target[t]] - mx).exp() / sum;
            nats -= (p as f64).max(1e-12).ln();
            n_pos += 1;
        }
    }
    (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("no pude cargar el checkpoint");
    let n_blocks = model.blocks.len();
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    println!("eva coaliciones: {weights} | {n_blocks} bloques | {n_val} ventanas de validación\n");
    println!("  {:<12} {:>10}  {:>10}", "activos", "bpb", "Δ vs completo");

    let mut base = 0.0f32;
    let total = 1usize << n_blocks;
    let mut filas: Vec<(usize, u32, f32)> = Vec::new(); // (mascara_activos, n_activos, bpb)

    for mask in 0..total {
        // mask = bits de bloques ACTIVOS (1=activo). skips = los apagados.
        let skips: Vec<usize> = (0..n_blocks).filter(|&i| mask & (1 << i) == 0).collect();
        let bpb = if skips.is_empty() {
            bpb_con_skip(&model, &ds, n_train, n, &[])
        } else {
            bpb_con_skip(&model, &ds, n_train, n, &skips)
        };
        if mask as usize == total - 1 {
            base = bpb;
        }
        filas.push((mask, (mask as u32).count_ones(), bpb));
    }

    filas.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0))); // más activos primero
    for (mask, n_activos, bpb) in &filas {
        let activos: Vec<String> = (0..n_blocks).filter(|&i| mask & (1 << i) != 0).map(|i| i.to_string()).collect();
        let etiqueta = if activos.is_empty() { "ninguno".to_string() } else { activos.join(",") };
        println!("  [{:<10}] {:>10.4}  {:>+10.4}  (n_activos={n_activos})", etiqueta, bpb, bpb - base);
    }

    println!("\neva coaliciones: completo (5 activos) = {base:.4} bpb -- referencia para los Δ de arriba");
}
