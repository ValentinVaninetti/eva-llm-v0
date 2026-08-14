//! Pedido de GPT: evidencia CAUSAL de por qué ClockMem se achata hacia
//! memoria corta, no otro número de bpb agregado.
//!
//! Parte 1 -- sensibilidad: gradiente de la pérdida respecto de CADA
//! `log_clock` (el parámetro real; alpha=sigmoid(log_clock)), agregado
//! canal por canal sobre todo val. Una sola pasada con backward por
//! ventana -- no es una corrida de entrenamiento, es UN gradiente medido,
//! sin optimizer.step().
//!
//! Parte 2 -- perturbación finita: para un subconjunto de canales
//! (percentiles de alpha por bloque), mover alpha ×0.5 y ×2 SIN
//! reentrenar, medir Δbpb sobre todo val, y devolver el canal a su valor
//! original antes del siguiente.
//!
//! Convención de signo, explícita para no confundir la lectura: gradient
//! descent hace `param -= lr*grad`. Si grad(log_clock) > 0, el próximo
//! paso EMPUJA log_clock hacia abajo -> alpha hacia abajo (más rápido).
//! Si grad < 0, empuja alpha hacia arriba (más lento).

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::autograd::backward;
use eva_llm_v0::tensor::ops;
use std::io::Write;

// Respeta EVA_ALPHA_ANTISAT (algebraic_sigmoid) y EVA_ALPHA_TEMP
// (sigmoid(z/T)) -- el checkpoint pudo entrenarse con cualquiera de las
// dos. Mismas fórmulas que clock.rs (privadas ahí, duplicadas acá igual
// que en otros scripts de hoy).
fn antisat() -> bool { std::env::var("EVA_ALPHA_ANTISAT").is_ok() }
fn temperatura() -> f32 { std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0) }
fn sigmoid(x: f32) -> f32 {
    if antisat() { 0.5 * (1.0 + x / (1.0 + x * x).sqrt()) } else { 1.0 / (1.0 + (-x / temperatura()).exp()) }
}
fn logit(p: f32) -> f32 {
    if antisat() { (2.0 * p - 1.0) / (2.0 * (p * (1.0 - p)).sqrt()) } else { temperatura() * (p / (1.0 - p)).ln() }
}

fn bpb_total(model: &eva_llm_v0::model::EvaModel, ds: &TextDataset, from: usize, to: usize) -> f32 {
    let mut nats = 0.0f64;
    let mut n_pos = 0usize;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        nats += ops::cross_entropy(&logits, &target).data[0] as f64 * input.len() as f64;
        n_pos += input.len();
    }
    (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0; let mut sxx = 0.0; let mut syy = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mx; let dy = ys[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 { return 0.0; }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let mut model = load_model(&weights).expect("no pude cargar el checkpoint");
    let dim = model.cfg.dim;
    let n_blocks = model.blocks.len();
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    println!("eva alpha_gradiente: {} bloques, dim {dim}, {} ventanas de val\n", n_blocks, n_val);

    // ================= PARTE 1: sensibilidad =================
    let mut suma_grad = vec![vec![0f64; dim]; n_blocks];
    let mut suma_abs = vec![vec![0f64; dim]; n_blocks];
    let mut suma_sq = vec![vec![0f64; dim]; n_blocks];
    let mut n_muestras = 0usize;

    for wi in n_train..n {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        let loss = ops::cross_entropy(&logits, &target);
        let grads = backward(&loss);
        for (bi, block) in model.blocks.iter().enumerate() {
            if let Mixer::Clock(clock) = &block.mixer {
                if let Some(g) = grads.get(&clock.log_clock.id) {
                    for c in 0..dim {
                        let gv = g[c] as f64;
                        suma_grad[bi][c] += gv;
                        suma_abs[bi][c] += gv.abs();
                        suma_sq[bi][c] += gv * gv;
                    }
                }
            }
        }
        n_muestras += 1;
    }
    println!("=== PARTE 1: gradiente de la pérdida respecto de log_clock, {n_muestras} ventanas ===");
    println!("  convención: grad>0 empuja alpha MÁS CHICO (rápido) en el próximo paso; grad<0 empuja MÁS GRANDE (lento)\n");

    // Nombre derivado del checkpoint -- si no, correr esto sobre dos
    // checkpoints distintos pisa el resultado bruto del primero (ya me
    // pasó una vez esta noche).
    let base = std::path::Path::new(&weights).file_stem().unwrap().to_string_lossy();
    let ruta_csv = format!("/tmp/alpha_gradiente_bruto_{base}.csv");
    let mut archivo = std::fs::File::create(&ruta_csv)
        .expect("no pude crear el archivo de resultado bruto");
    writeln!(archivo, "bloque,canal,alpha,grad_medio,abs_grad_medio,desvio_grad").unwrap();

    let mut alphas_todas = Vec::new();
    let mut grads_todas = Vec::new();
    let mut abs_grads_todas = Vec::new();

    for bi in 0..n_blocks {
        let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
        let mut filas: Vec<(usize, f32, f64, f64, f64)> = (0..dim).map(|c| {
            let alpha = sigmoid(clock.log_clock.data[c]);
            let media = suma_grad[bi][c] / n_muestras as f64;
            let media_abs = suma_abs[bi][c] / n_muestras as f64;
            let var = suma_sq[bi][c] / n_muestras as f64 - media * media;
            (c, alpha, media, media_abs, var.max(0.0).sqrt())
        }).collect();
        filas.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for &(c, alpha, g, ga, gs) in &filas {
            writeln!(archivo, "{bi},{c},{alpha:.6},{g:.8},{ga:.8},{gs:.8}").unwrap();
            alphas_todas.push(alpha as f64);
            grads_todas.push(g);
            abs_grads_todas.push(ga);
        }

        // resumen por banda de alpha, igual estilo que alpha_dispersion.rs
        let bandas = [("rápido <0.05", 0.0f32, 0.05f32), ("medio 0.05-0.3", 0.05, 0.3), ("lento >0.3", 0.3, 1.01)];
        println!("  bloque {bi}:");
        for (nombre, lo, hi) in bandas {
            let sub: Vec<&(usize, f32, f64, f64, f64)> = filas.iter().filter(|f| f.1 >= lo && f.1 < hi).collect();
            if sub.is_empty() { println!("    {nombre:<16} sin canales"); continue; }
            let n = sub.len() as f64;
            let g_medio = sub.iter().map(|f| f.2).sum::<f64>() / n;
            let ga_medio = sub.iter().map(|f| f.3).sum::<f64>() / n;
            println!("    {nombre:<16} n={:<4} grad_medio={:+.6}  |grad|_medio={:.6}", sub.len(), g_medio, ga_medio);
        }
        let corr = pearson(
            &filas.iter().map(|f| f.1 as f64).collect::<Vec<_>>(),
            &filas.iter().map(|f| f.2).collect::<Vec<_>>(),
        );
        println!("    correlación(alpha, grad firmado) = {corr:.3}");
    }

    let corr_global = pearson(&alphas_todas, &grads_todas);
    let corr_global_abs = pearson(&alphas_todas, &abs_grads_todas);
    println!("\n  correlación GLOBAL (alpha, grad firmado)  = {corr_global:.3}   (pregunta D)");
    println!("  correlación GLOBAL (alpha, |grad|)        = {corr_global_abs:.3}   (pregunta B: negativa = rápidos reciben más señal)");
    println!("  resultado bruto (2560 filas) en {ruta_csv}");

    // ================= PARTE 2: perturbación finita =================
    println!("\n=== PARTE 2: perturbación finita, sin reentrenar ===");
    let base_bpb = bpb_total(&model, &ds, n_train, n);
    println!("  bpb base (sin perturbar): {base_bpb:.4}\n");
    println!("  {:<4} {:<6} {:>8} {:>10} {:>10} {:>10}", "blk", "canal", "alpha", "×0.5 Δbpb", "×2 Δbpb", "banda");

    for bi in 0..n_blocks {
        let percentiles: Vec<usize> = {
            let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
            let mut idx: Vec<usize> = (0..dim).collect();
            idx.sort_by(|&a, &b| clock.log_clock.data[a].partial_cmp(&clock.log_clock.data[b]).unwrap());
            [0, dim/4, dim/2, 3*dim/4, dim-1].iter().map(|&r| idx[r]).collect()
        };
        for &canal in &percentiles {
            let alpha_orig = {
                let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
                sigmoid(clock.log_clock.data[canal])
            };
            let mut deltas = [0f32; 2];
            for (k, factor) in [0.5f32, 2.0f32].iter().enumerate() {
                let alpha_nuevo = (alpha_orig * factor).clamp(1e-4, 1.0 - 1e-4);
                let lc_nuevo = logit(alpha_nuevo);
                let lc_viejo = {
                    let Mixer::Clock(clock) = &mut model.blocks[bi].mixer else { unreachable!() };
                    let viejo = clock.log_clock.data[canal];
                    std::sync::Arc::make_mut(&mut clock.log_clock.data)[canal] = lc_nuevo;
                    viejo
                };
                let bpb_pert = bpb_total(&model, &ds, n_train, n);
                deltas[k] = bpb_pert - base_bpb;
                let Mixer::Clock(clock) = &mut model.blocks[bi].mixer else { unreachable!() };
                std::sync::Arc::make_mut(&mut clock.log_clock.data)[canal] = lc_viejo; // restaurar
            }
            let banda = if alpha_orig < 0.05 { "rápido" } else if alpha_orig < 0.3 { "medio" } else { "lento" };
            println!("  {:<4} {:<6} {:>8.4} {:>+10.5} {:>+10.5} {:>10}", bi, canal, alpha_orig, deltas[0], deltas[1], banda);
        }
    }

    println!("\n=== VEREDICTO ===");
    println!("  A. sesgo sistemático hacia más rápido: corr(alpha,grad)={corr_global:.3} -- si es");
    println!("     positiva y clara, el gradiente actual SÍ empuja a los lentos hacia abajo.");
    println!("  B/C. mirar |grad| por banda arriba -- si 'rápido' tiene |grad| mucho más chico que");
    println!("     'lento', los canales rápidos ya están asentados (gradiente casi nulo) y son los");
    println!("     lentos los que siguen recibiendo señal activa (en cualquier dirección).");
    println!("  D. perturbación: si Δbpb sube consistente al alejarse del alpha aprendido (en las");
    println!("     dos direcciones), la pérdida SÍ prefiere localmente lo que aprendió -- selección");
    println!("     activa. Si Δbpb es ~0 en varias filas, esos canales son indiferentes: el");
    println!("     achatamiento ahí no lo explica la loss, hay que mirar parametrización/AdamW/wd.");
}
