//! Requested independently: CAUSAL evidence for why ClockMem flattens toward
//! short memory, not another aggregate bpb number.
//!
//! Part 1 -- sensitivity: gradient of the loss with respect to EACH
//! `log_clock` (the real parameter; alpha=sigmoid(log_clock)), aggregated
//! channel by channel over all of val. A single pass with backward per
//! window -- not a training run, it's ONE measured gradient, no
//! optimizer.step().
//!
//! Part 2 -- finite perturbation: for a subset of channels (alpha
//! percentiles per block), move alpha x0.5 and x2 WITHOUT retraining,
//! measure delta bpb over all of val, and restore the channel to its
//! original value before the next one.
//!
//! Sign convention, spelled out to avoid misreading: gradient descent does
//! `param -= lr*grad`. If grad(log_clock) > 0, the next step PUSHES
//! log_clock down -> alpha down (faster). If grad < 0, it pushes alpha up
//! (slower).

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::autograd::backward;
use eva_llm_v0::tensor::ops;
use std::io::Write;

// Respects EVA_ALPHA_ANTISAT (algebraic_sigmoid) and EVA_ALPHA_TEMP
// (sigmoid(z/T)) -- the checkpoint could have trained with either one.
// Same formulas as clock.rs (private there, duplicated here just like in
// other scripts today).
fn antisat() -> bool { std::env::var("EVA_ALPHA_ANTISAT").is_ok() }
fn temperature() -> f32 { std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0) }
fn sigmoid(x: f32) -> f32 {
    if antisat() { 0.5 * (1.0 + x / (1.0 + x * x).sqrt()) } else { 1.0 / (1.0 + (-x / temperature()).exp()) }
}
fn logit(p: f32) -> f32 {
    if antisat() { (2.0 * p - 1.0) / (2.0 * (p * (1.0 - p)).sqrt()) } else { temperature() * (p / (1.0 - p)).ln() }
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

    let mut model = load_model(&weights).expect("could not load the checkpoint");
    let dim = model.cfg.dim;
    let n_blocks = model.blocks.len();
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    println!("eva alpha_gradient: {} blocks, dim {dim}, {} val windows\n", n_blocks, n_val);

    // ================= PART 1: sensitivity =================
    let mut grad_sum = vec![vec![0f64; dim]; n_blocks];
    let mut abs_sum = vec![vec![0f64; dim]; n_blocks];
    let mut sq_sum = vec![vec![0f64; dim]; n_blocks];
    let mut n_samples = 0usize;

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
                        grad_sum[bi][c] += gv;
                        abs_sum[bi][c] += gv.abs();
                        sq_sum[bi][c] += gv * gv;
                    }
                }
            }
        }
        n_samples += 1;
    }
    println!("=== PART 1: gradient of the loss with respect to log_clock, {n_samples} windows ===");
    println!("  convention: grad>0 pushes alpha SMALLER (faster) on the next step; grad<0 pushes it LARGER (slower)\n");

    // Name derived from the checkpoint -- otherwise running this over two
    // different checkpoints overwrites the first one's raw result
    // (happened to me once already tonight).
    let base = std::path::Path::new(&weights).file_stem().unwrap().to_string_lossy();
    let csv_path = format!("/tmp/alpha_gradient_bruto_{base}.csv");
    let mut file = std::fs::File::create(&csv_path)
        .expect("could not create the raw result file");
    writeln!(file, "block,channel,alpha,mean_grad,mean_abs_grad,grad_stdev").unwrap();

    let mut all_alphas = Vec::new();
    let mut all_grads = Vec::new();
    let mut all_abs_grads = Vec::new();

    for bi in 0..n_blocks {
        let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
        let mut rows: Vec<(usize, f32, f64, f64, f64)> = (0..dim).map(|c| {
            let alpha = sigmoid(clock.log_clock.data[c]);
            let mean = grad_sum[bi][c] / n_samples as f64;
            let mean_abs = abs_sum[bi][c] / n_samples as f64;
            let var = sq_sum[bi][c] / n_samples as f64 - mean * mean;
            (c, alpha, mean, mean_abs, var.max(0.0).sqrt())
        }).collect();
        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for &(c, alpha, g, ga, gs) in &rows {
            writeln!(file, "{bi},{c},{alpha:.6},{g:.8},{ga:.8},{gs:.8}").unwrap();
            all_alphas.push(alpha as f64);
            all_grads.push(g);
            all_abs_grads.push(ga);
        }

        // Summary by alpha band, same style as alpha_dispersion.rs
        let bands = [("fast <0.05", 0.0f32, 0.05f32), ("medium 0.05-0.3", 0.05, 0.3), ("slow >0.3", 0.3, 1.01)];
        println!("  block {bi}:");
        for (name, lo, hi) in bands {
            let sub: Vec<&(usize, f32, f64, f64, f64)> = rows.iter().filter(|f| f.1 >= lo && f.1 < hi).collect();
            if sub.is_empty() { println!("    {name:<16} no channels"); continue; }
            let n = sub.len() as f64;
            let mean_g = sub.iter().map(|f| f.2).sum::<f64>() / n;
            let mean_ga = sub.iter().map(|f| f.3).sum::<f64>() / n;
            println!("    {name:<16} n={:<4} mean_grad={:+.6}  mean_|grad|={:.6}", sub.len(), mean_g, mean_ga);
        }
        let corr = pearson(
            &rows.iter().map(|f| f.1 as f64).collect::<Vec<_>>(),
            &rows.iter().map(|f| f.2).collect::<Vec<_>>(),
        );
        println!("    correlation(alpha, signed grad) = {corr:.3}");
    }

    let global_corr = pearson(&all_alphas, &all_grads);
    let global_corr_abs = pearson(&all_alphas, &all_abs_grads);
    println!("\n  GLOBAL correlation (alpha, signed grad)  = {global_corr:.3}   (question D)");
    println!("  GLOBAL correlation (alpha, |grad|)        = {global_corr_abs:.3}   (question B: negative = fast channels get more signal)");
    println!("  raw result (2560 rows) in {csv_path}");

    // ================= PART 2: finite perturbation =================
    println!("\n=== PART 2: finite perturbation, no retraining ===");
    let base_bpb = bpb_total(&model, &ds, n_train, n);
    println!("  base bpb (unperturbed): {base_bpb:.4}\n");
    println!("  {:<4} {:<6} {:>8} {:>10} {:>10} {:>10}", "blk", "channel", "alpha", "x0.5 dBpb", "x2 dBpb", "band");

    for bi in 0..n_blocks {
        let percentiles: Vec<usize> = {
            let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
            let mut idx: Vec<usize> = (0..dim).collect();
            idx.sort_by(|&a, &b| clock.log_clock.data[a].partial_cmp(&clock.log_clock.data[b]).unwrap());
            [0, dim/4, dim/2, 3*dim/4, dim-1].iter().map(|&r| idx[r]).collect()
        };
        for &channel in &percentiles {
            let orig_alpha = {
                let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
                sigmoid(clock.log_clock.data[channel])
            };
            let mut deltas = [0f32; 2];
            for (k, factor) in [0.5f32, 2.0f32].iter().enumerate() {
                let new_alpha = (orig_alpha * factor).clamp(1e-4, 1.0 - 1e-4);
                let new_lc = logit(new_alpha);
                let old_lc = {
                    let Mixer::Clock(clock) = &mut model.blocks[bi].mixer else { unreachable!() };
                    let old = clock.log_clock.data[channel];
                    std::sync::Arc::make_mut(&mut clock.log_clock.data)[channel] = new_lc;
                    old
                };
                let perturbed_bpb = bpb_total(&model, &ds, n_train, n);
                deltas[k] = perturbed_bpb - base_bpb;
                let Mixer::Clock(clock) = &mut model.blocks[bi].mixer else { unreachable!() };
                std::sync::Arc::make_mut(&mut clock.log_clock.data)[channel] = old_lc; // restore
            }
            let band = if orig_alpha < 0.05 { "fast" } else if orig_alpha < 0.3 { "medium" } else { "slow" };
            println!("  {:<4} {:<6} {:>8.4} {:>+10.5} {:>+10.5} {:>10}", bi, channel, orig_alpha, deltas[0], deltas[1], band);
        }
    }

    println!("\n=== VERDICT ===");
    println!("  A. systematic bias toward faster: corr(alpha,grad)={global_corr:.3} -- if it's");
    println!("     positive and clear, the current gradient IS pushing the slow ones down.");
    println!("  B/C. look at |grad| per band above -- if 'fast' has |grad| much smaller than");
    println!("     'slow', the fast channels are already settled (near-zero gradient) and it's");
    println!("     the slow ones still getting active signal (in either direction).");
    println!("  D. perturbation: if delta bpb consistently rises moving away from the learned");
    println!("     alpha (in both directions), the loss DOES locally prefer what it learned --");
    println!("     active selection. If delta bpb is ~0 on several rows, those channels are");
    println!("     indifferent: the flattening there isn't explained by the loss, look at the");
    println!("     parametrization/AdamW/wd instead.");
}
