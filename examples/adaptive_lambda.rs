//! Per-position adaptive lambda (from the FREE signal p[argmax]) against a
//! fixed global lambda, kNN-LM style.
//!
//! Both are put in the SAME log-linear form, to isolate one variable (adaptive
//! vs fixed) rather than confounding it with "log-linear vs probability space".
//!
//!   bias_i_k = max(ln(q_i_k + eps), -7.7)       the cap, no beta inside
//!   FIXED:     z_i = z_m,i + beta * bias_i
//!   ADAPTIVE:  z_i = z_m,i + beta * gate_i * bias_i
//!   gate_i   = 0 if p[argmax]_i >= 0.8, else clamp(1 - p[argmax]_i, 0, 1)
//!
//! The formula is spelled out because the original specification reused "beta"
//! in two places, and a literal reading would square it in the adaptive case.
//!
//! A single beta, swept over DEVELOPMENT (the window half already used to train
//! the probe in `table_probe.rs`) and chosen separately for fixed and adaptive.
//! The VALIDATION half is touched exactly once, for the final number.
//!
//! Touches no `src/`: `forward_hidden`, `Recall::lookup_detail` and
//! `bet::classify_row` are public. Zero new ops.

use eva_llm_v0::bet::classify_row;
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::save::load_model;

const CAP: f32 = 7.7;
const CONFIDENCE_FLOOR: f32 = 0.8;
const EPS: f32 = 1e-9;

/// Everything needed per position to recombine with any beta without
/// re-running the model.
struct Pos {
    target: usize,
    model_logits: Vec<f32>,
    /// bias_i_k with the cap already applied, or None if the table didn't
    /// answer here (no-hit position: combines the same as the model alone).
    bias: Option<Vec<f32>>,
    p_argmax: f32,
}

fn collect(model: &EvaModel, ds: &TextDataset, table: &Recall, windows: &[usize], seq: usize) -> Vec<Pos> {
    let vocab = model.cfg.vocab;
    let mut out = Vec::new();
    for &wi in windows {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_hidden(&input);
        for t in 0..input.len() {
            let row = logits.data[t * vocab..(t + 1) * vocab].to_vec();
            let (p_argmax, _, _, _) = classify_row(&row, target[t]);

            let global_pos = wi * seq + t + 1;
            let from = global_pos.saturating_sub(8);
            let ctx = &ds.ids[from..global_pos];
            let bias = if ctx.is_empty() {
                None
            } else {
                table.lookup_detail(ctx, vocab).map(|(q, _count)| {
                    q.iter().map(|&qi| (qi + EPS).ln().max(-CAP)).collect::<Vec<f32>>()
                })
            };
            out.push(Pos { target: target[t], model_logits: row, bias, p_argmax });
        }
    }
    out
}

/// bits/byte of the combination, over a set of positions, for a beta and a
/// mode (adaptive or fixed). Returns (total_bpb, delegation_bpb, safe_bpb,
/// n_delegation, n_safe) for the breakdown we asked for.
fn bpb(positions: &[Pos], beta: f32, adaptive: bool) -> (f64, f64, f64, usize, usize) {
    let mut nats_total = 0.0f64;
    let (mut nats_deleg, mut nats_safe) = (0.0f64, 0.0f64);
    let (mut n_deleg, mut n_safe) = (0usize, 0usize);

    for p in positions {
        let lambda = match &p.bias {
            None => 0.0,
            Some(_) if !adaptive => beta,
            Some(_) => beta * if p.p_argmax >= CONFIDENCE_FLOOR { 0.0 } else { (1.0 - p.p_argmax).clamp(0.0, 1.0) },
        };
        let mx_m = p.model_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Combined z, subtracting the model-alone max for stability (the
        // bias is bounded by the cap, it can't destabilize things further).
        let mut sum = 0.0f32;
        let mut z_tgt = 0.0f32;
        for (k, &zm) in p.model_logits.iter().enumerate() {
            let zc = zm - mx_m + lambda * p.bias.as_ref().map(|b| b[k]).unwrap_or(0.0);
            let e = zc.exp();
            sum += e;
            if k == p.target {
                z_tgt = zc;
            }
        }
        let nats = -(z_tgt - sum.ln());
        nats_total += nats as f64;
        if p.p_argmax < CONFIDENCE_FLOOR {
            nats_deleg += nats as f64;
            n_deleg += 1;
        } else {
            nats_safe += nats as f64;
            n_safe += 1;
        }
    }
    let ln2 = std::f64::consts::LN_2;
    let n = positions.len().max(1) as f64;
    (
        nats_total / n / ln2,
        nats_deleg / n_deleg.max(1) as f64 / ln2,
        nats_safe / n_safe.max(1) as f64 / ln2,
        n_deleg,
        n_safe,
    )
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();
    let table = Recall::build(&bytes_train);

    // Same split as table_probe.rs: half of val to choose beta
    // (development), the other half gets touched exactly once for the
    // final number.
    let half = n_train + (n_val / 2);
    let dev_windows: Vec<usize> = (n_train..half).collect();
    let val_windows: Vec<usize> = (half..n).collect();
    println!("eva adaptive_lambda: {} development windows | {} final validation windows",
        dev_windows.len(), val_windows.len());

    let pos_dev = collect(&model, &ds, &table, &dev_windows, seq);
    let pos_val = collect(&model, &ds, &table, &val_windows, seq);
    println!("eva adaptive_lambda: {} dev positions | {} val positions\n",
        pos_dev.len(), pos_val.len());

    let (base_dev, ..) = bpb(&pos_dev, 0.0, false);
    println!("baseline (model alone, beta=0), dev:  {base_dev:.4} bits/byte");

    // 0.1 and 0.2 added at the lead's request: don't leave the optimum at
    // the edge of the previous grid (0.3 had won, but it was the low end).
    let betas = [0.0f32, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];
    println!("\n=== beta sweep over DEVELOPMENT ===");
    println!("   beta  fixed (bpb)   adaptive (bpb)");
    let (mut best_fixed, mut best_fixed_bpb) = (0.0f32, f64::INFINITY);
    let (mut best_adap, mut best_adap_bpb) = (0.0f32, f64::INFINITY);
    for &b in &betas {
        let (f, ..) = bpb(&pos_dev, b, false);
        let (a, ..) = bpb(&pos_dev, b, true);
        println!("  {b:>4.1}    {f:.4}        {a:.4}");
        if f < best_fixed_bpb { best_fixed_bpb = f; best_fixed = b; }
        if a < best_adap_bpb { best_adap_bpb = a; best_adap = b; }
    }
    println!("\n  best fixed beta:    {best_fixed:.1}  ({best_fixed_bpb:.4} bpb on dev)");
    println!("  best adaptive beta: {best_adap:.1}  ({best_adap_bpb:.4} bpb on dev)");

    // THE NUMBER: validation, touched exactly once, with beta already chosen.
    let (base_val, ..) = bpb(&pos_val, 0.0, false);
    let (fixed_val, ..) = bpb(&pos_val, best_fixed, false);
    let (adap_val, deleg_val, safe_val, n_deleg, n_safe) = bpb(&pos_val, best_adap, true);

    println!("\n=== THE NUMBER: bits/byte on VALIDATION (touched exactly once) ===");
    println!("  model alone (beta=0):                {base_val:.4}");
    println!("  FIXED lambda, kNN-LM style (beta={best_fixed:.1}):    {fixed_val:.4}");
    println!("  ADAPTIVE lambda via p[argmax] (beta={best_adap:.1}): {adap_val:.4}");

    // Same per-zone breakdown but with the MODEL ALONE, to compare apples
    // to apples (not the isolated number, the delta against the baseline
    // in the SAME zone) -- exactly what was asked for.
    let (_, base_deleg, base_safe, ..) = bpb(&pos_val, 0.0, false);

    println!("\n=== bonus: where the gain comes from (adaptive vs model alone, by zone) ===");
    println!("  zone            n       model alone   adaptive     delta");
    println!("  delegation   {n_deleg:>6}   {base_deleg:>10.4}   {deleg_val:>10.4}   {:+.4}",
        deleg_val - base_deleg);
    println!("  safe         {n_safe:>6}   {base_safe:>10.4}   {safe_val:>10.4}   {:+.4}",
        safe_val - base_safe);
    println!("  (the safe zone has to give ~0.0000 difference -- the floor forces lambda=0");
    println!("  there, so it's a control that the implementation doesn't touch what it shouldn't.)");

    println!("\n=== VERDICT ===");
    if adap_val < fixed_val - 0.001 {
        println!("  Adaptive beats fixed ({adap_val:.4} < {fixed_val:.4}).");
    } else if fixed_val < adap_val - 0.001 {
        println!("  Fixed beats adaptive ({fixed_val:.4} < {adap_val:.4}) -- the p[argmax]");
        println!("  adaptation didn't add anything over the simple global lambda.");
    } else {
        println!("  They tie within noise -- the gate doesn't change anything measurable here.");
    }
    if fixed_val < base_val - 0.001 || adap_val < base_val - 0.001 {
        println!("  And both, or at least one, beat the model alone -- the log-linear");
        println!("  combination adds something over not using the table at all.");
    } else {
        println!("  Neither clearly beats the model alone -- the log-linear combination");
        println!("  doesn't add anything at this scale, published or not.");
    }
}
