//! The detached probe, against the REAL bar (~53%, not
//! ~97%).
//!
//! Same question that killed the before-spending stake, applied to
//! something different: it isn't "do I know if I'll get it right?", it's
//! "do I know if the table is right here?" Small head `g = sigmoid(w*h+b)`
//! (same shape as `StakeHead`, reused as-is -- zero new op), DETACHED from
//! the model: it doesn't send it gradient, so if the already-trained model
//! doesn't have the signal, this proves it without spending a single step
//! of real training.
//!
//! Does NOT touch `src/`: everything it uses is already public
//! (`forward_hidden`, `Recall::lookup_detail`, `ops::stake_loss` with
//! span_len=1, `StakeHead`).

use eva_llm_v0::bet::classify_row;
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::rng::Rng;
use eva_llm_v0::save::load_model;
use eva_llm_v0::stake::StakeHead;
use eva_llm_v0::tensor::autograd::backward;
use eva_llm_v0::tensor::ops as ops;
use eva_llm_v0::tensor::Tensor;

/// One already-detached hidden row + its label + the count that produced it.
struct Example {
    hidden: Vec<f32>,
    correct: f32,
    count: u32,
    /// p[argmax] the model ALREADY declares for free (`bet::classify_row`)
    /// -- the bar to beat. If this only rediscovers that, it adds nothing
    /// new.
    model_conf: f32,
}

fn collect(
    model: &eva_llm_v0::model::EvaModel,
    ds: &TextDataset,
    table: &Recall,
    windows: &[usize],
    seq: usize,
    dim: usize,
) -> Vec<Example> {
    let mut out = Vec::new();
    for &wi in windows {
        let (input, target) = ds.window(wi);
        let (logits, hidden) = model.forward_hidden(&input);
        let vocab = model.cfg.vocab;
        for t in 0..input.len() {
            // Same indexing as Step 0, already fixed there once.
            let global_pos = wi * seq + t + 1;
            let from = global_pos.saturating_sub(8);
            let ctx = &ds.ids[from..global_pos];
            if ctx.is_empty() {
                continue;
            }
            let Some((q, count)) = table.lookup_detail(ctx, 256) else { continue };
            // Only hit positions: this is the same population Step 0
            // measured (the bar), so the comparison is apples to apples. A
            // miss doesn't even raise the question "is the table right?".
            let argmax = q.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
                .0;
            let correct = (argmax == target[t]) as u8 as f32;
            let (model_conf, _, _, _) = classify_row(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
            out.push(Example {
                hidden: hidden.data[t * dim..(t + 1) * dim].to_vec(),
                correct,
                count,
                model_conf,
            });
        }
    }
    out
}

fn band(count: u32) -> &'static str {
    match count {
        2..=4 => "2-4",
        5..=9 => "5-9",
        10..=49 => "10-49",
        _ => "50+",
    }
}

fn auroc(pairs: &[(f64, bool)]) -> f64 {
    let mut v: Vec<(f64, bool)> = pairs.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n_pos = v.iter().filter(|(_, p)| *p).count() as f64;
    let n_neg = v.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let mut pos_rank_sum = 0.0f64;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 { j += 1; }
        let mid_rank = (i + 1 + j) as f64 / 2.0;
        for e in &v[i..j] {
            if e.1 { pos_rank_sum += mid_rank; }
        }
        i = j;
    }
    (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("could not load the checkpoint");
    let dim = model.cfg.dim;
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();
    let table = Recall::build(&bytes_train);

    // 50/50 of the VALIDATION windows, not train: the probe trains and
    // evaluates within the same population Step 0 measured, but without
    // testing on what it trained on.
    let half = n_train + (n_val / 2);
    let probe_train_windows: Vec<usize> = (n_train..half).collect();
    let probe_eval_windows: Vec<usize> = (half..n).collect();
    println!("eva table_probe: {} model params | {} probe-train windows | {} probe-eval windows",
        model.param_count(), probe_train_windows.len(), probe_eval_windows.len());

    let examples_train = collect(&model, &ds, &table, &probe_train_windows, seq, dim);
    let examples_eval = collect(&model, &ds, &table, &probe_eval_windows, seq, dim);
    println!("eva table_probe: {} (hit) examples to train the probe, {} to evaluate it",
        examples_train.len(), examples_eval.len());

    // A single large tensor (S=n_examples, D=dim), hidden already detached
    // -- stake_loss with span_len=1 gives one stake per row.
    let hidden_train = Tensor::new(
        examples_train.iter().flat_map(|e| e.hidden.iter().copied()).collect(),
        vec![examples_train.len(), dim],
    );
    let good_train: Vec<f32> = examples_train.iter().map(|e| e.correct).collect();

    let mut rng = Rng::new(11);
    let mut head = StakeHead::new(dim, &mut rng);
    let mut opt = eva_llm_v0::optim::AdamW::new(1e-2, 0.0);

    println!("\n=== training the small head (detached, zero gradient to the model) ===");
    for epoch in 0..60 {
        let loss = ops::stake_loss(&hidden_train, &head.w, &head.b, &good_train, 1);
        let loss_v = loss.data[0];
        let grads = backward(&loss);
        let mut params = head.params_mut();
        opt.step(&mut params, &grads);
        if epoch % 10 == 0 || epoch == 59 {
            println!("  epoch {epoch:>2} | loss (MSE) {loss_v:.4}");
        }
    }

    // Evaluation: pooled AUROC and per count-band, over examples the probe
    // NEVER saw during training (the other half of the validation windows).
    let hidden_eval = Tensor::new(
        examples_eval.iter().flat_map(|e| e.hidden.iter().copied()).collect(),
        vec![examples_eval.len(), dim],
    );
    let stakes = head.stakes(&hidden_eval, 1);

    let mut pool: Vec<(f64, bool)> = Vec::new();
    let mut by_band: std::collections::HashMap<&str, Vec<(f64, bool)>> = std::collections::HashMap::new();
    for (e, &g) in examples_eval.iter().zip(&stakes) {
        let pair = (g as f64, e.correct > 0.5);
        pool.push(pair);
        by_band.entry(band(e.count)).or_default().push(pair);
    }

    let global_bar = examples_eval.iter().filter(|e| e.correct > 0.5).count() as f32
        / examples_eval.len().max(1) as f32;

    println!("\n=== RESULT: AUROC of the probe against table_correct, on eval ===");
    println!("  bar (fraction of table hits on eval): {:.1}%", 100.0 * global_bar);
    println!("  pooled AUROC: {:.3}  (n={})", auroc(&pool), pool.len());
    println!("\n  band       n       bar        AUROC");
    for b in ["2-4", "5-9", "10-49", "50+"] {
        if let Some(pairs) = by_band.get(b) {
            let band_bar = pairs.iter().filter(|(_, c)| *c).count() as f32 / pairs.len().max(1) as f32;
            println!("  {b:<8}  {:>5}   {:5.1}%    {:.3}", pairs.len(), 100.0 * band_bar, auroc(pairs));
        }
    }

    // Risk I flagged in my own proposal: that the small head only
    // rediscovers something already free (the confidence the model
    // ALREADY declares, bet::classify_row) instead of learning something
    // new about the table.
    let pool_conf: Vec<(f64, bool)> = examples_eval.iter()
        .map(|e| (e.model_conf as f64, e.correct > 0.5))
        .collect();
    let free_conf_auroc = auroc(&pool_conf);

    let global = auroc(&pool);
    println!("\n=== CONTROL: is this new, or is it the model's free confidence? ===");
    println!("  AUROC of p[argmax] ALONE (bet::classify_row, zero new parameters): {free_conf_auroc:.3}");
    println!("  AUROC of the trained head:                                        {global:.3}");
    if global > free_conf_auroc + 0.03 {
        println!("  The head beats the free confidence -- it learned something p[argmax]");
        println!("  didn't have. Worth it as a mechanism, not just rediscovering the free one.");
    } else {
        println!("  The head does NOT beat p[argmax] alone -- what it measures was already");
        println!("  free in the confidence the model declares. Nothing new needs training:");
        println!("  use p[argmax] directly as the gate, without spending dim+1 extra parameters.");
    }

    println!("\n=== VERDICT ===");
    if global < 0.55 {
        println!("  pooled AUROC {global:.3} -- chance level against the bar (~{:.0}%). The model", 100.0*global_bar);
        println!("  does NOT know when the table is right. The head dies here, without touching train.rs.");
    } else if global > 0.6 {
        println!("  pooled AUROC {global:.3} -- clearly beats the bar. The signal exists in the");
        println!("  hidden state. Still alive: joint training with a small beta, control beta=0.");
    } else {
        println!("  pooled AUROC {global:.3} -- gray zone between 0.55 and 0.6. Not fully dead nor");
        println!("  alive; decide whether a second seed is worth it before building.");
    }
}
