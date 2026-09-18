//! The kid does not learn from its mistakes WHILE IN USE.
//!
//! The table is built once from train and is read-only afterwards, so when the
//! final system errs and the real answer arrives (here, the corpus's real byte,
//! simulating streaming / transcription / correction) that error is lost.
//!
//! The teacher, as specified: it marks the EXACT PAIR (context -> real byte),
//! not a concept; only where the FINAL system erred AND memory was ignorant (a
//! miss, or a hit with high H(q)) -- if the table already knew, the hit
//! corrects it and marking there is redundant; and it enters as a +1 count
//! under the same discipline as `build` (MIN_COUNT=2 in `find`), not as a
//! "this is THE answer" flag. The teacher who marks, not the one who shouts.
//!
//! Only `src/` change: `Recall::observe`, committed separately. The rest is
//! re-aggregation over the same frozen forward; only the table differs.
//!
//! The number that decides: treatment second half vs CONTROL second half, where
//! the control is the same stream with memory always frozen. Not "first vs
//! second half within treatment", which confounds learning with the text simply
//! getting easier later.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::save::load_model;

const CAP: f32 = 7.7; // same Seneca cap as always
const EPS: f32 = 1e-9;
const LAMBDA: f32 = 0.2; // fixed lambda published this week (kNN-LM style)
const H_HIGH: f32 = 2.0; // same "ignorant memory" threshold as the regularizer

fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

/// The FINAL system: model + table combined log-linearly (same formula as
/// `adaptive_lambda.rs`, fixed lambda). Returns (nats for this position,
/// combined argmax, Some(H(q)) if there was a hit).
fn combine(model_logits: &[f32], target: usize, table: &Recall, ctx: &[usize], vocab: usize) -> (f32, usize, Option<f32>) {
    let hit = if ctx.is_empty() { None } else { table.lookup_detail(ctx, vocab) };
    let mx_m = model_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bias: Option<Vec<f32>> = hit.as_ref().map(|(q, _)| {
        q.iter().map(|&qi| (qi + EPS).ln().max(-CAP)).collect()
    });
    let h_q = hit.as_ref().map(|(q, _)| shannon_bits(q));

    let mut sum = 0.0f32;
    let mut z_tgt = 0.0f32;
    let (mut argmax, mut best) = (0usize, f32::NEG_INFINITY);
    for (k, &zm) in model_logits.iter().enumerate() {
        let zc = zm - mx_m + LAMBDA * bias.as_ref().map(|b| b[k]).unwrap_or(0.0);
        let e = zc.exp();
        sum += e;
        if k == target {
            z_tgt = zc;
        }
        if zc > best {
            best = zc;
            argmax = k;
        }
    }
    let nats = -(z_tgt - sum.ln());
    (nats, argmax, h_q)
}

struct Result {
    n: usize,
    nats: f64,
    correct: usize,
    lessons: usize,
}

impl Result {
    fn new() -> Self {
        Result { n: 0, nats: 0.0, correct: 0, lessons: 0 }
    }
    fn bpb(&self) -> f64 {
        self.nats / self.n.max(1) as f64 / std::f64::consts::LN_2
    }
    fn accuracy(&self) -> f32 {
        100.0 * self.correct as f32 / self.n.max(1) as f32
    }
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("could not load the checkpoint");
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();

    let control_table = Recall::build(&bytes_train);
    let mut treatment_table = Recall::build(&bytes_train);

    // Ordered, contiguous stream -- same order as the real corpus. The
    // model's forward is identical for both conditions (frozen, not
    // trained here); only the table differs between control and treatment.
    let windows: Vec<usize> = (n_train..n).collect();
    println!("eva learn_on_use: {} stream windows ({} positions)", windows.len(), windows.len() * seq);

    let mut half1_control = Result::new();
    let mut half2_control = Result::new();
    let mut half1_treat = Result::new();
    let mut half2_treat = Result::new();

    let total_positions = windows.len() * seq;
    let mut i_stream = 0usize;

    for &wi in &windows {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_hidden(&input);
        for t in 0..input.len() {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let global_pos = wi * seq + t + 1;
            let from = global_pos.saturating_sub(8);
            let ctx = &ds.ids[from..global_pos];
            let tgt = target[t];

            let first_half = i_stream < total_positions / 2;
            i_stream += 1;

            // CONTROL: memory always frozen.
            let (nats_c, argmax_c, _) = combine(row, tgt, &control_table, ctx, vocab);
            let r_c = if first_half { &mut half1_control } else { &mut half2_control };
            r_c.n += 1;
            r_c.nats += nats_c as f64;
            if argmax_c == tgt { r_c.correct += 1; }

            // TREATMENT: predicts with the current state, then teaches if
            // applicable -- THIS position's lesson can never leak into
            // THIS SAME position's prediction.
            let (nats_t, argmax_t, h_q) = combine(row, tgt, &treatment_table, ctx, vocab);
            let r_t = if first_half { &mut half1_treat } else { &mut half2_treat };
            r_t.n += 1;
            r_t.nats += nats_t as f64;
            let got_it = argmax_t == tgt;
            if got_it { r_t.correct += 1; }

            let ignorant_memory = h_q.map(|h| h > H_HIGH).unwrap_or(true); // None = miss = ignorant
            if !got_it && ignorant_memory {
                treatment_table.observe(ctx, tgt as u8);
                r_t.lessons += 1;
            }
        }
    }

    println!("\n=== THE NUMBER: bpb and accuracy, first vs second half of the stream ===");
    println!("  {:<28} {:>10} {:>10} {:>8}", "condition", "bpb", "accuracy", "n");
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "control  1st half", half1_control.bpb(), half1_control.accuracy(), half1_control.n);
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "control  2nd half", half2_control.bpb(), half2_control.accuracy(), half2_control.n);
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "treatment 1st half", half1_treat.bpb(), half1_treat.accuracy(), half1_treat.n);
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "treatment 2nd half", half2_treat.bpb(), half2_treat.accuracy(), half2_treat.n);
    println!("\n  lessons (wrong AND memory ignorant) in 1st half: {}", half1_treat.lessons);
    println!("  lessons (wrong AND memory ignorant) in 2nd half: {}", half2_treat.lessons);

    println!("\n=== VERDICT -- the comparison that decides: 2nd half treatment vs 2nd half control ===");
    let delta_bpb = half2_treat.bpb() - half2_control.bpb();
    let delta_acc = half2_treat.accuracy() - half2_control.accuracy();
    println!("  delta bpb (treatment - control), 2nd half: {delta_bpb:+.4}");
    println!("  delta accuracy (treatment - control), 2nd half: {delta_acc:+.1} points");
    // Methodological control: if the 2nd half is ALREADY easier than the
    // 1st in the control itself (frozen memory), the text got easier for
    // reasons unrelated to learning -- that has to be discounted before
    // celebrating any improvement from the treatment.
    let text_drift = half2_control.bpb() - half1_control.bpb();
    println!("  (text drift, control 2nd-1st half: {text_drift:+.4} bpb -- if it isn't ~0,");
    println!("   the corpus alone already changes in difficulty, read the delta above with that in mind)");
    if delta_bpb < -0.005 {
        println!("  Treatment improves over control in the 2nd half -- learning while in use");
        println!("  contributes something measurable beyond the text's own drift.");
    } else if delta_bpb > 0.005 {
        println!("  Treatment gets worse than control -- the lessons injected net noise");
        println!("  (ambiguous votes weighing badly) instead of helping.");
    } else {
        println!("  No measurable difference -- at this stream scale (n={} per half) there isn't", half2_control.n);
        println!("  enough repetition within a single val pass for learning-while-in-use to");
        println!("  show. Doesn't refute the mechanism, says more stream/repetition is needed.");
    }
}
