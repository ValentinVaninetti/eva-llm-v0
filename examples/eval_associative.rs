//! Eval for the associative-recall benchmark (`generate_associative.rs`).
//!
//! Recomputes, from the raw byte stream alone (same convention the
//! generator's own self-check uses), which positions are QUERY positions
//! (a key byte whose value was already bound earlier in this window) vs
//! BINDING positions (first occurrence of that key -- unpredictable by
//! construction, the equivalent of `cold` in entity_retrieval.rs). No
//! side-channel metadata file: the byte ranges (KEY/VALUE/FILLER) are
//! fixed constants shared with the generator.
//!
//! Metric: accuracy/bpb of predicting the byte AFTER a query key (i.e. the
//! bound value), bucketed by the gap (in bytes) since that key's previous
//! occurrence in the window -- this is exactly the "does the model recall
//! this specific key's binding, at whatever distance it happened"
//! question, with no fixed N to lean on.
//!
//! USAGE: cargo run --release --example eval_associative -- <weights> <data>

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::save::load_model;

const KEY_LO: u16 = 0;
const KEY_HI: u16 = 32;

fn main() {
    let weights = std::env::args().nth(1).expect("usage: eval_associative <weights> <data> [max_windows_per_section]");
    let data = std::env::args().nth(2).expect("usage: eval_associative <weights> <data> [max_windows_per_section]");
    // Optional cap on how many windows each section scores. Large corpora
    // (20k+ windows) take over an hour to score in full; a few hundred
    // windows already give tens of thousands of query positions, which is
    // far more than the buckets need. The cap takes the FIRST n windows of
    // each section, so the val sample stays strictly inside the val split.
    let max_per_section = std::env::args().nth(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(usize::MAX);

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n_win = ds.num_windows();
    let n_val = (((n_win as f32) * 0.1).round() as usize).clamp(1, n_win / 2);
    let n_train = n_win - n_val;

    let val_to = n_win.min(n_train.saturating_add(max_per_section));
    let train_to = n_train.min(max_per_section);

    println!("eva eval_associative: {weights} | {data} | seq={seq}");
    println!("  windows: {n_train} train, {n_val} val (contiguous cut)");
    if max_per_section != usize::MAX {
        println!("  scoring a capped sample: {} val windows, {} train windows", val_to - n_train, train_to);
    }
    println!();

    report("VAL (unseen text)", &model, &ds, n_train, val_to);
    report("TRAIN (memorization reference)", &model, &ds, 0, train_to);
}

struct Bucket {
    n: usize,
    correct: usize,
    nats: f64,
}
impl Bucket {
    fn new() -> Self { Bucket { n: 0, correct: 0, nats: 0.0 } }
    fn add(&mut self, ok: bool, nat: f64) { self.n += 1; self.correct += ok as usize; self.nats += nat; }
    fn acc(&self) -> f32 { 100.0 * self.correct as f32 / self.n.max(1) as f32 }
    fn bpb(&self) -> f32 { (self.nats / self.n.max(1) as f64 / std::f64::consts::LN_2) as f32 }
}

fn bucket_idx(gap: usize) -> usize {
    match gap {
        0..=32 => 0,
        33..=128 => 1,
        129..=256 => 2,
        257..=512 => 3,
        _ => 4,
    }
}
const BUCKET_NAMES: [&str; 5] = ["gap<=32", "32<gap<=128", "128<gap<=256", "256<gap<=512", "gap>512"];

fn report(label: &str, model: &EvaModel, ds: &TextDataset, from: usize, to: usize) {
    let seq = ds.seq;
    let mut buckets: [Bucket; 5] = [Bucket::new(), Bucket::new(), Bucket::new(), Bucket::new(), Bucket::new()];
    let mut binding = Bucket::new(); // first occurrence: unpredictable, reference floor
    let mut filler = Bucket::new(); // non-key positions: sanity reference (should be near-uniform too, no structure to learn)

    // "Coin-flip" breakdown of what the model predicts at query positions.
    // Hypothesis under test: a model that has learned WHICH VALUES are in
    // play in this window, but not WHICH KEY BINDS TO WHICH, scores
    // 1/(active keys) by construction -- 50% with 2 keys. If that is what
    // is happening, its errors concentrate on the value bound to a
    // DIFFERENT key of the same window, rather than on arbitrary bytes.
    let mut q_total = 0usize;
    let mut q_right = 0usize;      // top-1 == this key's bound value
    let mut q_other_key = 0usize;  // top-1 == some OTHER key's bound value
    let mut q_other_val = 0usize;  // top-1 is in the VALUE range but bound to nobody here
    let mut q_off_range = 0usize;  // top-1 isn't even a value byte

    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        let v = logits.shape[1];
        let mut last_pos: [Option<usize>; 32] = [None; 32];
        let mut bound_seen: [bool; 32] = [false; 32];
        let mut bound_val: [Option<usize>; 32] = [None; 32];

        for t in 0..seq.min(input.len()) {
            let b = input[t] as u16;
            let row = &logits.data[t * v..(t + 1) * v];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let esum: f64 = row.iter().map(|&x| ((x - m) as f64).exp()).sum();
            let label = target[t];
            let nat = -((row[label] - m) as f64 - esum.ln());
            let mut best = 0usize;
            for j in 1..v { if row[j] > row[best] { best = j; } }
            let ok = best == label;

            if (KEY_LO..KEY_HI).contains(&b) {
                let key = b as usize;
                if !bound_seen[key] {
                    bound_seen[key] = true;
                    bound_val[key] = Some(label);
                    binding.add(ok, nat);
                } else {
                    let gap = t - last_pos[key].unwrap();
                    buckets[bucket_idx(gap)].add(ok, nat);

                    q_total += 1;
                    if ok {
                        q_right += 1;
                    } else if (KEY_HI as usize..64).contains(&best)
                        && bound_val.iter().enumerate().any(|(k, bv)| k != key && *bv == Some(best))
                    {
                        q_other_key += 1;
                    } else if (KEY_HI as usize..64).contains(&best) {
                        q_other_val += 1;
                    } else {
                        q_off_range += 1;
                    }
                }
                last_pos[key] = Some(t);
            } else {
                filler.add(ok, nat);
            }
        }
    }

    println!("=== {label} ===");
    println!("  {:<14} {:>7} {:>7} {:>8}", "bucket", "n", "acc%", "bpb");
    for i in 0..5 {
        println!("  {:<14} {:>7} {:>7.2} {:>8.4}", BUCKET_NAMES[i], buckets[i].n, buckets[i].acc(), buckets[i].bpb());
    }
    println!("  {:<14} {:>7} {:>7.2} {:>8.4}   <- unpredictable by construction (first occurrence)", "binding (cold)", binding.n, binding.acc(), binding.bpb());
    println!("  {:<14} {:>7} {:>7.2} {:>8.4}   <- non-key positions, structural reference", "filler", filler.n, filler.acc(), filler.bpb());
    println!("  chance floor: value alphabet is 32 symbols -> 3.125% acc, 5.0 bpb if guessing uniformly among values");

    let pct = |x: usize| 100.0 * x as f32 / q_total.max(1) as f32;
    println!("\n  what the top-1 prediction IS at query positions (n={q_total}):");
    println!("    {:<34} {:>7.2}%", "this key's bound value (correct)", pct(q_right));
    println!("    {:<34} {:>7.2}%   <- the coin-flip signature", "ANOTHER key's bound value", pct(q_other_key));
    println!("    {:<34} {:>7.2}%", "a value byte bound to nobody here", pct(q_other_val));
    println!("    {:<34} {:>7.2}%", "not even a value byte", pct(q_off_range));
    println!("  reading: if 'another key's bound value' absorbs most of the errors, the model");
    println!("  narrowed to the right candidate SET but did not learn the key->value BINDING.\n");
}
