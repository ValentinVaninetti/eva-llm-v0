//! Distribution of counts in the gate's synthetic corpus, and the effective
//! lambda that `mixed_ce_count` produces with c0 = 2. Answers: in the
//! corpus where the gate was tested, how many positions keep the full
//! gradient (floor, c = 2) and how many end up dampened like boilerplate
//! (high c)?
//!
//! Uses only the public API (`TextDataset`, `Recall::lookup_detail`):
//! doesn't touch `src/`, doesn't load a model, doesn't use the GPU.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;

fn main() {
    let data = std::env::args().nth(1).unwrap_or_else(|| "data/sintetico-gate.txt".into());
    let seq = 64usize;
    let lambda = 0.35f32;
    let c0 = 2.0f32;

    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_train = n - (((n as f32) * 0.2).round() as usize);
    let train: Vec<usize> = ds.ids[..n_train * seq].to_vec();
    let table = Recall::build(&train);

    let mut floor = 0u32;  // c = 2  -> effective lambda 0 (full gradient)
    let mut low = 0u32;    // c in 3..=9
    let mut mid = 0u32;    // c in 10..=49
    let mut high = 0u32;   // c >= 50
    let mut miss = 0u32;
    for i in 0..train.len() {
        let ctx = &train[i.saturating_sub(8)..i];
        if ctx.is_empty() {
            continue;
        }
        match table.lookup_detail(ctx, 256) {
            None => miss += 1,
            Some((_, 2)) => floor += 1,
            Some((_, c)) if c <= 9 => low += 1,
            Some((_, c)) if c <= 49 => mid += 1,
            Some(_) => high += 1,
        }
    }
    let tot = (floor + low + mid + high + miss) as f32;
    let pct = |x: u32| 100.0 * x as f32 / tot;
    println!("eva count_gate: {n_train} training windows, {tot:.0} positions");
    println!("  count 2  (floor, lambda_i=0, FULL gradient): {:6.1}%", pct(floor));
    println!("  count 3-9 (lambda_i={:.2}-{:.2})             :   {:6.1}%",
        lambda * (1.0 - c0 / 9.0), lambda * (1.0 - c0 / 3.0), pct(low));
    println!("  count 10-49 (lambda_i={:.2}-{:.2})          :   {:6.1}%",
        lambda * (1.0 - c0 / 49.0), lambda * (1.0 - c0 / 10.0), pct(mid));
    println!("  count >=50 (lambda_i~={:.3})                 :   {:6.1}%",
        lambda * (1.0 - c0 / 50.0), pct(high));
    println!("  no answer (miss)                          :   {:6.1}%", pct(miss));
    let dampened = 100.0 * (high + mid + low) as f32 / tot;
    println!("\n  positions DAMPENED by the gate: {dampened:.1}% -- the gate only");
    println!("  leaves the full gradient on the {:.1}% that is count 2 (the rare facts).",
        pct(floor));
}
