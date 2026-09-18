//! Where does the ~99% coverage found in the training mix come from?
//! Suspicion: if most of the hits come from LOW orders (2-3, bigrams/
//! trigrams -- generic language statistics) and not from HIGH orders (6-8,
//! specific context), then a constant lambda is dampening the gradient for
//! the wrong reason almost all the time: the model would learn the same
//! thing with or without the table at those positions, and there was
//! nothing to "free up." Lambda should depend on WHICH ORDER answered, not
//! be a fixed number.
//!
//! Only uses the table (CPU, no model loaded, no GPU) -- zero collision
//! with the run. And doesn't touch `recall.rs`: everything it uses is
//! already public (`build`, `lookup`, `hit_profile`).

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;

/// STEP 0 of the auxiliary-signal proposal (TRABAJO-2026-08-12.md): the bar
/// that governs the two candidates (agreement head / sum of logits) isn't
/// arbitrary, it's "the table is already right almost always." If
/// argmax(table) == truth is already ~97% where the count is high, any new
/// mechanism has to beat THAT, not 50%.
///
/// Over VALIDATION (not train): this is the real condition either
/// candidate would have to work under -- generalizing to text the table
/// didn't memorize, same as everything else in this project measures.
fn step_0_bar_by_count(table: &Recall, ds: &TextDataset, n_train: usize, seq: usize) {
    let thresholds = [2u32, 5, 10, 50];
    let mut hits = vec![0usize; thresholds.len()];
    let mut totals = vec![0usize; thresholds.len()];

    for wi in n_train..ds.num_windows() {
        let (input, target) = ds.window(wi);
        for t in 0..input.len() {
            // Context = the real preceding bytes in the full corpus (can
            // come from the tail of train for the first val positions --
            // it's real text, not a leak: the table wasn't built with any
            // byte from here).
            //
            // NOTE: target[t] = ids[wi*seq + t + 1] (TextDataset::window
            // shifts the target by one relative to the input). The global
            // position of the byte being predicted is wi*seq+t+1, NOT
            // wi*seq+t -- I got this wrong the first time and the context
            // was missing the most recent byte. With the bug, the bar came
            // out ~5%; fixed, the real number shows below.
            let global_pos = wi * seq + t + 1;
            let from = global_pos.saturating_sub(8);
            let ctx = &ds.ids[from..global_pos];
            if ctx.is_empty() {
                continue;
            }
            let Some((q, count)) = table.lookup_detail(ctx, 256) else { continue };
            let argmax = q.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
                .0;
            let correct = argmax == target[t];
            for (u, &c0) in thresholds.iter().enumerate() {
                if count >= c0 {
                    totals[u] += 1;
                    if correct { hits[u] += 1; }
                }
            }
        }
    }

    println!("\n=== STEP 0: bar of argmax(table) == truth, over VALIDATION ===");
    println!("  count >= c0   n        accuracy");
    for (u, &c0) in thresholds.iter().enumerate() {
        let pct = 100.0 * hits[u] as f32 / totals[u].max(1) as f32;
        println!("  {c0:>8}   {:>6}   {pct:6.1}%", totals[u]);
    }
    println!("\n  VERDICT: this is the real bar for both candidates (agreement head /");
    println!("  sum of logits). If the new mechanism doesn't beat these numbers at");
    println!("  its own count threshold, it adds nothing -- the table alone already");
    println!("  gets that right, without spending a parameter or a line of gradient.");
}

fn main() {
    let data = std::env::args().nth(1).unwrap_or_else(|| "data/prosa250.txt".into());
    let seq = 64usize;
    let val = 0.1f32;

    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();

    let table = Recall::build(&bytes_train);
    println!("eva coverage_order: table built over {} train bytes ({} windows)",
        bytes_train.len(), n_train);

    // Walks the training text itself with a growing context window up to
    // 8, same as what the model would see at each position.
    let mut covered = 0usize;
    let mut total = 0usize;
    for i in 0..bytes_train.len() {
        let from = i.saturating_sub(8);
        let ctx = &bytes_train[from..i];
        if ctx.is_empty() {
            continue;
        }
        total += 1;
        if table.lookup(ctx, 256).is_some() {
            covered += 1;
        }
    }

    println!("\n=== coverage over the training text itself ===");
    println!("  {covered} / {total} positions ({:.1}%) -- this is the number being dampened", 100.0 * covered as f32 / total.max(1) as f32);

    println!("\n=== which ORDER each hit comes from ===");
    println!("  order   % of the hits");
    let mut low = 0.0f32; // 2-3
    let mut high = 0.0f32; // 6-8
    for (k, pct) in table.hit_profile() {
        println!("  {k:>5}   {pct:5.1}%");
        if k <= 3 { low += pct } else if k >= 6 { high += pct }
    }

    println!("\n=== VERDICT ===");
    println!("  low orders (2-3, generic):  {low:5.1}%");
    println!("  high orders (6-8, specific): {high:5.1}%");
    if low > high * 2.0 {
        println!("  Coverage is dominated by LOW orders -- these are bigrams/trigrams, not");
        println!("  memorized facts. A constant lambda dampens the gradient there the same");
        println!("  as on an order-8 hit, and there was nothing to 'free up': the model");
        println!("  would learn that statistic regardless, with or without the table.");
        println!("  Real candidate: lambda dependent on which order answered (or on the");
        println!("  count/entropy of the q distribution), not a fixed number.");
    } else {
        println!("  Coverage isn't dominated by low orders -- the 'lambda blind to order'");
        println!("  hypothesis doesn't explain the negative result by itself, look at");
        println!("  something else.");
    }

    step_0_bar_by_count(&table, &ds, n_train, seq);
}
