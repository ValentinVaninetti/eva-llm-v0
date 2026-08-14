//! Task 1 from TRABAJO-2026-08-12.md: does halving bisection locate the
//! first error within a bad span, or is there something better with what
//! we already have?
//!
//! ZERO new forward passes: re-aggregates the trace `bet::scan` already
//! returns. For each span (length 16) with `good < 0.5` over validation,
//! defines first_error = the first position with `correct == false`, and
//! compares three ways of estimating it:
//!
//! (a) whole-span bar -- doesn't localize anything, you'd have to redo the
//!     whole span. This is the "no localization" reference.
//! (b) bisection: splits the span in half, keeps the half with lower mean
//!     confidence, recurses down to a single token. This is the
//!     BitVMX-by-halves candidate as originally proposed.
//! (c) per-token change-point: the token with the lowest declared
//!     confidence (`conf`) within the span -- doesn't bisect anything,
//!     looks at the whole trace.
//!
//! Metrics: pooled AUROC ("is this token the first error?", over all bad
//! spans together), mean distance |predicted - real|, and the payoff --
//! if only the span from the predicted point to the end gets redone, how
//! much of the span is saved and what fraction of the time the predicted
//! point falls BEFORE or AT the real error (so the redo covers it).

use eva_llm_v0::bet::{scan, TokenObs};
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::save::load_model;

const SPAN: usize = 16;
const GOOD_THRESHOLD: f64 = 0.5;

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val: f32 = 0.1;

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    println!("eva bisector: {} params | {} val windows (same cut as eva bet)",
        model.param_count(), n_val);

    let obs = scan(&model, &ds, n_train, n);
    println!("eva bisector: {} per-token observations, a single pass", obs.len());

    // SPAN-long spans, never crossing windows -- same rule as span_analysis.
    let mut bad: Vec<&[TokenObs]> = Vec::new();
    let mut i = 0;
    while i < obs.len() {
        let end = (i / seq + 1) * seq;
        let mut t = i;
        while t + SPAN <= end {
            let span = &obs[t..t + SPAN];
            let good = span.iter().filter(|o| o.correct).count() as f64 / SPAN as f64;
            if good < GOOD_THRESHOLD {
                bad.push(span);
            }
            t += SPAN;
        }
        i = end;
    }
    println!("eva bisector: {} spans with good < {:.1} out of a total of {} possible spans\n",
        bad.len(), GOOD_THRESHOLD, obs.len() / SPAN);

    if bad.is_empty() {
        println!("no bad spans at this threshold -- nothing to localize, done.");
        return;
    }

    // Real first error per span.
    let first_error: Vec<usize> = bad.iter()
        .map(|t| t.iter().position(|o| !o.correct).unwrap_or(0))
        .collect();

    // (b) recursive halving bisection, and a per-token score = how many
    // times it survived as "the worse half."
    let mut b_pred = Vec::new();
    let mut b_scores_pool: Vec<(f64, bool)> = Vec::new(); // (score, is_first_error)
    for (span, &real) in bad.iter().zip(&first_error) {
        let mut score = vec![0u32; SPAN];
        let (mut lo, mut hi) = (0usize, SPAN);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let mean = |a: usize, b: usize| -> f64 {
                span[a..b].iter().map(|o| o.conf as f64).sum::<f64>() / (b - a) as f64
            };
            let (left, right) = (mean(lo, mid), mean(mid, hi));
            if left <= right { hi = mid } else { lo = mid }
            for s in score.iter_mut().take(hi).skip(lo) { *s += 1; }
        }
        b_pred.push(lo);
        for (k, &s) in score.iter().enumerate() {
            b_scores_pool.push((s as f64, k == real));
        }
    }

    // (c) change-point: the token with the lowest `conf` in the span. Score = 1-conf.
    let mut c_pred = Vec::new();
    let mut c_scores_pool: Vec<(f64, bool)> = Vec::new();
    for (span, &real) in bad.iter().zip(&first_error) {
        let (mut worst_i, mut worst_v) = (0usize, f32::INFINITY);
        for (k, o) in span.iter().enumerate() {
            if o.conf < worst_v { worst_v = o.conf; worst_i = k; }
            c_scores_pool.push(((1.0 - o.conf) as f64, k == real));
        }
        c_pred.push(worst_i);
    }

    let auroc_b = auroc(&b_scores_pool);
    let auroc_c = auroc(&c_scores_pool);
    let dist_b = mean_dist(&b_pred, &first_error);
    let dist_c = mean_dist(&c_pred, &first_error);
    let dist_random = SPAN as f64 / 3.0; // approx expected value of |U(0,15) - U(0,15)|

    println!("=== first-error localization, {} bad spans ===", bad.len());
    println!("  strategy           AUROC   mean dist.   random dist. (floor)");
    println!("  (b) bisection       {:.3}      {:5.2}            {:5.2}", auroc_b, dist_b, dist_random);
    println!("  (c) change-point    {:.3}      {:5.2}            {:5.2}", auroc_c, dist_c, dist_random);

    // Payoff: redo from the predicted point to the end of the span.
    let payoff = |preds: &[usize]| -> (f64, f64) {
        let frac: f64 = preds.iter().map(|&p| (SPAN - p) as f64 / SPAN as f64).sum::<f64>() / preds.len() as f64;
        let recall: f64 = preds.iter().zip(&first_error)
            .filter(|(&p, &r)| p <= r).count() as f64 / preds.len() as f64;
        (frac, recall)
    };
    let (frac_b, rec_b) = payoff(&b_pred);
    let (frac_c, rec_c) = payoff(&c_pred);

    println!("\n=== payoff: redoing only from the predicted point to the end ===");
    println!("  (a) whole span       saves   0.0% of the span | covers 100.0% of the errors (trivial)");
    println!("  (b) bisection        saves {:5.1}% of the span | covers {:5.1}% of the errors", 100.0 * frac_b, 100.0 * rec_b);
    println!("  (c) change-point     saves {:5.1}% of the span | covers {:5.1}% of the errors", 100.0 * frac_c, 100.0 * rec_c);

    println!("\n=== VERDICT ===");
    if auroc_c > auroc_b + 0.03 {
        println!("  per-token change-point clearly wins -- halving bisection is the wrong");
        println!("  object, as the literature predicted. The bisector doesn't get built.");
    } else if auroc_b > auroc_c + 0.03 {
        println!("  halving bisection wins -- worth pursuing, against the literature's");
        println!("  prediction. Worth re-measuring with another seed before building.");
    } else {
        println!("  they tie within noise -- neither localizes much better than the other.");
    }
    if auroc_b.max(auroc_c) < 0.6 {
        println!("  and neither pulls far from 0.5 (chance): localizing the FIRST token of");
        println!("  the error may not be a cheap problem, even though detecting the span was.");
    }
}

fn mean_dist(pred: &[usize], real: &[usize]) -> f64 {
    pred.iter().zip(real).map(|(&p, &r)| (p as f64 - r as f64).abs()).sum::<f64>() / pred.len() as f64
}

/// Rank-based AUROC (Mann-Whitney), over (score, is_positive) pairs.
fn auroc(pairs: &[(f64, bool)]) -> f64 {
    let mut v: Vec<(f64, bool)> = pairs.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = v.len() as f64;
    let n_pos = v.iter().filter(|(_, p)| *p).count() as f64;
    let n_neg = n - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    // Average rank for ties.
    let mut pos_rank_sum = 0.0f64;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 { j += 1; }
        let mid_rank = (i + 1 + j) as f64 / 2.0; // 1-based ranks, block average
        for k in i..j {
            if v[k].1 { pos_rank_sum += mid_rank; }
        }
        i = j;
    }
    (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}
