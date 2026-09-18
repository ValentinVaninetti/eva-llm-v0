//! "Confident and wrong" -- the inverse side of
//! "that it bets." The safe zone (p[argmax] >= 0.8) doesn't get touched by
//! the gate and gives 0.0000 difference -- but it still has errors (0.79
//! bpb isn't zero) and nobody measured WHERE they fall.
//!
//! the hypothesis: confident errors concentrate where the context is
//! RARE in the table (low count) -- confidence that exceeds coverage. If
//! the "high confidence x rare context" cell has the SAME precision as
//! "high confidence x common context", there's no overreach and the point
//! closes. If it stands out, the "curious gate" is born.
//!
//! A single pass: MODEL precision (not the table's) cross-tabbed by
//! p[argmax] decile x context count-band (miss / c=2 / 3-9 / 10-49 / 50+).
//! Does NOT touch `src/`: only `forward_hidden`, `classify_row`,
//! `Recall::lookup_detail`, all public, zero new op.
//!
//! Entropy-by-count: the theory to
//! explain the reversal above with a single cause -- high count might not
//! mean "common, reliable context" but rather "low-information context"
//! (a generic fragment with more valid continuations). Measured directly:
//! Shannon H(q) over the distribution `lookup_detail` already returns
//! (re-aggregation, no new op), by count band. The number that kills it:
//! if H doesn't grow with count, the theory falls right here.
//!
//! H(p), the Shannon entropy of the MODEL's output distribution
//! (not the table's -- that's H(q), above). It's the free, single-pass
//! approximation to "if I were made to generate again, how much would I
//! disagree with myself" -- the cost of a real multi-sample approach
//! (self-consistency/debate), without paying for it. The number that
//! kills it: if H(p) doesn't separate anything `p[argmax]` doesn't
//! already separate, the cheap "deliberation" line closes. Re-aggregation
//! over already-computed logits, zero new op.

use eva_llm_v0::bet::classify_row;
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::save::load_model;

const CONFIDENCE_FLOOR: f32 = 0.8; // same floor as adaptive_lambda.rs

struct Pos {
    p_argmax: f32,
    correct: bool,
    band: &'static str,
    /// H(q) in bits over the table's continuation distribution at this
    /// position. None if there was no hit (band "miss": no q).
    entropy: Option<f32>,
    /// H(p) in bits over the MODEL's output distribution. Always
    /// available -- doesn't depend on the table having answered.
    h_p: f32,
}

/// Shannon entropy in bits (log2), 0*log2(0) := 0 by convention.
fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

/// H(p) directly from raw logits: softmax + Shannon, no new op.
fn logits_entropy_bits(row: &[f32]) -> f32 {
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&z| (z - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    shannon_bits(&probs)
}

/// Rank-based AUROC (Mann-Whitney), same method as `table_probe.rs`.
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
        for k in i..j {
            if v[k].1 { pos_rank_sum += mid_rank; }
        }
        i = j;
    }
    (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

const BANDS: [&str; 5] = ["miss", "c=2", "3-9", "10-49", "50+"];

fn count_band(hit: Option<(Vec<f32>, u32)>) -> &'static str {
    match hit {
        None => "miss",
        Some((_, 2)) => "c=2",
        Some((_, c)) if c <= 9 => "3-9",
        Some((_, c)) if c <= 49 => "10-49",
        Some(_) => "50+",
    }
}

fn collect(
    model: &eva_llm_v0::model::EvaModel,
    ds: &TextDataset,
    table: &Recall,
    windows: &[usize],
    seq: usize,
) -> Vec<Pos> {
    let vocab = model.cfg.vocab;
    let mut out = Vec::new();
    for &wi in windows {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_hidden(&input);
        for t in 0..input.len() {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let (p_argmax, _, correct, _) = classify_row(row, target[t]);
            let h_p = logits_entropy_bits(row);

            // Same indexing as Step 0 / table_probe, already fixed once.
            let global_pos = wi * seq + t + 1;
            let from = global_pos.saturating_sub(8);
            let ctx = &ds.ids[from..global_pos];
            let hit = if ctx.is_empty() { None } else { table.lookup_detail(ctx, vocab) };
            let entropy = hit.as_ref().map(|(q, _)| shannon_bits(q));
            let band = count_band(hit);

            out.push(Pos { p_argmax, correct, band, entropy, h_p });
        }
    }
    out
}

/// Decile 0..=9 for p in [0,1]; 1.0 falls into decile 9.
fn decile(p: f32) -> usize {
    ((p * 10.0).floor() as usize).min(9)
}

fn crosstab(positions: &[Pos]) -> std::collections::HashMap<(usize, &'static str), (usize, usize)> {
    let mut m: std::collections::HashMap<(usize, &'static str), (usize, usize)> = std::collections::HashMap::new();
    for p in positions {
        let e = m.entry((decile(p.p_argmax), p.band)).or_insert((0, 0));
        e.1 += 1;
        if p.correct {
            e.0 += 1;
        }
    }
    m
}

fn print_table(name: &str, positions: &[Pos]) {
    let m = crosstab(positions);
    println!("\n=== {name}: model precision, p[argmax] decile x count band ===");
    print!("  decile   ");
    for b in BANDS {
        print!("{b:>14}");
    }
    println!();
    for d in 0..10 {
        print!("  [{:.1},{:.1})", d as f32 / 10.0, (d + 1) as f32 / 10.0);
        for b in BANDS {
            match m.get(&(d, b)) {
                Some(&(correct, n)) if n > 0 => {
                    print!("  {:5.1}%(n={n:<5})", 100.0 * correct as f32 / n as f32);
                }
                _ => print!("  {:>13}", "-"),
            }
        }
        println!();
    }
}

/// The number that kills it: within the safe zone (p[argmax]>=0.8,
/// deciles 8-9), compares precision on rare context (c=2) vs common
/// context (10-49, 50+).
fn number_that_kills_it(name: &str, positions: &[Pos]) {
    let safe: Vec<&Pos> = positions.iter().filter(|p| p.p_argmax >= CONFIDENCE_FLOOR).collect();
    let n_total = safe.len();
    let precision = |band: &str| -> Option<(f32, usize)> {
        let sub: Vec<&&Pos> = safe.iter().filter(|p| p.band == band).collect();
        if sub.is_empty() {
            return None;
        }
        let c = sub.iter().filter(|p| p.correct).count();
        Some((c as f32 / sub.len() as f32, sub.len()))
    };
    println!("\n=== {name}: THE NUMBER -- safe zone (p[argmax]>={CONFIDENCE_FLOOR}), n={n_total} ===");
    for b in BANDS {
        match precision(b) {
            Some((p, n)) => println!("  band {b:<6}  precision {:5.1}%  (n={n})", 100.0 * p),
            None => println!("  band {b:<6}  no data"),
        }
    }
    if let (Some((p_rare, n_rare)), Some((p_common, n_common))) = (precision("c=2"), precision("50+")) {
        let delta = p_common - p_rare;
        println!("\n  delta precision (50+ minus c=2): {:+.1} points  (n_rare={n_rare}, n_common={n_common})", 100.0 * delta);
        if n_rare < 30 || n_common < 30 {
            println!("  WARNING: at least one cell has n<30 -- not enough to decide alone.");
        }
        if delta > 0.03 {
            println!("  Common context has clearly higher precision in the safe zone --");
            println!("  there's overreach: high confidence with rare context is LESS reliable.");
            println!("  Worth building the curious gate.");
        } else if delta < -0.03 {
            println!("  RARE context has higher precision (surprise, against the hypothesis) --");
            println!("  no overreach in the expected direction. Review before acting.");
        } else {
            println!("  No clear difference -- no measurable overreach. The safe zone is");
            println!("  already nearly an even match, and the topic closes with this result.");
        }
    } else {
        println!("\n  Missing at least one of the c=2 / 50+ bands within the safe zone -- the");
        println!("  delta cannot be computed with this cut.");
    }
}

/// The theory, measured directly: mean H(q) per count band. If it
/// grows with count, "high count" is low-information context (more valid
/// continuations), which would explain why both the table's bar and the
/// model's precision drop with high count. If it doesn't grow, the theory
/// falls right here.
fn entropy_by_band(name: &str, positions: &[Pos]) {
    let vocab_bits = 8.0f32; // theoretical ceiling: log2(256)
    println!("\n=== {name}: Shannon H(q) (bits) by count band, ceiling={vocab_bits} bits ===");
    let mut means = Vec::new();
    for b in ["c=2", "3-9", "10-49", "50+"] {
        let vals: Vec<f32> = positions.iter()
            .filter(|p| p.band == b)
            .filter_map(|p| p.entropy)
            .collect();
        if vals.is_empty() {
            println!("  band {b:<6}  no data");
            continue;
        }
        let n = vals.len();
        let mean = vals.iter().sum::<f32>() / n as f32;
        let mut sorted = vals.clone();
        sorted.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let median = sorted[n / 2];
        println!("  band {b:<6}  mean H {mean:.3} bits  |  median H {median:.3}  |  n={n}");
        means.push((b, mean));
    }

    println!("\n=== {name}: THE NUMBER -- does H grow with count? ===");
    if let (Some(&(_, h_rare)), Some(&(_, h_common))) =
        (means.iter().find(|(b, _)| *b == "c=2"), means.iter().find(|(b, _)| *b == "50+"))
    {
        let delta = h_common - h_rare;
        println!("  delta H (50+ minus c=2): {delta:+.3} bits");
        let monotonic = means.windows(2).all(|w| w[1].1 >= w[0].1 - 0.02);
        println!("  non-decreasing monotonic c=2->3-9->10-49->50+: {monotonic}");
        if delta > 0.1 && monotonic {
            println!("  H grows with count, monotonically -- confirms the theory: high count is");
            println!("  low-information context (more valid continuations), explains both");
            println!("  anomalies (table's bar and model's precision dropping with count).");
        } else if delta > 0.1 {
            println!("  H grows end to end but is NOT monotonic -- partial signal, not the clean");
            println!("  theory. Note it, don't close it as confirmed.");
        } else {
            println!("  H does NOT clearly grow with count -- the theory FALLS here. The");
            println!("  \"confident and wrong\" reversal stays unexplained, another cause is needed.");
        }
    } else {
        println!("  Missing a band to compute the delta.");
    }
}

/// Closing the "cheap deliberation" thread: does H(p) predict model
/// correctness better, or at least IN ADDITION to, what p[argmax] already
/// predicts for free? If not, the single-pass approximation to "how much
/// would I disagree with myself" adds nothing confidence doesn't already
/// say, and the cheap line closes.
fn hp_vs_pargmax(name: &str, positions: &[Pos]) {
    let pool_pargmax: Vec<(f64, bool)> = positions.iter().map(|p| (p.p_argmax as f64, p.correct)).collect();
    // score = -H(p): lower entropy has to correlate with correctness, same
    // direction as p[argmax].
    let pool_hp: Vec<(f64, bool)> = positions.iter().map(|p| (-p.h_p as f64, p.correct)).collect();

    let auc_pargmax = auroc(&pool_pargmax);
    let auc_hp = auroc(&pool_hp);

    println!("\n=== {name}: THE NUMBER -- H(p) vs p[argmax], n={} ===", positions.len());
    println!("  AUROC p[argmax] (free, already published today):        {auc_pargmax:.3}");
    println!("  AUROC -H(p) (free approx. to \"would disagree with myself\"): {auc_hp:.3}");
    println!("  (the two global AUROCs barely separate -- they're glued together by");
    println!("  construction, a peaked distribution has both high p[argmax] AND low H(p)");
    println!("  at once. The question that matters is the conditional one, below.)");

    // Residual: within the safe zone (where the expensive errors matter),
    // split by H(p) median -- if p[argmax] already ate all the signal,
    // both halves should come out with matching precision. THIS, not the
    // global AUROC above, is the number that decides the verdict.
    let safe: Vec<&Pos> = positions.iter().filter(|p| p.p_argmax >= CONFIDENCE_FLOOR).collect();
    let mut conditional_delta: Option<(f32, usize, usize)> = None;
    if safe.len() >= 20 {
        let mut hs: Vec<f32> = safe.iter().map(|p| p.h_p).collect();
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_h = hs[hs.len() / 2];
        let (low, high): (Vec<&&Pos>, Vec<&&Pos>) = safe.iter().partition(|p| p.h_p <= median_h);
        let precision = |g: &[&&Pos]| -> f32 {
            if g.is_empty() { return f32::NAN; }
            g.iter().filter(|p| p.correct).count() as f32 / g.len() as f32
        };
        let (p_low, p_high) = (precision(&low), precision(&high));
        println!("\n  within the safe zone (p[argmax]>={CONFIDENCE_FLOOR}), split by median H(p)={median_h:.3}:");
        println!("    low H(p) (more deterministic)   precision {:5.1}%  (n={})", 100.0 * p_low, low.len());
        println!("    high H(p) (less deterministic)  precision {:5.1}%  (n={})", 100.0 * p_high, high.len());
        println!("    delta: {:+.1} points", 100.0 * (p_high - p_low));
        conditional_delta = Some((p_low - p_high, low.len(), high.len()));
    }

    println!("\n=== {name}: VERDICT (on the CONDITIONAL delta, not the global AUROC) ===");
    match conditional_delta {
        Some((delta, n_low, n_high)) if delta > 0.03 && n_low >= 30 && n_high >= 30 => {
            println!("  At equal confidence, H(p) separates {:.1} points of precision (n={n_low}/{n_high}).", 100.0 * delta);
            println!("  H(p) adds something to p[argmax] that p[argmax] alone doesn't see. First");
            println!("  real cheap justification to consider the expensive version (real");
            println!("  multi-sample). Doesn't close the deliberation line, it opens it.");
        }
        _ => {
            println!("  No clear conditional difference (or insufficient n) -- H(p) adds nothing");
            println!("  p[argmax] didn't already have. Closes the cheap deliberation line.");
        }
    }
}

/// 2D gate, approved independently ("the ball you left"): a narrower safe zone
/// than `number_that_kills_it`'s (which only uses p[argmax]) -- it ALSO
/// requires the shape of the rest of the distribution to be deterministic
/// (H(p) below its own median within the safe zone). Formalizes as its own
/// measurement what came out as a byproduct in `hp_vs_pargmax`.
fn gate_2d(name: &str, positions: &[Pos]) {
    let safe: Vec<&Pos> = positions.iter().filter(|p| p.p_argmax >= CONFIDENCE_FLOOR).collect();
    if safe.len() < 20 {
        println!("\n=== {name}: 2D gate -- insufficient n in the safe zone ===");
        return;
    }
    let mut hs: Vec<f32> = safe.iter().map(|p| p.h_p).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_h = hs[hs.len() / 2];

    let precision_of = |g: &[&Pos]| -> f32 {
        if g.is_empty() { return f32::NAN; }
        g.iter().filter(|p| p.correct).count() as f32 / g.len() as f32
    };
    let precision_1d = precision_of(&safe);
    let narrow: Vec<&Pos> = safe.iter().filter(|p| p.h_p <= median_h).copied().collect();
    let precision_2d = precision_of(&narrow);

    println!("\n=== {name}: 2D gate -- p[argmax]>={CONFIDENCE_FLOOR} AND H(p)<={median_h:.3} ===");
    println!("  1D safe zone (p[argmax] only):          precision {:5.1}%  (n={})", 100.0 * precision_1d, safe.len());
    println!("  2D safe zone (+H(p)<=median):            precision {:5.1}%  (n={}, {:.0}% of 1D)",
        100.0 * precision_2d, narrow.len(), 100.0 * narrow.len() as f32 / safe.len() as f32);
    println!("  precision gain from narrowing: {:+.1} points, at a cost of {:.0}% less coverage",
        100.0 * (precision_2d - precision_1d), 100.0 * (1.0 - narrow.len() as f32 / safe.len() as f32));
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

    // Same cut as the rest of the round: half of val to look at (dev), the
    // other half gets touched exactly once for the final number.
    let half = n_train + (n_val / 2);
    let dev_windows: Vec<usize> = (n_train..half).collect();
    let val_windows: Vec<usize> = (half..n).collect();
    println!("eva confident_wrong: {} dev windows | {} val windows", dev_windows.len(), val_windows.len());

    let pos_dev = collect(&model, &ds, &table, &dev_windows, seq);
    print_table("DEV (looking, not deciding)", &pos_dev);
    number_that_kills_it("DEV", &pos_dev);
    entropy_by_band("DEV (looking, not deciding)", &pos_dev);
    hp_vs_pargmax("DEV (looking, not deciding)", &pos_dev);
    gate_2d("DEV (looking, not deciding)", &pos_dev);

    let pos_val = collect(&model, &ds, &table, &val_windows, seq);
    print_table("VAL (touched once)", &pos_val);
    number_that_kills_it("VAL", &pos_val);
    entropy_by_band("VAL (touched once)", &pos_val);
    hp_vs_pargmax("VAL (touched once)", &pos_val);
    gate_2d("VAL (touched once)", &pos_val);
}
