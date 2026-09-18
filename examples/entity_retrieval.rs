//! Entity retrieval benchmark on real text (2026-08-15).
//!
//! The Quijote T=1 vs T=1.3 result (1.716 vs 1.720 global bpb) cannot separate
//! "does the long-range memory help?" from "does a 13M model learn language?".
//! This benchmark is the directed alternative we proposed: it plants a
//! controlled but semantically natural signal -- proper names/entities that
//! reappear -- and measures whether the model assigns higher probability to an
//! entity's bytes when it has already seen that entity, as a function of the
//! distance (in bytes) back to the previous occurrence.
//!
//! Metric per (entity, distance bucket), measured at the entity's FIRST byte:
//!   n        : occurrences measured
//!   top1/top5: % where the correct next byte is in the model's top-k
//!   mean_lnp : mean SURPRISAL (nats) of the correct byte -- this is
//!              `-log p`, computed below as
//!              `-((row[first_byte] - m) - ln(esum))`. **LOWER IS BETTER.**
//!              The name is a historical misnomer kept for log compatibility:
//!              it is a negative log-probability, not a log-probability.
//!   Δlnp     : mean_lnp of this bucket minus the same entity's reference
//!              (nowin in no-carry mode, cold in carry mode).
//!              **NEGATIVE = the prior occurrence in memory HELPED** (less
//!              surprised than with no memory available) -> retrieval signal.
//!              Positive = worse than having no memory at all.
//!
//!              SIGN CORRECTION (2026-08-25): this header previously said
//!              "Positive = ... helped", which is backwards for a surprisal.
//!              PAPER-DRAFT.md 4.8 inherited that error and reported the
//!              real-text retrieval result with its conclusion inverted. The
//!              sign is settled by two quantities in the same table that do
//!              not depend on it: at 32<d<=128 the model scores top1 50.00%
//!              and rank 6.9 against a no-memory reference of 34.94% / 9.8,
//!              i.e. it is BETTER there -- and that is the bucket whose Δlnp
//!              is -0.829. Negative Δlnp means better.
//!   rank     : mean rank of the correct byte
//!   name_bpb : per-byte bpb over the whole entity (secondary; mixes short
//!              range once the first byte is right)
//!
//! References:
//!   nowin (no-carry only): a prior occurrence exists but OUTSIDE the current
//!     window, so the model cannot see it -- the honest ">512 with no memory"
//!     baseline for the architecture AS TRAINED (windows are independent).
//!   cold : the first occurrence of the entity in the whole corpus -- pure
//!     no-memory baseline for that entity (small n by construction).
//!
//! Modes:
//!   default          : windows independent, exactly like training (distances
//!                      capped by seq; >512 falls into `nowin`).
//!   EVA_ENTITY_CARRY=1: ClockMem state carried across windows
//!                      (`EvaModel::forward_carrying`), so distances can reach
//!                      512/1024+. NOTE: the model was trained with
//!                      independent windows; carried state is out of
//!                      distribution, so carry-mode numbers are exploratory.
//!
//! USAGE: cargo run --release --example entity_retrieval -- <weights> <data>
//!   Set EVA_ALPHA_TEMP to match the checkpoint (1.0 for *_T1, 1.3 for *_T13).
//!   env: EVA_ENTITY_CARRY=1, EVA_ENTITY_MIN=n (default 30),
//!       EVA_ENTITY_N=n (default 12, reported individually),
//!       EVA_ENTITY_TRAIN=1 (also evaluate the train region).
//!   The headline is VALIDATION (last 10% contiguous windows, same cut as
//!   mask_eval and train).

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::Tensor;
use std::collections::HashMap;

fn env_usize(name: &str, dflt: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(dflt)
}

/// A "letter" byte for word segmentation: ASCII alpha or a UTF-8 byte that
/// belongs to a Spanish accented character (lead 0xC0-0xDF + continuation
/// 0x80-0xBF). Keeps words like "días" together without gluing on em dashes.
fn is_letter(b: u8) -> bool {
    (b'A'..=b'Z').contains(&b)
        || (b'a'..=b'z').contains(&b)
        || (0xC0..=0xDF).contains(&b)
        || (0x80..=0xBF).contains(&b)
}

struct Entity {
    bytes: Vec<u8>,
    occ: Vec<usize>,
    cap: usize,
    ratio: f32,
}

fn find_occurrences(ids: &[usize], pat: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let (n, m) = (ids.len(), pat.len());
    if m == 0 || m > n {
        return out;
    }
    let mut i = 0;
    while i + m <= n {
        let mut ok = true;
        for k in 0..m {
            if ids[i + k] as u8 != pat[k] {
                ok = false;
                break;
            }
        }
        if ok {
            out.push(i);
            i += m;
        } else {
            i += 1;
        }
    }
    out
}

#[derive(Default, Clone)]
struct Stats {
    n: usize,
    ok1: usize,
    ok5: usize,
    sum_lnp: f64,
    sum_rank: f64,
    sum_name: f64,
    name_n: usize,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.n += o.n;
        self.ok1 += o.ok1;
        self.ok5 += o.ok5;
        self.sum_lnp += o.sum_lnp;
        self.sum_rank += o.sum_rank;
        self.sum_name += o.sum_name;
        self.name_n += o.name_n;
    }
    fn top1(&self) -> f32 {
        100.0 * self.ok1 as f32 / self.n.max(1) as f32
    }
    fn top5(&self) -> f32 {
        100.0 * self.ok5 as f32 / self.n.max(1) as f32
    }
    fn mean_lnp(&self) -> f32 {
        (self.sum_lnp / self.n.max(1) as f64) as f32
    }
    fn mean_rank(&self) -> f32 {
        (self.sum_rank / self.n.max(1) as f64) as f32
    }
    fn name_bpb(&self) -> f32 {
        (self.sum_name / self.name_n.max(1) as f64 / std::f64::consts::LN_2) as f32
    }
}

struct BucketInfo {
    label: &'static str,
    /// reference used for the Δlnp column: None => no reference column.
    is_ref: bool,
}

fn main() {
    let weights = std::env::args().nth(1).expect("usage: entity_retrieval <weights> <data>");
    let data = std::env::args().nth(2).expect("usage: entity_retrieval <weights> <data>");
    let carry = std::env::var("EVA_ENTITY_CARRY").is_ok();
    let min_count = env_usize("EVA_ENTITY_MIN", 30);
    let report_n = env_usize("EVA_ENTITY_N", 12);
    let also_train = std::env::var("EVA_ENTITY_TRAIN").is_ok();

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n_win = ds.num_windows();
    assert!(n_win > 0, "the corpus is too small for seq {seq}");
    let n_val = (((n_win as f32) * 0.1).round() as usize).clamp(1, n_win / 2);
    let n_train = n_win - n_val;

    let temp = std::env::var("EVA_ALPHA_TEMP").ok().unwrap_or_else(|| "1.0".into());
    let ssm = std::env::var("EVA_SSM_ALPHA").ok();
    println!("eva entity_retrieval: {weights} | {data}");
    println!("  seq={seq} windows: {n_train} train, {n_val} val (contiguous cut)");
    println!("  mode: {} | EVA_ALPHA_TEMP={temp} | min_count={min_count}{}",
        if carry { "carry (state across windows)" } else { "independent windows (as trained)" },
        if let Some(a) = &ssm { format!(" | EVA_SSM_ALPHA={a} (fixed clock)") } else { String::new() });

    let entities = detect_entities(&ds.ids, min_count);
    println!("\n  entities detected ({} pass the filter, top {} shown):", entities.len(), report_n.min(entities.len()));
    for (i, e) in entities.iter().enumerate().take(report_n) {
        println!("    {:>2}. {:20} count={:>5} ratio={:.2}",
            i + 1, String::from_utf8_lossy(&e.bytes), e.cap, e.ratio);
    }

    let n_buckets = if carry { 7 } else { 6 };
    let (buckets, ref_idx): (Vec<BucketInfo>, usize) = if carry {
        let b = vec![
            BucketInfo { label: "d <= 32", is_ref: false },
            BucketInfo { label: "32 < d <= 128", is_ref: false },
            BucketInfo { label: "128 < d <= 256", is_ref: false },
            BucketInfo { label: "256 < d <= 512", is_ref: false },
            BucketInfo { label: "512 < d <= 1024", is_ref: false },
            BucketInfo { label: "d > 1024", is_ref: false },
            BucketInfo { label: "cold (first occurrence)", is_ref: true },
        ];
        (b, 6)
    } else {
        let b = vec![
            BucketInfo { label: "d <= 32", is_ref: false },
            BucketInfo { label: "32 < d <= 128", is_ref: false },
            BucketInfo { label: "128 < d <= 256", is_ref: false },
            BucketInfo { label: "256 < d <= 512", is_ref: false },
            BucketInfo { label: "prior OUTSIDE window (no memory)", is_ref: true },
            BucketInfo { label: "cold (first occurrence)", is_ref: true },
        ];
        (b, 4)
    };

    evaluate("VAL", &model, &ds, &entities, n_train, n_win, n_buckets, &buckets, ref_idx, carry);
    if also_train {
        evaluate("TRAIN", &model, &ds, &entities, 0, n_train, n_buckets, &buckets, ref_idx, carry);
    }

    println!("\nnotes:");
    println!("  - mean_lnp is SURPRISAL (-log p), measured on the entity's FIRST byte.");
    println!("    LOWER IS BETTER. (The name is a historical misnomer.)");
    println!("  - Δlnp vs '{}': NEGATIVE means the model is LESS surprised by", buckets[ref_idx].label);
    println!("    the entity's byte when it saw the entity before at that distance,");
    println!("    i.e. negative = memory helped. Cross-check with top1% and rank,");
    println!("    which do not depend on this sign.");
    println!("  - {} mode has no 'outside window' reference: long-distance buckets are", if carry { "carry" } else { "no-carry" });
    println!("    the decay curve themselves (and 'cold' the no-memory extreme).");
    if !carry {
        println!("  - 'prior OUTSIDE window' is the de-facto >512 no-memory baseline for");
        println!("    the architecture as trained (windows independent).");
    }
    println!("  - temperature must match training (T=1.3 checkpoint -> EVA_ALPHA_TEMP=1.3).");
}

fn detect_entities(ids: &[usize], min_count: usize) -> Vec<Entity> {
    let mut cap: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut low: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut i = 0;
    while i < ids.len() {
        if !is_letter(ids[i] as u8) {
            i += 1;
            continue;
        }
        let start = i;
        while i < ids.len() && is_letter(ids[i] as u8) {
            i += 1;
        }
        let word: Vec<u8> = ids[start..i].iter().map(|&x| x as u8).collect();
        let first_upper = word[0].is_ascii_uppercase();
        let has_lower_after = word[1..].iter().any(|&b| b.is_ascii_lowercase());
        if word.len() >= 2 && first_upper && has_lower_after {
            *cap.entry(word.clone()).or_insert(0) += 1;
        }
        if word[0].is_ascii_lowercase() {
            *low.entry(word).or_insert(0) += 1;
        }
    }

    let mut list: Vec<(Vec<u8>, usize, f32)> = Vec::new();
    for (w, &c) in &cap {
        if *w.get(0).unwrap() > b'Z' {
            continue; // non-ASCII first byte cannot be a latin proper name here
        }
        let mut lw = w.clone();
        lw[0] = lw[0].to_ascii_lowercase();
        let lc = low.get(&lw).copied().unwrap_or(0);
        let ratio = c as f32 / (c + lc) as f32;
        if c >= min_count && ratio >= 0.9 {
            list.push((w.clone(), c, ratio));
        }
    }
    list.sort_by(|a, b| b.1.cmp(&a.1));

    list.into_iter()
        .map(|(bytes, cap, ratio)| Entity {
            occ: find_occurrences(ids, &bytes),
            bytes,
            cap,
            ratio,
        })
        .collect()
}

fn evaluate(
    tag: &str,
    model: &EvaModel,
    ds: &TextDataset,
    entities: &[Entity],
    from: usize,
    to: usize,
    n_buckets: usize,
    buckets: &[BucketInfo],
    ref_idx: usize,
    carry: bool,
) {
    let seq = ds.seq;
    let mut agg: Vec<Stats> = (0..n_buckets).map(|_| Stats::default()).collect();
    let mut per_entity: Vec<Vec<Stats>> = entities.iter().map(|_| {
        (0..n_buckets).map(|_| Stats::default()).collect()
    }).collect();

    let mut states = model.fresh_states();
    for wi in from..to {
        let base = wi * seq;
        let (input, _target) = ds.window(wi);
        let logits = if carry {
            model.forward_carrying(&input, &mut states)
        } else {
            model.forward(&input)
        };
        for (ei, e) in entities.iter().enumerate() {
            // occurrences of this entity inside this window: p in [base+1, base+seq)
            let lo = e.occ.partition_point(|&p| p < base + 1);
            for &p in e.occ[lo..].iter().take_while(|&&p| p < base + seq) {
                let ip = p - 1 - base; // input index predicting byte at absolute p
                let prev = {
                    let idx = e.occ.partition_point(|&q| q < p);
                    if idx == 0 {
                        None
                    } else {
                        Some(e.occ[idx - 1])
                    }
                };
                let b = match prev {
                    None => n_buckets - 1, // cold
                    Some(prev_p) => {
                        if carry {
                            let d = p - prev_p;
                            bucket_of_carry(d)
                        } else if prev_p >= base {
                            let d = p - prev_p;
                            bucket_of(d)
                        } else {
                            ref_idx // prior outside window: the no-memory reference
                        }
                    }
                };
                let mut st = Stats::default();
                measure(&mut st, &logits, ip, e.bytes[0], &e.bytes, base, p, seq);
                per_entity[ei][b].add(&st);
                agg[b].add(&st);
            }
        }
    }

    println!("\n=== {tag} (recurrent entity-first-byte positions) ===");
    println!("  {:<32} {:>5} {:>6} {:>6} {:>8} {:>8} {:>5} {:>8}",
        "bucket", "n", "top1%", "top5%", "mean_lnp", "Δlnp", "rank", "name_bpb");
    let ref_stats = &agg[ref_idx];
    for (i, bi) in buckets.iter().enumerate() {
        let s = &agg[i];
        let dlnp = if bi.is_ref {
            f32::NAN
        } else if ref_stats.n > 0 {
            s.mean_lnp() - ref_stats.mean_lnp()
        } else {
            f32::NAN
        };
        println!("  {:<32} {:>5} {:>6.2} {:>6.2} {:>8.3} {:>+8.3} {:>5.1} {:>8.3}",
            bi.label, s.n, s.top1(), s.top5(), s.mean_lnp(), dlnp, s.mean_rank(), s.name_bpb());
    }

    if !carry {
        println!("\n  per-entity detail (Δlnp vs that entity's 'outside window' reference):");
        println!("  {:<18} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "entity", "count", "d32", "d128", "d256", "d512", "nowin");
        for (ei, e) in entities.iter().enumerate().take(12) {
            let ref_s = &per_entity[ei][ref_idx];
            if ref_s.n == 0 {
                continue;
            }
            let deltas: Vec<String> = (0..n_buckets - 2)
                .map(|i| {
                    let s = &per_entity[ei][i];
                    if s.n == 0 {
                        "-".to_string()
                    } else {
                        format!("{:+6.3}", s.mean_lnp() - ref_s.mean_lnp())
                    }
                })
                .collect();
            println!("  {:<18} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}",
                String::from_utf8_lossy(&e.bytes), e.cap,
                deltas[0], deltas[1], deltas[2], deltas[3], ref_s.mean_lnp());
        }
    }
}

fn bucket_of(d: usize) -> usize {
    if d <= 32 {
        0
    } else if d <= 128 {
        1
    } else if d <= 256 {
        2
    } else {
        3
    }
}

fn bucket_of_carry(d: usize) -> usize {
    if d <= 32 {
        0
    } else if d <= 128 {
        1
    } else if d <= 256 {
        2
    } else if d <= 512 {
        3
    } else if d <= 1024 {
        4
    } else {
        5
    }
}

fn measure(st: &mut Stats, logits: &Tensor, ip: usize, first_byte: u8, name: &[u8], base: usize, p: usize, seq: usize) {
    let v = logits.shape[1];
    let row = &logits.data[ip * v..(ip + 1) * v];
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut esum = 0.0f64;
    for &x in row {
        esum += ((x - m) as f64).exp();
    }
    let lnp = -((row[first_byte as usize] - m) as f64 - esum.ln());
    st.n += 1;
    st.sum_lnp += lnp;

    let mut order: Vec<usize> = (0..v).collect();
    order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    let rank = order.iter().position(|&j| j == first_byte as usize).unwrap() + 1;
    st.sum_rank += rank as f64;
    if rank == 1 {
        st.ok1 += 1;
    }
    if rank <= 5 {
        st.ok5 += 1;
    }

    let l = name.len();
    if l >= 2 && p + l - 1 < base + seq {
        let mut s = 0.0f64;
        for k in 1..l {
            let r = &logits.data[(ip + k) * v..(ip + k + 1) * v];
            let mk = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut ek = 0.0f64;
            for &x in r {
                ek += ((x - mk) as f64).exp();
            }
            s += -((r[name[k] as usize] - mk) as f64 - ek.ln());
        }
        st.sum_name += s;
        st.name_n += 1;
    }
}
