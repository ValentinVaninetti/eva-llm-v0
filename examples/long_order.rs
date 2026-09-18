//! Is the table's bottleneck
//! CONTEXT (order 8 cuts off information that would disambiguate further
//! back) or DATA (even looking further back, the corpus genuinely has more
//! than one valid continuation)? This separates them by re-building the
//! table with longer orders (16 and 32) and checking whether the bar goes
//! up / H(q) goes down over the SAME query population as the usual chain
//! (8,6,4,3,2).
//!
//! Does NOT touch `src/recall.rs`: `ORDERS` is capped at 8 there on
//! purpose (packs into a u64 without collisions). For 16 and 32 a longer
//! key is needed -- here an exact `Vec<u8>` is used as the `HashMap` key
//! (no information loss, no need to pack into an integer: two different
//! contexts are never the same key). Minimal, local reimplementation of
//! the same backoff `Recall` uses, only for this measurement. No model:
//! "bar" and H(q) are properties of the table alone.

use eva_llm_v0::data::TextDataset;
use std::collections::HashMap;

const MIN_COUNT: u32 = 2; // same threshold as src/recall.rs (private there)
const SEQ: usize = 64; // same seq as the rest of today's round, for the same cut

fn pack(ctx: &[usize]) -> Vec<u8> {
    ctx.iter().map(|&b| b as u8).collect()
}

/// context bytes -> (next byte, count), one table per order.
type Table = HashMap<Vec<u8>, Vec<(u8, u32)>>;

fn build_table(train: &[usize], orders: &[usize]) -> Vec<Table> {
    orders.iter().map(|&k| {
        let mut t: Table = HashMap::new();
        if train.len() > k {
            for i in 0..train.len() - k {
                let key = pack(&train[i..i + k]);
                let next = train[i + k] as u8;
                let e = t.entry(key).or_default();
                match e.iter_mut().find(|(b, _)| *b == next) {
                    Some((_, c)) => *c += 1,
                    None => e.push((next, 1)),
                }
            }
        }
        t
    }).collect()
}

/// Same as `Recall::find`: falls back to lower orders until it finds one
/// with enough occurrences. Also returns which order answered, for the
/// coverage breakdown.
fn lookup<'a>(
    tables: &'a [Table],
    orders: &[usize],
    ctx: &[usize],
) -> Option<(&'a Vec<(u8, u32)>, usize)> {
    for (ti, &k) in orders.iter().enumerate() {
        if ctx.len() < k {
            continue;
        }
        let key = pack(&ctx[ctx.len() - k..]);
        let Some(hits) = tables[ti].get(&key) else { continue };
        let total: u32 = hits.iter().map(|(_, c)| *c).sum();
        if total < MIN_COUNT {
            continue;
        }
        return Some((hits, k));
    }
    None
}

fn distribution(hits: &[(u8, u32)], vocab: usize) -> Vec<f32> {
    let total: u32 = hits.iter().map(|(_, c)| *c).sum();
    let mut p = vec![0.0f32; vocab];
    let inv = 1.0 / total as f32;
    for &(b, c) in hits {
        p[b as usize] = c as f32 * inv;
    }
    p
}

fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

fn evaluate(name: &str, orders: &[usize], tables: &[Table], ids: &[usize], queries: &[usize], vocab: usize) {
    let mut n_hit = 0usize;
    let mut n_correct = 0usize;
    let mut h_sum = 0.0f64;
    let mut coverage: HashMap<usize, usize> = HashMap::new();

    let max_order = *orders.iter().max().unwrap();
    for &i in queries {
        let ctx = &ids[i - max_order..i];
        match lookup(tables, orders, ctx) {
            Some((hits, order_used)) => {
                n_hit += 1;
                *coverage.entry(order_used).or_insert(0) += 1;
                let dist = distribution(hits, vocab);
                let argmax = dist.iter().enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (k, &v)| if v > bv { (k, v) } else { (bi, bv) }).0;
                if argmax == ids[i] {
                    n_correct += 1;
                }
                h_sum += shannon_bits(&dist) as f64;
            }
            None => {
                *coverage.entry(0).or_insert(0) += 1; // 0 = miss
            }
        }
    }

    let bar = 100.0 * n_correct as f32 / n_hit.max(1) as f32;
    let mean_h = h_sum / n_hit.max(1) as f64;
    println!("\n=== {name} (orders {orders:?}) ===");
    println!("  coverage: {n_hit}/{} ({:.1}%) -- rest miss", queries.len(), 100.0 * n_hit as f32 / queries.len() as f32);
    println!("  bar (argmax==truth, over hits): {bar:.1}%");
    println!("  mean H(q) (bits, over hits): {mean_h:.3}");
    print!("  breakdown of hits by order:");
    let mut ords: Vec<usize> = orders.to_vec();
    ords.push(0);
    for &o in &ords {
        if let Some(&n) = coverage.get(&o) {
            let label = if o == 0 { "miss".to_string() } else { o.to_string() };
            print!("  {label}={n}({:.1}%)", 100.0 * n as f32 / queries.len() as f32);
        }
    }
    println!();
}

fn main() {
    let data = std::env::args().nth(1).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;
    let vocab = 256usize;

    let ds = TextDataset::from_file(&data, SEQ).expect("could not read the corpus");
    let n_windows = ds.num_windows();
    let n_val = (((n_windows as f32) * val).round() as usize).clamp(1, n_windows / 2);
    let n_train = n_windows - n_val;
    let cut = n_train * SEQ;
    let train: Vec<usize> = ds.ids[..cut].to_vec();

    // Queries: same fixed set for all three tables, valid for the maximum
    // order (32) -- that way the comparison is even, the population doesn't
    // change.
    let max_order = 32;
    let queries: Vec<usize> = (cut.max(max_order)..ds.ids.len()).collect();
    println!("eva long_order: {} train bytes | {} validation queries (all with >=32 bytes of context)",
        train.len(), queries.len());

    let base = [8usize, 6, 4, 3, 2];
    let ext16 = [16usize, 8, 6, 4, 3, 2];
    let ext32 = [32usize, 16, 8, 6, 4, 3, 2];

    let t_base = build_table(&train, &base);
    let t_16 = build_table(&train, &ext16);
    let t_32 = build_table(&train, &ext32);

    evaluate("BASE (order 8, the usual chain)", &base, &t_base, &ds.ids, &queries, vocab);
    evaluate("EXTENDED order 16", &ext16, &t_16, &ds.ids, &queries, vocab);
    evaluate("EXTENDED order 32", &ext32, &t_32, &ds.ids, &queries, vocab);

    println!("\n=== VERDICT -- context bottleneck or data bottleneck? ===");
    println!("  If order 16/32 almost never answers (coverage ~0%) and the global");
    println!("  bar/H(q) don't move: the corpus is too small for a longer context");
    println!("  to repeat -- there's nothing to disambiguate with, even if in");
    println!("  principle the information existed. That's a DATA limit (corpus");
    println!("  size), not necessarily that 'more context wouldn't help' on a");
    println!("  bigger corpus. If instead order 16/32 DOES answer with non-trivial");
    println!("  coverage and there the bar rises / H(q) drops clearly relative to");
    println!("  what order 8 answered at those same positions, the bottleneck was");
    println!("  context: the table was throwing away information that would have");
    println!("  disambiguated.");
}
