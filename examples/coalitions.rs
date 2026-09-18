//! Requested independently: all 32 possible combinations of active blocks (2^5,
//! including empty and full), bpb of each, on A and B -- to compare the
//! whole SURFACE, not just individual blocks or a single pair. Reuses
//! `forward_skips` (already public, the same one `ceiling` uses), zero new
//! ops, nothing trained.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::save::load_model;

fn bpb_with_skip(model: &eva_llm_v0::model::EvaModel, ds: &TextDataset, from: usize, to: usize, skips: &[usize]) -> f32 {
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let mut nats = 0.0f64;
    let mut n_pos = 0usize;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_skips(&input, skips);
        for t in 0..seq {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&z| (z - mx).exp()).sum();
            let p = (row[target[t]] - mx).exp() / sum;
            nats -= (p as f64).max(1e-12).ln();
            n_pos += 1;
        }
    }
    (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("could not load the checkpoint");
    let n_blocks = model.blocks.len();
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    println!("eva coalitions: {weights} | {n_blocks} blocks | {n_val} validation windows\n");
    println!("  {:<12} {:>10}  {:>10}", "active", "bpb", "delta vs full");

    let mut base = 0.0f32;
    let total = 1usize << n_blocks;
    let mut rows: Vec<(usize, u32, f32)> = Vec::new(); // (active_mask, n_active, bpb)

    for mask in 0..total {
        // mask = bits of ACTIVE blocks (1=active). skips = the disabled ones.
        let skips: Vec<usize> = (0..n_blocks).filter(|&i| mask & (1 << i) == 0).collect();
        let bpb = if skips.is_empty() {
            bpb_with_skip(&model, &ds, n_train, n, &[])
        } else {
            bpb_with_skip(&model, &ds, n_train, n, &skips)
        };
        if mask == total - 1 {
            base = bpb;
        }
        rows.push((mask, (mask as u32).count_ones(), bpb));
    }

    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0))); // most active first
    for (mask, n_active, bpb) in &rows {
        let active: Vec<String> = (0..n_blocks).filter(|&i| mask & (1 << i) != 0).map(|i| i.to_string()).collect();
        let label = if active.is_empty() { "none".to_string() } else { active.join(",") };
        println!("  [{:<10}] {:>10.4}  {:>+10.4}  (n_active={n_active})", label, bpb, bpb - base);
    }

    println!("\neva coalitions: full (5 active) = {base:.4} bpb -- reference for the deltas above");
}
