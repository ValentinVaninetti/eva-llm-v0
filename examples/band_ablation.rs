//! Requested independently after confirming T=1.3: do the fast/medium/slow bands
//! serve DIFFERENT functions, or are they just equivalent parameters with
//! a different alpha? FULL band ablation (every channel of that band, in
//! ALL blocks at once, switched off in one shot) -- not channel by
//! channel.
//!
//! No new op and no change to `src/`: reimplements the block forward
//! (`EvaBlock::forward_from` in `src/model/block.rs`) by hand in this
//! example, using only public fields, inserting a mask over the mixer's
//! output before adding it to the residual. It's an evaluation pass (no
//! gradient), doesn't touch training.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::nn::Module;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::ops;
use eva_llm_v0::tensor::Tensor;

fn antisat() -> bool { std::env::var("EVA_ALPHA_ANTISAT").is_ok() }
fn temperature() -> f32 { std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0) }
fn alpha_of(lc: f32) -> f32 {
    if antisat() { 0.5 * (1.0 + lc / (1.0 + lc * lc).sqrt()) } else { 1.0 / (1.0 + (-lc / temperature()).exp()) }
}

/// Same shape as `EvaBlock::forward_from`, with a mask over the mixer's
/// output (zeroing specific channels before adding to the residual).
/// `mask[block][channel] = true` means "off".
fn forward_masked(model: &EvaModel, ids: &[usize], mask: &[Vec<bool>]) -> Tensor {
    let mut x = model.embed.embed(ids);
    if let Some(pos) = &model.pos {
        let n = ids.len().min(model.cfg.seq_len);
        x = ops::add(&x, &ops::slice_rows(pos, n));
    }
    for (i, b) in model.blocks.iter().enumerate() {
        let h = ops::add(&x, &b.conv.forward(&b.norm0.forward(&x)));
        let mut m = b.mixer.forward(&b.norm1.forward(&h));
        if mask[i].iter().any(|&v| v) {
            let (s, d) = (m.shape[0], m.shape[1]);
            let mut data = (*m.data).clone();
            for t in 0..s {
                for c in 0..d {
                    if mask[i][c] { data[t * d + c] = 0.0; }
                }
            }
            m = Tensor::new(data, vec![s, d]);
        }
        let h2 = ops::add(&h, &m);
        x = ops::add(&h2, &b.glu.forward(&b.norm2.forward(&h2)));
    }
    let x = model.norm_out.forward(&x);
    ops::matmul(&x, &model.head_w)
}

fn bpb_with_mask(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, mask: &[Vec<bool>]) -> f32 {
    let mut nats = 0.0f64;
    let mut n_pos = 0usize;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = forward_masked(model, &input, mask);
        nats += ops::cross_entropy(&logits, &target).data[0] as f64 * input.len() as f64;
        n_pos += input.len();
    }
    (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7_temp13.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("could not load the checkpoint");
    let n_blocks = model.blocks.len();
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    // Classify each channel, in each block, by band -- same bands as
    // always.
    let mut band: Vec<Vec<&'static str>> = Vec::new();
    let mut count = std::collections::HashMap::new();
    for bi in 0..n_blocks {
        let Mixer::Clock(clock) = &model.blocks[bi].mixer else { continue };
        let dim = clock.log_clock.data.len();
        let mut row = Vec::with_capacity(dim);
        for c in 0..dim {
            let a = alpha_of(clock.log_clock.data[c]);
            let b = if a < 0.05 { "fast" } else if a < 0.3 { "medium" } else { "slow" };
            row.push(b);
            *count.entry(b).or_insert(0usize) += 1;
        }
        band.push(row);
    }
    println!("eva band_ablation: {weights} | total count per band: {:?}\n", count);

    let empty_mask: Vec<Vec<bool>> = band.iter().map(|f| vec![false; f.len()]).collect();
    let base = bpb_with_mask(&model, &ds, n_train, n, &empty_mask);
    println!("bpb without ablation (control): {base:.4}\n");

    for target in ["fast", "medium", "slow"] {
        let mask: Vec<Vec<bool>> = band.iter().map(|f| f.iter().map(|&b| b == target).collect()).collect();
        let n_off: usize = mask.iter().map(|f| f.iter().filter(|&&x| x).count()).sum();
        let bpb = bpb_with_mask(&model, &ds, n_train, n, &mask);
        println!("turn off band '{target}' ({n_off} channels out of {}): bpb {bpb:.4}  delta={:+.4} ({:+.1}%)",
            band.len() * band[0].len(), bpb - base, 100.0 * (bpb - base) / base);
    }

    // Control: turn off a RANDOM number of channels the same size as
    // 'medium' (the largest band), to see whether the damage from a
    // specific band differs from turning off "any" group of that size.
    println!("\n=== control: same N of channels, chosen without looking at the band ===");
    let n_medium = *count.get("medium").unwrap_or(&0);
    let mut rng = eva_llm_v0::rng::Rng::new(99);
    let total_channels: usize = band.iter().map(|f| f.len()).sum();
    let mut all: Vec<(usize, usize)> = Vec::new();
    for (bi, f) in band.iter().enumerate() { for c in 0..f.len() { all.push((bi, c)); } }
    let mut idx: Vec<usize> = (0..all.len()).collect();
    rng.shuffle(&mut idx);
    let chosen = &idx[..n_medium.min(total_channels)];
    let mut random_mask: Vec<Vec<bool>> = band.iter().map(|f| vec![false; f.len()]).collect();
    for &e in chosen { let (bi, c) = all[e]; random_mask[bi][c] = true; }
    let random_bpb = bpb_with_mask(&model, &ds, n_train, n, &random_mask);
    println!("turn off {n_medium} RANDOM channels (same N as 'medium'): bpb {random_bpb:.4}  delta={:+.4} ({:+.1}%)",
        random_bpb - base, 100.0 * (random_bpb - base) / base);
}
