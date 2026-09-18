//! RETROSPECTIVE CEILING: how much can be saved, at most, by skipping blocks
//! that do not change the answer?
//!
//! The full model runs over UNSEEN text, then again with one block left out at
//! a time (the residual stream passes through as if it were not there), and
//! each variant is measured on bits/byte, how many positions change argmax,
//! wall time (the physical cost, per house rule), and the block weights no
//! longer read.
//!
//! The ceiling is what an ideal rule that already knew each block's effect
//! would have saved -- not what a runtime gate would achieve, but the most
//! there is to gain. If dropping blocks barely hurts quality there is room for
//! a mechanism that skips them; if dropping any destroys quality, the line dies
//! here and the scheduler never gets written.
//!
//! Nothing is tuned against this number: it is measured on validation, using
//! the same contiguous cut as training.

use std::time::Instant;

use crate::data::TextDataset;
use crate::model::EvaModel;
use crate::nn::Module;

/// `p[target]` and the argmax of a row of logits, with a stable softmax.
fn softmax_p(logits: &[f32], tgt: usize) -> (f32, usize) {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut p_tgt = 0.0f32;
    let mut argmax = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - mx).exp();
        sum += e;
        if i == tgt {
            p_tgt = e;
        }
        if e > best {
            best = e;
            argmax = i;
        }
    }
    (p_tgt / sum, argmax)
}

/// One variant with block `idx` left out.
pub struct BlockRes {
    pub idx: usize,
    /// bits/byte of the variant over validation.
    pub bits_per_byte: f32,
    /// Delta bits/byte against the full model.
    pub delta_bits: f32,
    /// Relative delta (0.01 = 1%): the metric that decides.
    pub rel_delta: f32,
    /// Fraction of positions where argmax changed versus the full model.
    pub pred_change: f32,
    /// Total wall time of the variant, measured.
    pub time_total_s: f64,
    /// What the block costs in time = full - variant.
    pub block_time_s: f64,
    /// Parameter bytes of the block (what no longer gets read).
    pub weight_bytes: usize,
}

pub struct CeilingReport {
    pub n_pos: usize,
    pub n_val: usize,
    pub full_bits_per_byte: f32,
    pub full_time_s: f64,
    pub total_weight_bytes: usize,
    pub blocks: Vec<BlockRes>,
}

/// Result of leaving out an explicit set in ONE real pass.
pub struct ComboRes {
    pub skips: Vec<usize>,
    pub full_bits_per_byte: f32,
    pub bits_per_byte: f32,
    pub delta_bits: f32,
    pub rel_delta: f32,
    pub pred_change: f32,
    pub full_time_s: f64,
    pub time_total_s: f64,
    pub weight_bytes: usize,
}

fn pass(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, skips: &[usize]) -> (f64, Vec<usize>, f64) {
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let mut nats = 0.0f64;
    let mut argmax = Vec::with_capacity((to - from) * seq);
    let t0 = Instant::now();
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_skips(&input, skips);
        for t in 0..seq {
            let (p, am) = softmax_p(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
            nats -= (p as f64).max(1e-12).ln();
            argmax.push(am);
        }
    }
    (nats, argmax, t0.elapsed().as_secs_f64())
}

/// Measures one real combination, not the sum of individual ablations.
pub fn measure_combo(
    model: &EvaModel, ds: &TextDataset, from: usize, to: usize, skips: &[usize],
) -> Result<ComboRes, String> {
    if skips.is_empty() || skips.iter().any(|&i| i >= model.blocks.len()) {
        return Err("the combination has to leave out at least one existing block".into());
    }
    if skips.windows(2).any(|w| w[0] >= w[1]) {
        return Err("the skipped blocks have to be sorted and without repeats".into());
    }
    let n_pos = (to - from) * model.cfg.seq_len;
    let (nats_full, full_argmax, full_time_s) = pass(model, ds, from, to, &[]);
    let (nats, argmax, time_total_s) = pass(model, ds, from, to, skips);
    let full_bits = (nats_full / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;
    let bits = (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;
    let changes = argmax.iter().zip(&full_argmax).filter(|(a, b)| a != b).count();
    let weight_bytes = skips.iter().map(|&i| {
        model.blocks[i].parameters().iter().map(|t| t.numel()).sum::<usize>() * 4
    }).sum();
    Ok(ComboRes {
        skips: skips.to_vec(), full_bits_per_byte: full_bits, bits_per_byte: bits,
        delta_bits: bits - full_bits, rel_delta: (bits - full_bits) / full_bits.max(1e-9),
        pred_change: changes as f32 / n_pos.max(1) as f32,
        full_time_s, time_total_s, weight_bytes,
    })
}

/// Measures the ceiling over windows `ds[from..to]`.
///
/// One pass per variant: first the full model (the baseline), then one for
/// each block left out. Each variant's time is measured on the wall clock;
/// what the block costs is the difference against the full model, over the
/// SAME text.
pub fn measure(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) -> Result<CeilingReport, String> {
    if model.blocks.is_empty() {
        return Err("the model has no blocks: there's no ceiling to measure".into());
    }
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let n_val = to - from;
    let n_pos = n_val * seq;

    // Full pass: the baseline.
    let mut nats_full = 0.0f64;
    let mut full_argmax = Vec::with_capacity(n_pos);
    let t0 = Instant::now();
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_skip(&input, None);
        for t in 0..seq {
            let (p, am) = softmax_p(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
            nats_full -= (p as f64).max(1e-12).ln();
            full_argmax.push(am);
        }
    }
    let full_time_s = t0.elapsed().as_secs_f64();
    let full_bits = (nats_full / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;

    // One pass per skipped block.
    let mut blocks = Vec::with_capacity(model.blocks.len());
    for idx in 0..model.blocks.len() {
        let mut nats = 0.0f64;
        let mut changes = 0usize;
        let t0 = Instant::now();
        for wi in from..to {
            let rel = wi - from;
            let (input, target) = ds.window(wi);
            let (logits, _) = model.forward_skip(&input, Some(idx));
            for t in 0..seq {
                let (p, am) = softmax_p(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
                nats -= (p as f64).max(1e-12).ln();
                changes += (am != full_argmax[rel * seq + t]) as usize;
            }
        }
        let time_s = t0.elapsed().as_secs_f64();
        let bpb = (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;
        let weight_bytes: usize =
            model.blocks[idx].parameters().iter().map(|t: &&crate::tensor::Tensor| t.numel()).sum::<usize>() * 4;
        blocks.push(BlockRes {
            idx,
            bits_per_byte: bpb,
            delta_bits: bpb - full_bits,
            rel_delta: (bpb - full_bits) / full_bits.max(1e-9),
            pred_change: changes as f32 / n_pos.max(1) as f32,
            time_total_s: time_s,
            block_time_s: full_time_s - time_s,
            weight_bytes,
        });
    }

    Ok(CeilingReport {
        n_pos,
        n_val,
        full_bits_per_byte: full_bits,
        full_time_s,
        total_weight_bytes: model.param_count() * 4,
        blocks,
    })
}

/// How much an ideal rule that knew the real effect and skipped every block
/// with relative bits/byte loss `<= threshold` would have saved.
///
/// Returns (blocks skipped, % of time saved, % of weights not read,
/// projected bits/byte if every chosen block were skipped). The projected
/// loss is the SUM of the individual losses: running the combination
/// together could give more or less, and that isn't assumed -- this is the
/// ceiling, not the implementation.
pub fn ideal_rule(report: &CeilingReport, threshold: f32) -> (usize, f32, f32, f32) {
    let mut skip = 0usize;
    let mut t = 0.0f64;
    let mut w = 0usize;
    let mut loss = 0.0f32;
    for b in &report.blocks {
        if b.rel_delta <= threshold {
            skip += 1;
            t += b.block_time_s;
            w += b.weight_bytes;
            loss += b.delta_bits;
        }
    }
    (
        skip,
        (100.0 * t / report.full_time_s.max(1e-12)) as f32,
        (100.0 * w as f64 / report.total_weight_bytes.max(1) as f64) as f32,
        report.full_bits_per_byte + loss,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Arch;
    use crate::model::EvaConfig;

    fn mini_model() -> EvaModel {
        EvaModel::new(EvaConfig {
            vocab: 256,
            dim: 8,
            ffn_dim: 16,
            blocks: 2,
            conv_kernel: 3,
            eps: 1e-5,
            seq_len: 6,
            arch: Arch::Clock,
        })
    }

    fn mini_ds() -> TextDataset {
        TextDataset::from_str("hola mundo, esta es una prueba de texto para el techo. ", 6)
    }

    #[test]
    fn skip_none_is_exactly_forward_hidden() {
        let m = mini_model();
        let ids = vec![1usize, 2, 3, 4, 5, 6];
        let (a, ha) = m.forward_hidden(&ids);
        let (b, hb) = m.forward_skip(&ids, None);
        assert_eq!(a.data, b.data, "logits have to be identical");
        assert_eq!(ha.data, hb.data, "hidden has to be identical");
    }

    #[test]
    fn skipping_one_block_only_changes_that_block() {
        let m = mini_model();
        let ids = vec![1usize, 2, 3, 4, 5, 6];
        // Skip 0 and skip 1 have to differ from the full model (if the
        // block does nothing, there's no ceiling to measure), and they can
        // differ from each other.
        let (full, _) = m.forward_skip(&ids, None);
        let (s0, _) = m.forward_skip(&ids, Some(0));
        let (s1, _) = m.forward_skip(&ids, Some(1));
        assert_ne!(full.data, s0.data, "dropping block 0 cannot change nothing");
        assert_ne!(full.data, s1.data, "dropping block 1 cannot change nothing");
    }

    #[test]
    fn softmax_matches_the_hand_computation() {
        let logits = [0.0f32, 2.0, 1.0];
        let s = 1.0 + 2.0f32.exp() + 1.0f32.exp();
        let (p, am) = softmax_p(&logits, 0);
        assert!((p - 1.0 / s).abs() < 1e-5, "p[truth] {p}");
        assert_eq!(am, 1, "the argmax is index 1");
    }

    #[test]
    fn measure_returns_one_result_per_block() {
        let m = mini_model();
        let ds = mini_ds();
        let n = ds.num_windows();
        assert!(n >= 2, "the test dataset has to be splittable");
        let rep = measure(&m, &ds, 0, n).unwrap();
        assert_eq!(rep.blocks.len(), m.blocks.len());
        assert_eq!(rep.n_pos, n * m.cfg.seq_len);
        assert!(rep.full_bits_per_byte > 0.0, "bits/byte of the full model");
        for b in &rep.blocks {
            assert!(b.weight_bytes > 0);
            assert!(b.pred_change >= 0.0 && b.pred_change <= 1.0, "pred_change {}", b.pred_change);
        }
    }

    #[test]
    fn combo_rejects_invalid_indices_and_measures_a_set() {
        let m = mini_model();
        let ds = mini_ds();
        let n = ds.num_windows();
        assert!(measure_combo(&m, &ds, 0, n, &[0, 0]).is_err(), "does not accept repeats");
        assert!(measure_combo(&m, &ds, 0, n, &[2]).is_err(), "does not accept out-of-range");
        let r = measure_combo(&m, &ds, 0, n, &[0, 1]).expect("measures a valid combination");
        assert_eq!(r.skips, vec![0, 1]);
        assert!(r.bits_per_byte.is_finite());
        assert!(r.time_total_s >= 0.0);
        assert!(r.weight_bytes > 0);
    }

    #[test]
    fn ideal_rule_sorts_correctly() {
        let m = mini_model();
        let ds = mini_ds();
        let n = ds.num_windows();
        let rep = measure(&m, &ds, 0, n).unwrap();
        // With threshold 0 nobody qualifies; with a huge threshold everyone
        // does.
        let (zero, _, _, _) = ideal_rule(&rep, 0.0);
        let (all, _, _, _) = ideal_rule(&rep, f32::MAX);
        assert_eq!(zero, 0);
        assert_eq!(all, rep.blocks.len());
    }
}
