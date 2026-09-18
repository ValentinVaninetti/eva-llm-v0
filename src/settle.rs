//! What fraction of the parameters stop moving, and when?
//!
//! Before writing a freezing mechanism, find out whether there is anything to
//! freeze. Not "does the gradient ever come out small?" -- that happens all the
//! time and means nothing -- but: does a parameter stay still for SEVERAL
//! consecutive windows of real training, and does that count grow as the model
//! learns? If not, there is nothing to freeze and the line closes here.
//!
//! Training is split into windows of `every` steps. All parameters are
//! snapshotted at each window's start and end (a clone of the buffer, not the
//! graph, so it touches neither `optim.rs` nor `autograd.rs`), and movement is
//! measured relative to the parameter's own size:
//!
//! ```text
//! ratio = |end - start| / (|start| + 1e-6)
//! ```
//!
//! Still in this window = `ratio` below threshold. Settled = still for `streak`
//! CONSECUTIVE windows; a single moving window breaks the streak. Deliberately
//! not sticky: it is the same question the unfreezing side would ask.
//!
//! THE THRESHOLD IS A CHOICE, so three are reported (relaxed, medium, strict)
//! over the same run -- the snapshots exist already and recomputing is free. A
//! similar, growing fraction under all three means the number is real. If only
//! the relaxed one sees anything, it is an artifact of the threshold.


use crate::data::TextDataset;
use crate::model::{EvaConfig, EvaModel};
use crate::optim::AdamW;
use crate::rng::Rng;
use crate::tensor::autograd::backward;

/// Relaxed / medium / strict. Fixed and not parametrized: the point of this
/// experiment is to look at all three at once, not pick one.
const REL_EPS: [f32; 3] = [1e-2, 1e-3, 1e-4];

pub struct SettleConfig {
    pub data_path: String,
    pub dim: usize,
    pub ffn: usize,
    pub blocks: usize,
    pub seq_len: usize,
    pub epochs: usize,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    pub val_frac: f32,
    /// Steps per measurement window.
    pub every: usize,
    /// Consecutive still windows to count as "settled".
    pub streak: u32,
}

/// Where each tensor lives inside the flat snapshot vector.
struct Layout {
    name: String,
    offset: usize,
    len: usize,
}

/// How much a parameter moved, relative to its own size.
fn ratio(before: f32, after: f32) -> f32 {
    (after - before).abs() / (before.abs() + 1e-6)
}

fn flatten(model: &EvaModel) -> (Vec<f32>, Vec<Layout>) {
    let named = model.named_parameters();
    let mut flat = Vec::with_capacity(named.iter().map(|(_, t)| t.numel()).sum());
    let mut layout = Vec::with_capacity(named.len());
    for (name, t) in named {
        let offset = flat.len();
        flat.extend_from_slice(&t.data);
        layout.push(Layout { name, offset, len: t.numel() });
    }
    (flat, layout)
}

pub fn run(cfg: &SettleConfig) -> Result<(), String> {
    let ds = TextDataset::from_file(&cfg.data_path, cfg.seq_len)
        .map_err(|e| format!("could not read {}: {}", cfg.data_path, e))?;
    let n_windows = ds.num_windows();
    if n_windows == 0 {
        return Err(format!("the dataset is too small for seq_len {}", cfg.seq_len));
    }
    let n_val = (((n_windows as f32) * cfg.val_frac).round() as usize).clamp(1, n_windows / 2);
    let n_train = n_windows - n_val;

    let mcfg = EvaConfig {
        vocab: 256,
        dim: cfg.dim,
        ffn_dim: cfg.ffn,
        blocks: cfg.blocks,
        conv_kernel: 5,
        eps: 1e-5,
        seq_len: cfg.seq_len,
        arch: crate::model::Arch::Clock,
    };
    let mut model = EvaModel::new(mcfg);
    let mut rng = Rng::new(cfg.seed);
    let mut opt = AdamW::new(cfg.lr, cfg.wd);
    let total_params = model.param_count();
    println!("eva settle: {total_params} params | {n_train} training windows | measurement window every {} steps", cfg.every);

    // Streak per threshold, per element. Starts at 0: nobody is "old" before
    // training begins.
    let mut streaks: Vec<Vec<u32>> = REL_EPS.iter().map(|_| vec![0u32; total_params]).collect();
    let (mut start_flat, layout) = flatten(&model);

    let mut step = 0usize;
    let mut running = 0.0f32;
    let mut in_window = 0usize;
    let t0 = std::time::Instant::now();

    // History to see the trend: (step, settled fraction per threshold).
    let mut history: Vec<(usize, [f32; 3])> = Vec::new();

    for epoch in 0..cfg.epochs {
        let order = ds.shuffled_train_indices(n_train, &mut rng);
        for &wi in &order {
            let (input, target) = ds.window(wi);

            let (loss_v, grads) = {
                let logits = model.forward(&input);
                let loss = crate::tensor::ops::cross_entropy(&logits, &target);
                let v = loss.data[0];
                let g = backward(&loss);
                (v, g)
            };
            {
                let mut params = model.parameters_mut();
                opt.step(&mut params, &grads);
            }

            running += loss_v;
            step += 1;
            in_window += 1;

            if in_window == cfg.every {
                in_window = 0;
                let (end_flat, _) = flatten(&model);
                let mut settled = [0usize; 3];
                for i in 0..total_params {
                    let r = ratio(start_flat[i], end_flat[i]);
                    for (u, eps) in REL_EPS.iter().enumerate() {
                        if r < *eps {
                            streaks[u][i] += 1;
                        } else {
                            streaks[u][i] = 0;
                        }
                        if streaks[u][i] >= cfg.streak {
                            settled[u] += 1;
                        }
                    }
                }
                let frac = [
                    settled[0] as f32 / total_params as f32,
                    settled[1] as f32 / total_params as f32,
                    settled[2] as f32 / total_params as f32,
                ];
                history.push((step, frac));
                println!(
                    "step {step:>6} | epoch {} | loss {:.4} | settled (>={} still windows): relaxed {:.1}%  medium {:.1}%  strict {:.1}%",
                    epoch + 1,
                    running / cfg.every as f32,
                    cfg.streak,
                    100.0 * frac[0],
                    100.0 * frac[1],
                    100.0 * frac[2],
                );
                running = 0.0;
                start_flat = end_flat;
            }
        }
    }

    println!("\n=== per tensor, at the end ({} medium threshold, {:.0e}) ===", REL_EPS[1], REL_EPS[1]);
    let mut rows: Vec<(String, f32)> = layout
        .iter()
        .map(|l| {
            let settled = (l.offset..l.offset + l.len).filter(|&i| streaks[1][i] >= cfg.streak).count();
            (l.name.clone(), settled as f32 / l.len as f32)
        })
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, frac) in &rows {
        println!("  {name:<16} {:.1}%", 100.0 * frac);
    }

    println!("\n=== evolution of the settled fraction, medium threshold ===");
    for (s, f) in history.iter().step_by((history.len() / 12).max(1)) {
        println!("  step {s:>6}: {:.1}%", 100.0 * f[1]);
    }
    if let Some((_, last)) = history.last() {
        let grows = history.len() >= 2 && last[1] > history[history.len() / 2].1[1];
        println!("\nVERDICT: final settled fraction (relaxed/medium/strict) = {:.1}% / {:.1}% / {:.1}%",
            100.0 * last[0], 100.0 * last[1], 100.0 * last[2]);
        println!("  {}", if last[1] < 0.01 {
            "PRACTICALLY NOTHING SETTLES -> point 9 has no savings to draw from, closes here"
        } else if grows {
            "GROWS with training -> there's real signal for freezing, worth moving to the next step"
        } else {
            "there's a settled fraction but it does NOT clearly grow -> measure more epochs before building"
        });
    }
    println!("\neva settle: finished in {:.1}s", t0.elapsed().as_secs_f32());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_parameter_that_did_not_move_has_zero_ratio() {
        assert_eq!(0.0, ratio(0.5, 0.5));
    }

    #[test]
    fn a_parameter_that_doubled_has_ratio_one() {
        assert!((ratio(1.0, 2.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_row_that_never_had_gradient_does_not_blow_up() {
        // embed.table for a byte that never appears: before and after are
        // both zero. Without the +1e-6 this would be 0.0/0.0 = NaN, and
        // NaN < eps is always false -- that parameter would never count as
        // settled, the opposite result of what the real run gave. The
        // epsilon is what prevents that.
        assert_eq!(0.0, ratio(0.0, 0.0));
    }

    #[test]
    fn flatten_covers_every_parameter_exactly_once() {
        let mcfg = EvaConfig {
            vocab: 256,
            dim: 8,
            ffn_dim: 16,
            blocks: 2,
            conv_kernel: 3,
            eps: 1e-5,
            seq_len: 16,
            arch: crate::model::Arch::Clock,
        };
        let model = EvaModel::new(mcfg);
        let (flat, layout) = flatten(&model);
        assert_eq!(model.param_count(), flat.len(), "the flat vector doesn't cover every parameter");
        let covered: usize = layout.iter().map(|l| l.len).sum();
        assert_eq!(flat.len(), covered, "the layout doesn't add up to the same as the flat vector");
        // Contiguous and without gaps: each tensor starts where the
        // previous one ended.
        let mut expected = 0;
        for l in &layout {
            assert_eq!(expected, l.offset, "gap or overlap at {}", l.name);
            expected += l.len;
        }
    }

    #[test]
    fn a_streak_breaks_the_instant_it_moves() {
        // The streak isn't sticky: three still windows and one that moves
        // has to give streak 0, not 3. It's the central property of the
        // design.
        let eps = 1e-3;
        let mut streak = 0u32;
        for &r in &[0.0, 0.0, 0.0, 0.5] {
            if r < eps { streak += 1 } else { streak = 0 }
        }
        assert_eq!(0, streak, "the streak survived a large movement");
    }
}
