//! Token-by-token inference, carrying state.
//!
//! WHY, with the waste measured: `generate` used to redo the full pass over the
//! whole window for EVERY token, computing 64 rows to use one and throw away
//! 63 -- up to 64x wasted work per token at seq=64 -- and it built the entire
//! autograd graph, with its per-op input clones, only to discard it.
//!
//! WHAT MAKES IT POSSIBLE is the property that sets this architecture apart:
//! ClockMem's state is FIXED SIZE, D numbers, however long the context. A
//! transformer cannot do this; it needs a key/value cache that grows with every
//! token. Both are implemented here precisely so that difference can be
//! measured instead of claimed.
//!
//! THE DANGER, and why the test was written before the code: this is a SECOND
//! implementation of the same maths. The two paths can drift apart silently --
//! training stays correct while the fast path computes something else, with
//! nothing failing. `tests::stream_matches_batch` demands step-by-step equality
//! with the full pass. If that test goes red, this file is the one lying.

use crate::model::block::Mixer;
use crate::model::EvaModel;
use crate::nn::{Linear, RMSNorm};

/// State of a block between one token and the next.
struct BlockState {
    /// The conv's previous `k-1` inputs, in order, flattened.
    ///
    /// Starts at zeros and that **is** the padding at the start: the conv
    /// only adds `w * x[src]` when `src >= 0`, and multiplying by zero
    /// gives the same result as not adding it. No special case needed.
    conv: Vec<f32>,
    mixer: MixerState,
}

enum MixerState {
    /// D numbers. Never grows, no matter how long the context gets.
    Clock(Vec<f32>),
    /// Grows with every token: this is the difference, made into code.
    Attn { keys: Vec<f32>, vals: Vec<f32> },
}

pub struct Streamer<'a> {
    model: &'a EvaModel,
    blocks: Vec<BlockState>,
    /// Absolute position. Only used by attention's position table.
    t: usize,
    /// `sigmoid(log_clock)` per block, computed once.
    alphas: Vec<Vec<f32>>,
}

impl<'a> Streamer<'a> {
    pub fn new(model: &'a EvaModel) -> Self {
        let d = model.cfg.dim;
        let blocks = model
            .blocks
            .iter()
            .map(|b| BlockState {
                conv: vec![0.0; (b.conv.k - 1) * d],
                mixer: match &b.mixer {
                    Mixer::Clock(_) => MixerState::Clock(vec![0.0; d]),
                    Mixer::Attn(_) => MixerState::Attn { keys: Vec::new(), vals: Vec::new() },
                },
            })
            .collect();
        let alphas = model
            .blocks
            .iter()
            .map(|b| match &b.mixer {
                Mixer::Clock(m) => m.log_clock.data.iter().map(|&v| sigmoid(v)).collect(),
                Mixer::Attn(_) => Vec::new(),
            })
            .collect();
        Streamer { model, blocks, t: 0, alphas }
    }

    /// Consumes a token and returns the next one's logits.
    pub fn next(&mut self, id: usize) -> Vec<f32> {
        let x = self.advance(id);
        let x = rms_norm(&x, &self.model.norm_out);
        matvec(&x, &self.model.head_w.data, self.model.cfg.dim, self.model.cfg.vocab)
    }

    /// Consumes a token WITHOUT computing the output projection.
    ///
    /// For when the structure already decided what comes next: there's
    /// nothing to ask the model, but the state still has to advance or it
    /// drifts out of sync with the text. This is what makes restricting
    /// SAVE work instead of just avoiding the error.
    pub fn consume(&mut self, id: usize) {
        self.advance(id);
    }

    /// Logits for only some columns, when the structure left few options.
    /// Returns `(token, logit)` in the same order as `cols`.
    pub fn next_among(&mut self, id: usize, cols: &[usize]) -> Vec<f32> {
        let x = self.advance(id);
        let x = rms_norm(&x, &self.model.norm_out);
        let d = self.model.cfg.dim;
        let w = &self.model.head_w.data;
        let v = self.model.cfg.vocab;
        cols.iter()
            .map(|&c| (0..d).map(|i| x[i] * w[i * v + c]).sum())
            .collect()
    }

    /// The whole model minus the head: leaves the state ready and returns
    /// the position's representation.
    fn advance(&mut self, id: usize) -> Vec<f32> {
        let cfg = &self.model.cfg;
        let d = cfg.dim;

        let mut x = self.model.embed.table.data[id * d..(id + 1) * d].to_vec();
        if let Some(pos) = &self.model.pos {
            // The table has fixed length; past it the last row repeats,
            // which is what the window trimming used to do.
            let p = self.t.min(cfg.seq_len - 1);
            for (xi, pi) in x.iter_mut().zip(&pos.data[p * d..(p + 1) * d]) {
                *xi += pi;
            }
        }

        for i in 0..self.model.blocks.len() {
            x = self.block_step(i, &x);
        }
        self.t += 1;
        x
    }

    fn block_step(&mut self, i: usize, x: &[f32]) -> Vec<f32> {
        let b = &self.model.blocks[i];
        let d = self.model.cfg.dim;

        // h = x + conv(norm0(x))
        let n0 = rms_norm(x, &b.norm0);
        let c = self.conv_step(i, &n0);
        let mut h: Vec<f32> = x.iter().zip(&c).map(|(a, b)| a + b).collect();

        // h = h + mixer(norm1(h))
        let n1 = rms_norm(&h, &b.norm1);
        let m = self.mixer_step(i, &n1);
        for (hi, mi) in h.iter_mut().zip(&m) {
            *hi += mi;
        }

        // out = h + glu(norm2(h))
        let n2 = rms_norm(&h, &b.norm2);
        let a = matvec(&n2, &b.glu.w1.data, d, b.glu.w1.shape[1]);
        let g = matvec(&n2, &b.glu.w2.data, d, b.glu.w2.shape[1]);
        let gated: Vec<f32> = a.iter().zip(&g).map(|(x, y)| silu(*x) * y).collect();
        let f = matvec(&gated, &b.glu.w3.data, gated.len(), d);
        for (hi, fi) in h.iter_mut().zip(&f) {
            *hi += fi;
        }
        h
    }

    fn conv_step(&mut self, i: usize, x: &[f32]) -> Vec<f32> {
        let b = &self.model.blocks[i];
        let (d, k) = (self.model.cfg.dim, b.conv.k);
        let hist = &self.blocks[i].conv;
        let mut out = vec![0.0; d];
        for c in 0..d {
            let mut acc = b.conv.b.data[c];
            // `u` walks the kernel; the first k-1 weights go against
            // history and the last one against the current token, same as
            // the `src` in the batched version.
            for u in 0..k - 1 {
                acc += b.conv.w.data[c * k + u] * hist[u * d + c];
            }
            acc += b.conv.w.data[c * k + (k - 1)] * x[c];
            out[c] = acc;
        }
        // Slides the window: the oldest leaves, the current one enters.
        let hist = &mut self.blocks[i].conv;
        if k > 1 {
            hist.copy_within(d.., 0);
            hist[(k - 2) * d..].copy_from_slice(x);
        }
        out
    }

    fn mixer_step(&mut self, i: usize, x: &[f32]) -> Vec<f32> {
        let d = self.model.cfg.dim;
        match (&self.model.blocks[i].mixer, &mut self.blocks[i].mixer) {
            (Mixer::Clock(m), MixerState::Clock(s)) => {
                let q = linear(x, &m.wq, d);
                let kk = linear(x, &m.wk, d);
                let v = linear(x, &m.wv, d);
                let g = linear(x, &m.wg, d);
                let beta = m.beta.data[0];
                let alpha = &self.alphas[i];
                let mut out = vec![0.0; d];
                for c in 0..d {
                    s[c] = alpha[c] * s[c] + beta * kk[c] * v[c];
                    out[c] = q[c] * s[c] * sigmoid(g[c]);
                }
                out
            }
            (Mixer::Attn(m), MixerState::Attn { keys, vals }) => {
                let q = linear(x, &m.wq, d);
                keys.extend_from_slice(&linear(x, &m.wk, d));
                vals.extend_from_slice(&linear(x, &m.wv, d));
                // The cache gets trimmed to the window it was trained
                // with, which is what the old version's window trimming
                // did. Without a cap, attention looks further than it ever
                // saw during training, AND memory grows without bound.
                // ClockMem doesn't need this trim: its forgetting is built
                // into the mechanism.
                let cap = self.model.cfg.seq_len;
                if keys.len() / d > cap {
                    keys.drain(..d);
                    vals.drain(..d);
                }
                let n = keys.len() / d;
                let scale = 1.0 / (d as f32).sqrt();

                let mut scores: Vec<f32> = (0..n)
                    .map(|j| {
                        let kj = &keys[j * d..(j + 1) * d];
                        q.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;

                let mut ctx = vec![0.0; d];
                for j in 0..n {
                    let a = scores[j] * inv;
                    for (c, o) in ctx.iter_mut().enumerate() {
                        *o += a * vals[j * d + c];
                    }
                }
                linear(&ctx, &m.wo, d)
            }
            _ => unreachable!("the state does not match the mixer"),
        }
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn rms_norm(x: &[f32], n: &RMSNorm) -> Vec<f32> {
    let d = x.len();
    let sq: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sq / d as f32 + n.eps).sqrt();
    x.iter().zip(n.w.data.iter()).map(|(v, w)| v * inv * w).collect()
}

fn linear(x: &[f32], l: &Linear, out_d: usize) -> Vec<f32> {
    let mut y = matvec(x, &l.w.data, x.len(), out_d);
    for (yi, bi) in y.iter_mut().zip(l.b.data.iter()) {
        *yi += bi;
    }
    y
}

/// One row times a matrix. Uses the same `math::matmul` that training does
/// --with its AVX2 and its work splitting-- instead of writing a third
/// product implementation.
fn matvec(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    crate::math::matmul(x, w, 1, k, n, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, EvaConfig, EvaModel};

    fn make_model(arch: Arch) -> EvaModel {
        EvaModel::new(EvaConfig {
            vocab: 32,
            dim: 24,
            ffn_dim: 40,
            blocks: 3,
            conv_kernel: 4,
            eps: 1e-5,
            seq_len: 12,
            arch,
        })
    }

    /// THE TEST THAT JUSTIFIES THIS FILE.
    ///
    /// `stream.rs` is a second implementation of the same math, and two
    /// implementations can drift out of sync silently: the training one
    /// stays correct, the fast one computes something else, and nothing
    /// fails. This requires that the last row of the full pass matches
    /// what the step-by-step version returns, for every position.
    ///
    /// Exact bit equality isn't required: the batched pass sums in a
    /// different order than the single row does, and in floating point
    /// that differs in the last digit. A relative 1e-4 is required, which
    /// is several orders of magnitude below any difference that would mean
    /// a logic error.
    fn stream_matches_batch(arch: Arch) {
        let m = make_model(arch);
        let ids: Vec<usize> = vec![7, 3, 19, 0, 11, 4, 28, 15];

        let mut st = Streamer::new(&m);
        for (i, &id) in ids.iter().enumerate() {
            let step = st.next(id);

            // Full pass over the prefix: its last row predicts the same
            // thing as the step-by-step version after consuming that
            // token.
            let batch = m.forward(&ids[..=i]);
            let v = batch.shape[1];
            let last = &batch.data[(batch.shape[0] - 1) * v..];

            assert_eq!(last.len(), step.len());
            for (j, (a, b)) in last.iter().zip(&step).enumerate() {
                let err = (a - b).abs() / (a.abs() + b.abs()).max(1.0);
                assert!(
                    err < 1e-4,
                    "position {i}, logit {j}: batch={a} stream={b} (err {err:.2e})"
                );
            }
        }
    }

    #[test]
    fn stream_matches_batch_clock() {
        stream_matches_batch(Arch::Clock);
    }

    #[test]
    fn stream_matches_batch_attn() {
        stream_matches_batch(Arch::Attn);
    }

    /// The property that makes all of this worth it: ClockMem's state does
    /// NOT grow with context. If it ever does, we lost the one structural
    /// advantage we have over a transformer.
    #[test]
    fn clock_state_does_not_grow_with_context() {
        let m = make_model(Arch::Clock);
        let mut st = Streamer::new(&m);
        let size = |s: &Streamer| -> usize {
            s.blocks
                .iter()
                .map(|b| {
                    b.conv.len()
                        + match &b.mixer {
                            MixerState::Clock(v) => v.len(),
                            MixerState::Attn { keys, vals } => keys.len() + vals.len(),
                        }
                })
                .sum()
        };
        let initial = size(&st);
        for i in 0..200 {
            st.next(i % 32);
        }
        assert_eq!(initial, size(&st), "the state grew with context");
    }

    /// And the contrast, which is the point of having implemented both:
    /// attention's DOES grow, linearly. This isn't an implementation
    /// defect, it's the architecture.
    #[test]
    fn attention_cache_grows_and_that_is_the_difference() {
        let m = make_model(Arch::Attn);
        let mut st = Streamer::new(&m);
        let cache = |s: &Streamer| -> usize {
            s.blocks
                .iter()
                .map(|b| match &b.mixer {
                    MixerState::Attn { keys, vals } => keys.len() + vals.len(),
                    MixerState::Clock(v) => v.len(),
                })
                .sum()
        };
        assert_eq!(0, cache(&st));
        for i in 0..50 {
            st.next(i % 32);
        }
        // Grows up to the training window and stops there. Even capped,
        // it's 12 times ClockMem's state for the same model -- and
        // uncapped it would grow forever.
        assert_eq!(3 * 12 * 24 * 2, cache(&st));
        let clock = make_model(Arch::Clock);
        let mut sc = Streamer::new(&clock);
        for i in 0..50 {
            sc.next(i % 32);
        }
        let clock_state: usize = sc
            .blocks
            .iter()
            .map(|b| match &b.mixer {
                MixerState::Clock(v) => v.len(),
                MixerState::Attn { keys, vals } => keys.len() + vals.len(),
            })
            .sum();
        assert!(cache(&st) > 10 * clock_state, "the memory difference got lost");
    }
}
