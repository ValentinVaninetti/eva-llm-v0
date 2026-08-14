use crate::nn::{param, Linear, Module};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// Round 4, GPT's hypothesis: is the flattening of `alpha` sigmoid
/// saturation (measured: |grad| 20-100x smaller where alpha≈0), not a
/// preference of the loss? With this on, ClockMem uses
/// `ops::algebraic_sigmoid` (polynomial tail) instead of `ops::sigmoid`
/// (exponential tail), with everything else identical -- same
/// architecture, same `alpha` range, same `EVA_ALPHA_MAX`, the only thing
/// that changes is how much gradient signal survives near the extremes.
fn antisat() -> bool {
    std::env::var("EVA_ALPHA_ANTISAT").is_ok()
}

/// Round 4, second intervention (requested by GPT, after finding that
/// `algebraic_sigmoid` wasn't touching the right region): temperature on
/// THE SAME sigmoid, `alpha=sigmoid(z/T)` with T>1. Unlike
/// `algebraic_sigmoid`, this does NOT change the shape of the curve -- it
/// stretches the same sigmoid, so a given `z` (the same one any channel
/// already had) ends up with MORE derivative, verified with a script before
/// touching the code: at z=-3..-5 (where the fast band lives today), T=1.3
/// gives ~1.4x-2.4x more derivative than T=1. I chose T=1.3 explicitly for
/// that reason, not as an arbitrary round number.
///
/// Explicit trade-off, not hidden: for "only T changes" to be literally true
/// in the code (same `log_clock` values at startup, zero change to
/// initialization), the INITIAL `alpha` shifts a bit (less extreme -- with
/// T=1.3, alpha=0.018 at init becomes ≈0.044). There's no way to have
/// EXACTLY the same initial alpha AND more derivative there at the same time
/// with a single reparametrization family (it's the same reason
/// `algebraic_sigmoid` with an adjusted inverse ended up giving LESS signal
/// to the fast band, not more). The choice here is to preserve `z` (what the
/// optimizer actually sees), not `alpha`.
fn temperature() -> f32 {
    std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0)
}

fn alpha_squash(z: &Tensor) -> Tensor {
    if antisat() {
        ops::algebraic_sigmoid(z)
    } else {
        let t = temperature();
        if t != 1.0 { ops::sigmoid(&ops::scale(z, 1.0 / t)) } else { ops::sigmoid(z) }
    }
}

/// Inverse of `alpha_squash`, only to initialize `log_clock` pointing at the
/// same target `alpha` regardless of which squashing is active -- otherwise
/// changing the squashing would also change the initial range and it
/// wouldn't be a single variable between the two conditions anymore.
fn alpha_squash_inv(a: f32) -> f32 {
    if antisat() {
        // s(z)=0.5(1+z/sqrt(1+z^2))  =>  z = (2a-1) / (2*sqrt(a(1-a)))
        (2.0 * a - 1.0) / (2.0 * (a * (1.0 - a)).sqrt())
    } else {
        (a / (1.0 - a)).ln()
    }
}

pub struct ClockMem {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wg: Linear,
    pub log_clock: Tensor,
    pub beta: Tensor,
}

impl ClockMem {
    pub fn new(d: usize, rng: &mut Rng) -> Self {
        ClockMem {
            wq: Linear::new(d, d, rng),
            wk: Linear::new(d, d, rng),
            wv: Linear::new(d, d, rng),
            wg: Linear::new(d, d, rng),
            log_clock: {
                // CLOCK CEILING. With 0.9999 a channel forgets so slowly
                // that it accumulates ~10,000 terms: harmless within a
                // 64-token window, but with persistent state across the
                // whole corpus it SATURATES -- measured, the state's
                // magnitude reached 881. With 0.999 the effective memory is
                // ~1000 tokens, fifteen times the window, which is exactly
                // what we want without blowing up.
                let a_max = std::env::var("EVA_ALPHA_MAX")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.9999);
                let a_min = 0.01f32;
                let logits: Vec<f32> = (0..d)
                    .map(|i| {
                        let t = if d == 1 { 1.0 } else { i as f32 / (d - 1) as f32 };
                        let a = a_max * (a_min / a_max).powf(t);
                        alpha_squash_inv(a)
                    })
                    .collect();
                param(logits, vec![d])
            },
            beta: param(vec![1.0], vec![1]),
        }
    }
}

impl ClockMem {
    /// Same as `forward`, starting from the state the previous window left
    /// behind and returning the one left for the next.
    pub fn forward_from(&self, x: &Tensor, s0: &[f32]) -> (Tensor, Vec<f32>) {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let g = ops::sigmoid(&self.wg.forward(x));
        let alpha = alpha_squash(&self.log_clock);
        ops::clockmem_from(&q, &k, &v, &g, &alpha, &self.beta, s0)
    }
}

impl Module for ClockMem {
    fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let g = ops::sigmoid(&self.wg.forward(x));
        let alpha = alpha_squash(&self.log_clock);
        ops::clockmem(&q, &k, &v, &g, &alpha, &self.beta)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut out = Vec::new();
        out.extend(self.wq.parameters());
        out.extend(self.wk.parameters());
        out.extend(self.wv.parameters());
        out.extend(self.wg.parameters());
        out.push(&self.log_clock);
        out.push(&self.beta);
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out = Vec::new();
        out.extend(self.wq.parameters_mut());
        out.extend(self.wk.parameters_mut());
        out.extend(self.wv.parameters_mut());
        out.extend(self.wg.parameters_mut());
        out.push(&mut self.log_clock);
        out.push(&mut self.beta);
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut out = Vec::new();
        out.extend(self.wq.named_parameters(&format!("{}.wq", prefix)));
        out.extend(self.wk.named_parameters(&format!("{}.wk", prefix)));
        out.extend(self.wv.named_parameters(&format!("{}.wv", prefix)));
        out.extend(self.wg.named_parameters(&format!("{}.wg", prefix)));
        out.push((format!("{}.log_clock", prefix), &self.log_clock));
        out.push((format!("{}.beta", prefix), &self.beta));
        out
    }

    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        let mut out = Vec::new();
        out.extend(self.wq.named_parameters_mut(&format!("{}.wq", prefix)));
        out.extend(self.wk.named_parameters_mut(&format!("{}.wk", prefix)));
        out.extend(self.wv.named_parameters_mut(&format!("{}.wv", prefix)));
        out.extend(self.wg.named_parameters_mut(&format!("{}.wg", prefix)));
        out.push((format!("{}.log_clock", prefix), &mut self.log_clock));
        out.push((format!("{}.beta", prefix), &mut self.beta));
        out
    }
}
