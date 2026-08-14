//! Local credit: no gradient crosses from one block to another.
//!
//! THE BET. To backpropagate you have to hold the activations of the whole
//! pass: training memory scales with **depth x width x length**, and that's
//! what forces buying expensive hardware. It's not the compute, it's the
//! memory.
//!
//! If each block has its own objective and gradient doesn't cross, training
//! an N-block network needs the memory of **one**. No kernel optimization
//! does that.
//!
//! HOW IT'S CHARGED, which is the part that's easy to get wrong: it's not
//! enough to disconnect the inputs. If every block's losses get summed and
//! ONE backward runs at the end, the graph is still whole and nothing was
//! saved. The benefit only exists if each block **backpropagates, updates,
//! and frees** before the next one starts. That's why the loop below is
//! shaped the way it is.
//!
//! THE REASON TO DOUBT IT, and it's a strong one: **local rules do worse
//! than backprop in every serious published attempt.** This isn't proposed
//! because it's expected to win. It's proposed because it's always been
//! tested on architectures designed *for* backprop, never on a recurrent
//! model with per-channel decay where temporal credit is already analytic
//! and local; and because if it ties, the savings are structural. It gets
//! measured against the baseline and the verdict is accepted.

use crate::nn::{param, Module, RMSNorm};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// One prediction head per intermediate block.
///
/// The last block doesn't get one: it uses the model's real output, so
/// inference stays identical and the comparison with the baseline is over
/// the same thing.
///
/// These are training scaffolding and **aren't saved** in the checkpoint.
/// They cost `dim x vocab` each; with dim 256 and vocab 256 that's 65K per
/// intermediate block, and they have to be counted when comparing costs
/// even though they don't end up in the final model.
pub struct Heads {
    norms: Vec<RMSNorm>,
    projs: Vec<Tensor>,
}

impl Heads {
    pub fn new(intermediate: usize, dim: usize, vocab: usize, eps: f32, rng: &mut Rng) -> Self {
        let bound = 1.0 / (dim as f32).sqrt();
        Heads {
            norms: (0..intermediate).map(|_| RMSNorm::new(dim, eps)).collect(),
            projs: (0..intermediate)
                .map(|_| {
                    param(
                        (0..dim * vocab).map(|_| rng.uniform(-bound, bound)).collect(),
                        vec![dim, vocab],
                    )
                })
                .collect(),
        }
    }

    pub fn logits(&self, i: usize, y: &Tensor) -> Tensor {
        ops::matmul(&self.norms[i].forward(y), &self.projs[i])
    }

    pub fn params_mut(&mut self, i: usize) -> Vec<&mut Tensor> {
        let mut out = self.norms[i].parameters_mut();
        out.push(&mut self.projs[i]);
        out
    }

    pub fn count(&self) -> usize {
        self.projs.iter().map(|p| p.data.len()).sum::<usize>()
            + self.norms.iter().map(|n| n.w.data.len()).sum::<usize>()
    }
}

/// Peak resident memory of the process, in MB.
///
/// This is the number that decides what machine is needed. Read from
/// `VmHWM`, which is the kernel's watermark: it doesn't go down even after
/// the memory gets freed, which is exactly what we want to know.
pub fn peak_rss_mb() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return f64::NAN;
    };
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = v.split_whitespace().next().and_then(|k| k.parse::<f64>().ok()) {
                return kb / 1024.0;
            }
        }
    }
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_says_something_plausible() {
        let mb = peak_rss_mb();
        assert!(mb > 0.1 && mb < 1_000_000.0, "absurd memory peak: {mb}");
    }

    #[test]
    fn heads_produce_logits_of_the_right_shape() {
        let mut rng = Rng::new(1);
        let (dim, vocab, s) = (8, 5, 3);
        let h = Heads::new(2, dim, vocab, 1e-5, &mut rng);
        let y = param((0..s * dim).map(|i| i as f32 * 0.1).collect(), vec![s, dim]);
        let l = h.logits(0, &y);
        assert_eq!(vec![s, vocab], l.shape);
    }

    /// A head's gradient must NOT be able to reach whatever fed it if the
    /// input came in detached. If this fails, the credit isn't local and
    /// the memory measurement is measuring something else.
    #[test]
    fn a_detached_input_receives_no_gradient() {
        let mut rng = Rng::new(2);
        let (dim, vocab) = (6, 4);
        let h = Heads::new(1, dim, vocab, 1e-5, &mut rng);
        let upstream = param((0..2 * dim).map(|i| (i as f32) * 0.05).collect(), vec![2, dim]);

        let cut = upstream.detach();
        let l = h.logits(0, &cut);
        let grads = crate::tensor::autograd::backward(&ops::sum_all(&l));

        assert!(
            grads.get(&upstream.id).is_none(),
            "the gradient crossed the cut: the credit isn't local"
        );
    }
}
