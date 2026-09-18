//! AdamW.
//!
//! WHY THIS FILE HAS PARALLELISM, which isn't what you'd expect: measured
//! with `EVA_PROFILE=1` on a 10.7M-parameter training run, the optimizer
//! was taking **62.7 s out of 310, 20%**. Not from doing anything unusual
//! but from the boring part: it walked every parameter TWICE, scalar and on
//! a single thread, with a square root per element. That was 6.8 ns per
//! parameter to do four arithmetic ops.
//!
//! Three changes, in order of how much they gave: splitting the elements
//! into bands over the pool (they're independent of each other, nothing to
//! synchronize), merging the two passes into one --half the memory
//! traffic-- and pulling the divisions out of the loop.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::pool::Ptr;
use crate::tensor::Tensor;

/// Below this, nothing gets split: waking up the pool costs microseconds
/// and a small tensor finishes before that pays off. The parameters that
/// matter (the dim x ffn matrices) have hundreds of thousands of elements
/// and fall on the split side.
const SPLIT_FROM: usize = 32_768;

/// How many times `make_mut` had to COPY a parameter because the buffer was
/// shared.
///
/// Has to be zero. If it goes up, someone kept the graph alive while the
/// optimizer runs and the whole model is getting copied on every step --
/// no error, just slow. It's a failure mode this project has already been
/// bitten by too many times, so it gets counted.
static COPIES: AtomicUsize = AtomicUsize::new(0);

pub fn parameter_copies() -> usize {
    COPIES.load(Relaxed)
}

/// Mutates the tensor's buffer, counting whether a copy was needed.
fn mutate(t: &mut Tensor) -> &mut Vec<f32> {
    if std::sync::Arc::strong_count(&t.data) > 1 {
        COPIES.fetch_add(1, Relaxed);
    }
    std::sync::Arc::make_mut(&mut t.data)
}

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    /// Per-parameter learning-rate multiplier, keyed by tensor id. B2.1
    /// gives the windowed readout a separate, scaled LR. An empty map is a
    /// plain AdamW. (Note: this is a real LR multiplier applied inside the
    /// step -- scaling the GRADIENT instead would cancel out, because Adam
    /// normalizes by the second moment.)
    pub lr_mult: HashMap<usize, f32>,
    t: usize,
    state: HashMap<usize, (Vec<f32>, Vec<f32>)>,
}

/// What doesn't change within a step. Computed once and carried into the
/// loop.
#[derive(Clone, Copy)]
struct Consts {
    b1: f32,
    b2: f32,
    /// `1 - b1` and `1 - b2`, already subtracted.
    om1: f32,
    om2: f32,
    /// Reciprocals of the bias corrections: dividing per element when the
    /// divisor is the same for all ten million is throwing away cycles.
    inv_m: f32,
    inv_v: f32,
    lr: f32,
    wd: f32,
    eps: f32,
}

impl AdamW {
    pub fn new(lr: f32, wd: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd,
            lr_mult: HashMap::new(),
            t: 0,
            state: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &HashMap<usize, Vec<f32>>) {
        self.t += 1;
        let t = self.t as i32;
        let c = Consts {
            b1: self.beta1,
            b2: self.beta2,
            om1: 1.0 - self.beta1,
            om2: 1.0 - self.beta2,
            inv_m: 1.0 / (1.0 - self.beta1.powi(t)),
            inv_v: 1.0 / (1.0 - self.beta2.powi(t)),
            lr: self.lr,
            wd: self.wd,
            eps: self.eps,
        };

        for p in params.iter_mut() {
            let Some(g) = grads.get(&p.id) else { continue };
            debug_assert_eq!(g.len(), p.data.len(), "grad shape mismatch for param {}", p.id);
            // Per-parameter LR multiplier (Consts is Copy: cheap to adjust).
            let mut c = c;
            if let Some(mult) = self.lr_mult.get(&p.id) {
                c.lr = self.lr * *mult;
            }
            let (m, v) = self
                .state
                .entry(p.id)
                .or_insert_with(|| (vec![0.0; p.data.len()], vec![0.0; p.data.len()]));

            let n = p.data.len();
            let pool = crate::pool::global();
            let chunks = if n < SPLIT_FROM { 1 } else { pool.workers() + 1 };
            if chunks <= 1 {
                // `make_mut` copies ONLY if the buffer is shared. In the
                // training loop the graph has already been freed by the
                // time we get here, so the count is one and nothing gets
                // copied. If it ever does copy, it means someone kept the
                // graph alive -- and it shows up as a speed drop, not as an
                // error.
                update(mutate(p), g, m, v, c);
                continue;
            }

            let per = n.div_ceil(chunks);
            let (pp, gp) = (Ptr(mutate(p).as_mut_ptr()), Ptr(g.as_ptr()));
            let (mp, vp) = (Ptr(m.as_mut_ptr()), Ptr(v.as_mut_ptr()));
            pool.run(chunks, move |i| {
                let lo = i * per;
                if lo >= n {
                    return;
                }
                let len = per.min(n - lo);
                // SAFETY: the band [lo, lo+len) belongs to this piece and
                // no other; bands never overlap. All four buffers have the
                // same length `n`, and `run` doesn't return until every one
                // finished.
                update(
                    pp.offset_by(lo).as_mut(len),
                    gp.at(lo).as_ref(len),
                    mp.offset_by(lo).as_mut(len),
                    vp.offset_by(lo).as_mut(len),
                    c,
                );
            });
        }
    }
}

/// The AdamW step over one band. A single pass through the four buffers.
///
/// The four zipped `iter`s aren't a style choice: they remove the
/// compiler's bounds checks and let it vectorize the whole loop, square
/// root included.
fn update(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], c: Consts) {
    for (((p, g), m), v) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
        *m = c.b1 * *m + c.om1 * g;
        *v = c.b2 * *v + c.om2 * g * g;
        let mhat = *m * c.inv_m;
        let vhat = *v * c.inv_v;
        *p -= c.lr * (mhat / (vhat.sqrt() + c.eps) + c.wd * *p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splitting into bands cannot change the result: each element depends
    /// only on its own m, v, and gradient. If this fails, some band is
    /// stepping on another.
    #[test]
    fn splitting_gives_the_same_numbers_as_one_thread() {
        let n = SPLIT_FROM * 3 + 17; // deliberately not a multiple of anything
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();

        let mut serial = vec![0.5f32; n];
        let (mut m, mut v) = (vec![0.0; n], vec![0.0; n]);
        let c = Consts {
            b1: 0.9,
            b2: 0.999,
            om1: 0.1,
            om2: 0.001,
            inv_m: 1.0 / 0.1,
            inv_v: 1.0 / 0.001,
            lr: 1e-3,
            wd: 0.01,
            eps: 1e-8,
        };
        update(&mut serial, &grad, &mut m, &mut v, c);

        let mut par = vec![0.5f32; n];
        let (mut m2, mut v2) = (vec![0.0; n], vec![0.0; n]);
        let pool = crate::pool::global();
        let chunks = pool.workers() + 1;
        let per = n.div_ceil(chunks);
        let (pp, gp) = (Ptr(par.as_mut_ptr()), Ptr(grad.as_ptr()));
        let (mp, vp) = (Ptr(m2.as_mut_ptr()), Ptr(v2.as_mut_ptr()));
        pool.run(chunks, move |i| {
            let lo = i * per;
            if lo >= n {
                return;
            }
            let len = per.min(n - lo);
            update(
                pp.offset_by(lo).as_mut(len),
                gp.at(lo).as_ref(len),
                mp.offset_by(lo).as_mut(len),
                vp.offset_by(lo).as_mut(len),
                c,
            );
        });

        assert_eq!(serial, par, "splitting into bands changed the result");
    }
}
