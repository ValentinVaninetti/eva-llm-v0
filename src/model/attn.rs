//! Single-head causal attention. **Exists to have something to measure
//! against.**
//!
//! It isn't the direction of the project: it's the reference baseline. Until
//! this existed, the only evidence that EvaClock worked was that it
//! memorized a 55 KB corpus, which anything with enough parameters can do.
//!
//! THE COMPARISON IS FAIR BY CONSTRUCTION, which is the whole point: ClockMem
//! has `wq, wk, wv, wg` and this has `wq, wk, wv, wo` -- four DxD matrices in
//! both cases, **exactly the same number of parameters**. It plugs into the
//! same block, with the same conv, the same FFN, the same norms and the same
//! optimizer. The only thing that changes is the temporal mixer.
//!
//! A single head on purpose, not out of laziness: ClockMem doesn't have heads
//! either. Splitting this into heads while leaving the other alone wouldn't
//! compare the two mechanisms, it would compare one of them against itself
//! with more machinery around it.
//!
//! The difference we actually want to see is structural: here the state is
//! an SxS matrix that gets built in full (O(S^2) in compute and memory) and
//! can do associative retrieval --key A looks up value B--; ClockMem carries
//! a D-sized vector (O(S*D), O(D) at inference) and can't. The question is
//! how much that difference is worth in actual loss, not in principle.

use crate::nn::{Linear, Module};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

pub struct Attention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    scale: f32,
}

impl Attention {
    pub fn new(d: usize, rng: &mut Rng) -> Self {
        Attention {
            wq: Linear::new(d, d, rng),
            wk: Linear::new(d, d, rng),
            wv: Linear::new(d, d, rng),
            wo: Linear::new(d, d, rng),
            // 1/sqrt(d): without this the dot products grow with the
            // dimension, softmax saturates, and gradients die before
            // learning anything.
            scale: 1.0 / (d as f32).sqrt(),
        }
    }
}

impl Module for Attention {
    fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let scores = ops::scale(&ops::matmul(&q, &ops::transpose(&k)), self.scale);
        let a = ops::softmax_causal(&scores);
        self.wo.forward(&ops::matmul(&a, &v))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.each(|l| l.parameters())
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out = Vec::new();
        for l in [&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo] {
            out.extend(l.parameters_mut());
        }
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut out = Vec::new();
        for (name, l) in self.named() {
            out.extend(l.named_parameters(&format!("{prefix}.{name}")));
        }
        out
    }

    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        let mut out = Vec::new();
        for (name, l) in [
            ("wq", &mut self.wq),
            ("wk", &mut self.wk),
            ("wv", &mut self.wv),
            ("wo", &mut self.wo),
        ] {
            out.extend(l.named_parameters_mut(&format!("{prefix}.{name}")));
        }
        out
    }
}

impl Attention {
    /// The four projections in order, so the list doesn't need repeating in
    /// every method. The order is part of the weight format: changing it
    /// invalidates saved checkpoints.
    fn named(&self) -> [(&'static str, &Linear); 4] {
        [("wq", &self.wq), ("wk", &self.wk), ("wv", &self.wv), ("wo", &self.wo)]
    }

    fn each<'a, T>(&'a self, f: impl Fn(&'a Linear) -> Vec<T>) -> Vec<T> {
        self.named().iter().flat_map(|(_, l)| f(l)).collect()
    }
}
