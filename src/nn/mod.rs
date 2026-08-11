use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)>;
    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)>;
}

fn uniform(len: usize, bound: f32, rng: &mut Rng) -> Vec<f32> {
    (0..len).map(|_| rng.uniform(-bound, bound)).collect()
}

fn zeros(len: usize) -> Vec<f32> {
    vec![0.0; len]
}

fn ones(len: usize) -> Vec<f32> {
    vec![1.0; len]
}

pub fn param(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    let mut t = Tensor::new(data, shape);
    t.requires_grad = true;
    t
}

pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
}

impl Linear {
    pub fn new(in_d: usize, out_d: usize, rng: &mut Rng) -> Self {
        let bound = 1.0 / (in_d as f32).sqrt();
        Linear {
            w: param(uniform(in_d * out_d, bound, rng), vec![in_d, out_d]),
            b: param(zeros(out_d), vec![out_d]),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        ops::add(&ops::matmul(x, &self.w), &self.b)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w, &self.b]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w, &mut self.b]
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![(format!("{}.w", prefix), &self.w), (format!("{}.b", prefix), &self.b)]
    }
    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        vec![(format!("{}.w", prefix), &mut self.w), (format!("{}.b", prefix), &mut self.b)]
    }
}

pub struct RMSNorm {
    pub w: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(d: usize, eps: f32) -> Self {
        RMSNorm { w: param(ones(d), vec![d]), eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        ops::rms_norm(x, &self.w, self.eps)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w]
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![(format!("{}.w", prefix), &self.w)]
    }
    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        vec![(format!("{}.w", prefix), &mut self.w)]
    }
}

pub struct GLUFFN {
    pub w1: Tensor,
    pub w2: Tensor,
    pub w3: Tensor,
}

impl GLUFFN {
    pub fn new(d: usize, f: usize, rng: &mut Rng) -> Self {
        let bound = 1.0 / (d as f32).sqrt();
        GLUFFN {
            w1: param(uniform(d * f, bound, rng), vec![d, f]),
            w2: param(uniform(d * f, bound, rng), vec![d, f]),
            w3: param(uniform(f * d, bound, rng), vec![f, d]),
        }
    }
}

impl Module for GLUFFN {
    fn forward(&self, x: &Tensor) -> Tensor {
        let a = ops::silu(&ops::matmul(x, &self.w1));
        let b = ops::matmul(x, &self.w2);
        ops::matmul(&ops::mul(&a, &b), &self.w3)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w1, &self.w2, &self.w3]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w1, &mut self.w2, &mut self.w3]
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.w1", prefix), &self.w1),
            (format!("{}.w2", prefix), &self.w2),
            (format!("{}.w3", prefix), &self.w3),
        ]
    }
    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        vec![
            (format!("{}.w1", prefix), &mut self.w1),
            (format!("{}.w2", prefix), &mut self.w2),
            (format!("{}.w3", prefix), &mut self.w3),
        ]
    }
}

pub struct Embedding {
    pub table: Tensor,
}

impl Embedding {
    pub fn new(vocab: usize, d: usize, rng: &mut Rng) -> Self {
        let bound = 1.0 / (d as f32).sqrt();
        Embedding { table: param(uniform(vocab * d, bound, rng), vec![vocab, d]) }
    }

    pub fn embed(&self, ids: &[usize]) -> Tensor {
        ops::gather(&self.table, ids)
    }
}

pub struct DepthwiseConv1d {
    pub w: Tensor,
    pub b: Tensor,
    pub k: usize,
}

impl DepthwiseConv1d {
    pub fn new(d: usize, k: usize, rng: &mut Rng) -> Self {
        let bound = 1.0 / (k as f32).sqrt();
        DepthwiseConv1d {
            w: param(uniform(d * k, bound, rng), vec![d, k]),
            b: param(zeros(d), vec![d]),
            k,
        }
    }
}

impl Module for DepthwiseConv1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        ops::depthwise_conv1d(x, &self.w, &self.b, self.k)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w, &self.b]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w, &mut self.b]
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![(format!("{}.w", prefix), &self.w), (format!("{}.b", prefix), &self.b)]
    }
    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        vec![(format!("{}.w", prefix), &mut self.w), (format!("{}.b", prefix), &mut self.b)]
    }
}
