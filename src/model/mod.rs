pub mod block;
pub mod clock;

use crate::model::block::EvaBlock;
use crate::nn::{param, Embedding, Module, RMSNorm};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct EvaConfig {
    pub vocab: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub blocks: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    pub seq_len: usize,
}

impl Default for EvaConfig {
    fn default() -> Self {
        EvaConfig {
            vocab: 256,
            dim: 256,
            ffn_dim: 512,
            blocks: 6,
            conv_kernel: 5,
            eps: 1e-5,
            seq_len: 64,
        }
    }
}

pub struct EvaModel {
    pub cfg: EvaConfig,
    pub embed: Embedding,
    pub blocks: Vec<EvaBlock>,
    pub norm_out: RMSNorm,
    pub head_w: Tensor,
}

impl EvaModel {
    pub fn new(cfg: EvaConfig) -> Self {
        let mut rng = Rng::new(0xE7A1);
        let dim = cfg.dim;
        let bound = 1.0 / (dim as f32).sqrt();
        EvaModel {
            embed: Embedding::new(cfg.vocab, dim, &mut rng),
            blocks: (0..cfg.blocks)
                .map(|_| EvaBlock::new(dim, cfg.ffn_dim, cfg.conv_kernel, cfg.eps, &mut rng))
                .collect(),
            norm_out: RMSNorm::new(dim, cfg.eps),
            head_w: param(
                (0..cfg.vocab * dim).map(|_| rng.uniform(-bound, bound)).collect(),
                vec![dim, cfg.vocab],
            ),
            cfg,
        }
    }

    pub fn forward(&self, ids: &[usize]) -> Tensor {
        let mut x = self.embed.embed(ids);
        for b in &self.blocks {
            x = b.forward(&x);
        }
        let x = self.norm_out.forward(&x);
        ops::matmul(&x, &self.head_w)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut out = Vec::new();
        out.push(&self.embed.table);
        for b in &self.blocks {
            out.extend(b.parameters());
        }
        out.push(&self.norm_out.w);
        out.push(&self.head_w);
        out
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out = Vec::new();
        out.push(&mut self.embed.table);
        for b in &mut self.blocks {
            out.extend(b.parameters_mut());
        }
        out.push(&mut self.norm_out.w);
        out.push(&mut self.head_w);
        out
    }

    pub fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut out = Vec::new();
        out.push(("embed.table".to_string(), &self.embed.table));
        for (i, b) in self.blocks.iter().enumerate() {
            out.extend(b.named_parameters(&format!("blocks.{}", i)));
        }
        out.push(("norm_out.w".to_string(), &self.norm_out.w));
        out.push(("head_w".to_string(), &self.head_w));
        out
    }

    pub fn named_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut out = Vec::new();
        out.push(("embed.table".to_string(), &mut self.embed.table));
        for (i, b) in self.blocks.iter_mut().enumerate() {
            out.extend(b.named_parameters_mut(&format!("blocks.{}", i)));
        }
        out.push(("norm_out.w".to_string(), &mut self.norm_out.w));
        out.push(("head_w".to_string(), &mut self.head_w));
        out
    }

    pub fn param_count(&self) -> usize {
        self.parameters().iter().map(|t| t.numel()).sum()
    }
}
