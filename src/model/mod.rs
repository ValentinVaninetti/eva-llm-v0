pub mod attn;
pub mod block;
pub mod clock;

use crate::model::block::EvaBlock;
use crate::nn::{param, Embedding, Module, RMSNorm};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// What mixes information across positions. It's the only difference between
/// the two architectures this file knows how to build.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arch {
    /// EvaClock: D-sized recurrent state with a per-channel clock. O(S*D).
    Clock,
    /// Single-head causal attention. O(S^2). It's there to compare against,
    /// not the direction of the project.
    Attn,
}

impl Arch {
    pub fn from_str(s: &str) -> Result<Arch, String> {
        match s {
            "clock" => Ok(Arch::Clock),
            "attn" | "attention" | "transformer" => Ok(Arch::Attn),
            other => Err(format!("unknown architecture: '{other}' (clock or attn)")),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Arch::Clock => 0,
            Arch::Attn => 1,
        }
    }

    pub fn from_u32(v: u32) -> Arch {
        if v == 1 { Arch::Attn } else { Arch::Clock }
    }

    pub fn name(self) -> &'static str {
        match self {
            Arch::Clock => "clock",
            Arch::Attn => "attn",
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvaConfig {
    pub vocab: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub blocks: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    pub seq_len: usize,
    pub arch: Arch,
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
            arch: Arch::Clock,
        }
    }
}

pub struct EvaModel {
    pub cfg: EvaConfig,
    pub embed: Embedding,
    /// Learned absolute positions, ONLY for attention.
    ///
    /// ClockMem encodes position for free in the decay `alpha^(t-i)`:
    /// closeness lives inside the mechanism itself. Attention without this
    /// doesn't distinguish order, only content, and comparing them that way
    /// would mean beating a rival with one hand tied behind its back.
    ///
    /// This adds seq_len*dim (16K out of 2.77M, 0.6%) that ClockMem doesn't
    /// have. The disadvantage is deliberately on our side: if ClockMem still
    /// wins, the result is worth more.
    pub pos: Option<Tensor>,
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
            pos: (cfg.arch == Arch::Attn).then(|| {
                param(
                    (0..cfg.seq_len * dim).map(|_| rng.uniform(-bound, bound)).collect(),
                    vec![cfg.seq_len, dim],
                )
            }),
            blocks: (0..cfg.blocks)
                .map(|_| EvaBlock::new(dim, cfg.ffn_dim, cfg.conv_kernel, cfg.eps, cfg.arch, &mut rng))
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
        let (logits, _) = self.forward_hidden(ids);
        logits
    }

    /// Same as `forward`, but also returns the hidden state BEFORE the
    /// output RMSNorm.
    ///
    /// The magnitude of that vector is signal #4 (length as confidence).
    /// It's measured pre-norm on purpose: RMSNorm flattens length by
    /// construction (the entire point of decision 4 is that this
    /// information gets thrown away), so measuring it post-norm would be
    /// measuring noise.
    pub fn forward_hidden(&self, ids: &[usize]) -> (Tensor, Tensor) {
        self.forward_skip(ids, None)
    }

    /// Same as `forward_hidden`, but with the option to SKIP a block: the
    /// signal used by the `ceiling` benchmark (the retrospective compute
    /// ceiling via influence).
    ///
    /// `skip = None` is exactly `forward_hidden`. `skip = Some(i)` lets the
    /// residual stream pass through block i as identity: whatever the model
    /// loses that way is what that block contributes. This lives in the
    /// model and not in the benchmark on purpose: the block's logic is
    /// defined exactly once, in its `forward`, and this only decides
    /// whether it gets called.
    pub fn forward_skip(&self, ids: &[usize], skip: Option<usize>) -> (Tensor, Tensor) {
        match skip {
            Some(i) => self.forward_skips(ids, &[i]),
            None => self.forward_skips(ids, &[]),
        }
    }

    /// Experimental variant of `forward_hidden` that skips several blocks.
    /// Only `ceiling` uses it, to measure a REAL combination in one pass;
    /// it's not an inference policy and doesn't change the normal path.
    pub fn forward_skips(&self, ids: &[usize], skips: &[usize]) -> (Tensor, Tensor) {
        let mut x = self.embed.embed(ids);
        if let Some(pos) = &self.pos {
            // During generation the window grows one token at a time, so
            // the cut isn't decorative.
            let n = ids.len().min(self.cfg.seq_len);
            x = ops::add(&x, &ops::slice_rows(pos, n));
        }
        for (i, b) in self.blocks.iter().enumerate() {
            if skips.contains(&i) {
                continue;
            }
            x = b.forward(&x);
        }
        let hidden = x.clone();
        let x = self.norm_out.forward(&x);
        let logits = ops::matmul(&x, &self.head_w);
        (logits, hidden)
    }

    /// A pass that carries ClockMem's state from one window to the next.
    ///
    /// `states` comes in with what the previous window left behind and goes
    /// out with what it leaves for the next one. It's D numbers per block,
    /// fixed size: that's why this is free and an attention cache wouldn't
    /// be.
    pub fn forward_carrying(&self, ids: &[usize], states: &mut [Vec<f32>]) -> Tensor {
        let mut x = self.embed.embed(ids);
        if let Some(pos) = &self.pos {
            let n = ids.len().min(self.cfg.seq_len);
            x = ops::add(&x, &ops::slice_rows(pos, n));
        }
        for (i, b) in self.blocks.iter().enumerate() {
            let (y, fin) = b.forward_from(&x, Some(&states[i]));
            x = y;
            if let Some(f) = fin {
                states[i] = f;
            }
        }
        let x = self.norm_out.forward(&x);
        ops::matmul(&x, &self.head_w)
    }

    /// Zeroed states, one per block.
    pub fn fresh_states(&self) -> Vec<Vec<f32>> {
        vec![vec![0.0; self.cfg.dim]; self.blocks.len()]
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut out = Vec::new();
        out.push(&self.embed.table);
        if let Some(p) = &self.pos { out.push(p); }

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
        if let Some(p) = &mut self.pos { out.push(p); }

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
        if let Some(p) = &self.pos {
            out.push(("pos".to_string(), p));
        }
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
        if let Some(p) = &mut self.pos {
            out.push(("pos".to_string(), p));
        }
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
