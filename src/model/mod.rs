pub mod attn;
pub mod block;
pub mod clock;

use crate::model::block::EvaBlock;
use crate::nn::{param, Embedding, Module, RMSNorm};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// Qué mezcla la información entre posiciones. Es la única diferencia entre
/// las dos arquitecturas que sabe construir este archivo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arch {
    /// EvaClock: estado recurrente de D con reloj por canal. O(S·D).
    Clock,
    /// Atención causal de una cabeza. O(S²). Está para comparar, no es la
    /// dirección del proyecto.
    Attn,
}

impl Arch {
    pub fn from_str(s: &str) -> Result<Arch, String> {
        match s {
            "clock" => Ok(Arch::Clock),
            "attn" | "attention" | "transformer" => Ok(Arch::Attn),
            other => Err(format!("arquitectura desconocida: '{other}' (clock o attn)")),
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
    /// Posiciones absolutas aprendidas, SÓLO para la atención.
    ///
    /// ClockMem codifica la posición gratis en el decaimiento `α^(t-i)`: la
    /// cercanía está adentro del mecanismo. La atención sin esto no distingue
    /// orden, sólo contenido, y compararlas así es ganarle a un rival con una
    /// mano atada.
    ///
    /// Le suma seq_len*dim (16 K de 2.77 M, 0.6%) que ClockMem no tiene. La
    /// desventaja queda de nuestro lado a propósito: si igual gana ClockMem,
    /// el resultado vale más.
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
        let mut x = self.embed.embed(ids);
        if let Some(pos) = &self.pos {
            // En generación la ventana crece de a un token, así que el corte
            // no es decorativo.
            let n = ids.len().min(self.cfg.seq_len);
            x = ops::add(&x, &ops::slice_rows(pos, n));
        }
        for b in &self.blocks {
            x = b.forward(&x);
        }
        let x = self.norm_out.forward(&x);
        ops::matmul(&x, &self.head_w)
    }

    /// Pasada llevando el estado de ClockMem de una ventana a la siguiente.
    ///
    /// `states` entra con lo que dejó la ventana anterior y sale con lo que le
    /// deja a la próxima. Son D números por bloque, de tamaño fijo: por eso
    /// esto es gratis y un caché de atención no lo sería.
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

    /// Estados en cero, uno por bloque.
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
