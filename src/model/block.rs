use crate::model::attn::Attention;
use crate::model::clock::ClockMem;
use crate::model::Arch;
use crate::nn::{DepthwiseConv1d, GLUFFN, Module, RMSNorm};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// The two possible mixers. The comparison between them is the project's
/// central experiment, so they live at the same level and behind the same
/// interface.
pub enum Mixer {
    Clock(ClockMem),
    Attn(Attention),
}

impl Mixer {
    /// Returns the final state only if the mixer has fixed-size state.
    /// Attention returns `None` on purpose: its "state" is a cache that
    /// grows with context, so chaining it across windows wouldn't be free
    /// and the comparison would stop being fair.
    pub fn forward_from(&self, x: &Tensor, s0: Option<&[f32]>) -> (Tensor, Option<Vec<f32>>) {
        match (self, s0) {
            (Mixer::Clock(m), Some(s)) => {
                let (y, fin) = m.forward_from(x, s);
                (y, Some(fin))
            }
            _ => (self.forward(x), None),
        }
    }
}

impl Module for Mixer {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Mixer::Clock(m) => m.forward(x),
            Mixer::Attn(m) => m.forward(x),
        }
    }
    fn parameters(&self) -> Vec<&Tensor> {
        match self {
            Mixer::Clock(m) => m.parameters(),
            Mixer::Attn(m) => m.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        match self {
            Mixer::Clock(m) => m.parameters_mut(),
            Mixer::Attn(m) => m.parameters_mut(),
        }
    }
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        match self {
            Mixer::Clock(m) => m.named_parameters(prefix),
            Mixer::Attn(m) => m.named_parameters(prefix),
        }
    }
    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        match self {
            Mixer::Clock(m) => m.named_parameters_mut(prefix),
            Mixer::Attn(m) => m.named_parameters_mut(prefix),
        }
    }
}

pub struct EvaBlock {
    pub norm0: RMSNorm,
    pub conv: DepthwiseConv1d,
    pub norm1: RMSNorm,
    /// The temporal mixer, which is THE variable of the experiment.
    /// Everything else in the block is deliberately identical between
    /// architectures: if anything else changed too, the comparison wouldn't
    /// tell you which of the two changes actually mattered.
    ///
    /// An enum and not `Box<dyn Module>`: stateful inference needs to know
    /// WHICH ONE it is in order to carry the matching state --a fixed
    /// D-sized vector for ClockMem, a growing cache for attention-- and that
    /// can't be asked through a trait object. As a side effect it also keeps
    /// dynamic dispatch off the hot path.
    pub mixer: Mixer,
    pub norm2: RMSNorm,
    pub glu: GLUFFN,
}

impl EvaBlock {
    pub fn new(d: usize, ffn: usize, kernel: usize, eps: f32, arch: Arch, rng: &mut Rng) -> Self {
        EvaBlock {
            norm0: RMSNorm::new(d, eps),
            conv: DepthwiseConv1d::new(d, kernel, rng),
            norm1: RMSNorm::new(d, eps),
            mixer: match arch {
                Arch::Clock => Mixer::Clock(ClockMem::new(d, rng)),
                Arch::Attn => Mixer::Attn(Attention::new(d, rng)),
            },
            norm2: RMSNorm::new(d, eps),
            glu: GLUFFN::new(d, ffn, rng),
        }
    }
}

impl EvaBlock {
    pub fn forward_from(&self, x: &Tensor, s0: Option<&[f32]>) -> (Tensor, Option<Vec<f32>>) {
        let h = ops::add(x, &self.conv.forward(&self.norm0.forward(x)));
        let (m, fin) = self.mixer.forward_from(&self.norm1.forward(&h), s0);
        let h = ops::add(&h, &m);
        (ops::add(&h, &self.glu.forward(&self.norm2.forward(&h))), fin)
    }
}

impl Module for EvaBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = ops::add(x, &self.conv.forward(&self.norm0.forward(x)));
        let h = ops::add(&h, &self.mixer.forward(&self.norm1.forward(&h)));
        ops::add(&h, &self.glu.forward(&self.norm2.forward(&h)))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut out = Vec::new();
        out.extend(self.norm0.parameters());
        out.extend(self.conv.parameters());
        out.extend(self.norm1.parameters());
        out.extend(self.mixer.parameters());
        out.extend(self.norm2.parameters());
        out.extend(self.glu.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out = Vec::new();
        out.extend(self.norm0.parameters_mut());
        out.extend(self.conv.parameters_mut());
        out.extend(self.norm1.parameters_mut());
        out.extend(self.mixer.parameters_mut());
        out.extend(self.norm2.parameters_mut());
        out.extend(self.glu.parameters_mut());
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut out = Vec::new();
        out.extend(self.norm0.named_parameters(&format!("{}.norm0", prefix)));
        out.extend(self.conv.named_parameters(&format!("{}.conv", prefix)));
        out.extend(self.norm1.named_parameters(&format!("{}.norm1", prefix)));
        out.extend(self.mixer.named_parameters(&format!("{}.clock", prefix)));
        out.extend(self.norm2.named_parameters(&format!("{}.norm2", prefix)));
        out.extend(self.glu.named_parameters(&format!("{}.glu", prefix)));
        out
    }

    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        let mut out = Vec::new();
        out.extend(self.norm0.named_parameters_mut(&format!("{}.norm0", prefix)));
        out.extend(self.conv.named_parameters_mut(&format!("{}.conv", prefix)));
        out.extend(self.norm1.named_parameters_mut(&format!("{}.norm1", prefix)));
        out.extend(self.mixer.named_parameters_mut(&format!("{}.clock", prefix)));
        out.extend(self.norm2.named_parameters_mut(&format!("{}.norm2", prefix)));
        out.extend(self.glu.named_parameters_mut(&format!("{}.glu", prefix)));
        out
    }
}
