use crate::model::attn::Attention;
use crate::model::clock::ClockMem;
use crate::model::Arch;
use crate::nn::{DepthwiseConv1d, GLUFFN, Module, RMSNorm};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

pub struct EvaBlock {
    pub norm0: RMSNorm,
    pub conv: DepthwiseConv1d,
    pub norm1: RMSNorm,
    /// El mezclador temporal, que es LA variable del experimento. Todo lo
    /// demás del bloque es idéntico entre arquitecturas a propósito: si
    /// cambiara algo más, la comparación no diría cuál de los dos cambios fue.
    pub mixer: Box<dyn Module>,
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
                Arch::Clock => Box::new(ClockMem::new(d, rng)) as Box<dyn Module>,
                Arch::Attn => Box::new(Attention::new(d, rng)),
            },
            norm2: RMSNorm::new(d, eps),
            glu: GLUFFN::new(d, ffn, rng),
        }
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
