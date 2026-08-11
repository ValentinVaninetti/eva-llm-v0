use crate::nn::{param, Linear, Module};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

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
                // TECHO DEL RELOJ. Con 0.9999 un canal olvida tan despacio que
                // acumula ~10.000 términos: dentro de una ventana de 64 tokens
                // es inofensivo, pero con estado persistente a lo largo del
                // corpus SATURA -- medido, la magnitud del estado llegó a 881.
                // Con 0.999 la memoria efectiva es de ~1000 tokens, quince
                // veces la ventana, que es justo lo que se quiere sin explotar.
                let a_max = std::env::var("EVA_ALPHA_MAX")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.9999);
                let a_min = 0.01f32;
                let logits: Vec<f32> = (0..d)
                    .map(|i| {
                        let t = if d == 1 { 1.0 } else { i as f32 / (d - 1) as f32 };
                        let a = a_max * (a_min / a_max).powf(t);
                        (a / (1.0 - a)).ln()
                    })
                    .collect();
                param(logits, vec![d])
            },
            beta: param(vec![1.0], vec![1]),
        }
    }
}

impl ClockMem {
    /// Igual que `forward`, arrancando del estado que dejó la ventana anterior
    /// y devolviendo el que queda para la siguiente.
    pub fn forward_from(&self, x: &Tensor, s0: &[f32]) -> (Tensor, Vec<f32>) {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let g = ops::sigmoid(&self.wg.forward(x));
        let alpha = ops::sigmoid(&self.log_clock);
        ops::clockmem_from(&q, &k, &v, &g, &alpha, &self.beta, s0)
    }
}

impl Module for ClockMem {
    fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let g = ops::sigmoid(&self.wg.forward(x));
        let alpha = ops::sigmoid(&self.log_clock);
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
