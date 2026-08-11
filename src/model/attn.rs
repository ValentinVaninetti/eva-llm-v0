//! Atención causal de una cabeza. **Existe para tener contra qué medir.**
//!
//! No es la dirección del proyecto: es el patrón de referencia. Hasta que esto
//! estuvo, la única evidencia de que EvaClock funcionaba era que memorizaba un
//! corpus de 55 KB, cosa que hace cualquier cosa con suficientes parámetros.
//!
//! LA COMPARACIÓN ES JUSTA POR CONSTRUCCIÓN, que es todo el punto: ClockMem
//! tiene `wq, wk, wv, wg` y esto tiene `wq, wk, wv, wo` -- cuatro matrices DxD
//! en los dos casos, **exactamente la misma cantidad de parámetros**. Se
//! enchufa en el mismo bloque, con la misma conv, el mismo FFN, las mismas
//! normas y el mismo optimizador. Lo único que cambia es el mezclador temporal.
//!
//! Una sola cabeza a propósito, y no por pereza: ClockMem tampoco tiene
//! cabezas. Partir esto en cabezas y aquello no compararía los dos mecanismos
//! sino uno de ellos contra sí mismo con más maquinaria alrededor.
//!
//! La diferencia que se quiere ver es de fondo: acá el estado es una matriz
//! SxS que se construye entera (O(S²) en cómputo y memoria) y puede hacer
//! recuperación asociativa --clave A busca valor B--; ClockMem lleva un vector
//! de D (O(S·D), O(D) en inferencia) y no puede. La pregunta es cuánto vale esa
//! diferencia en pérdida real, no en principio.

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
            // 1/sqrt(d): sin esto los productos crecen con la dimensión, el
            // softmax se satura y los gradientes se mueren antes de aprender.
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
    /// Las cuatro proyecciones en orden, para no repetir la lista en cada
    /// método. El orden es parte del formato de pesos: cambiarlo invalida los
    /// checkpoints guardados.
    fn named(&self) -> [(&'static str, &Linear); 4] {
        [("wq", &self.wq), ("wk", &self.wk), ("wv", &self.wv), ("wo", &self.wo)]
    }

    fn each<'a, T>(&'a self, f: impl Fn(&'a Linear) -> Vec<T>) -> Vec<T> {
        self.named().iter().flat_map(|(_, l)| f(l)).collect()
    }
}
