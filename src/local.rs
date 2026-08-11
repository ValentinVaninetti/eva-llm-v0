//! Crédito local: que ningún gradiente cruce de un bloque a otro.
//!
//! LA APUESTA. Para retropropagar hay que sostener las activaciones de toda la
//! pasada: la memoria de entrenamiento escala con **profundidad × ancho ×
//! largo**, y eso es lo que obliga a comprar hardware caro. No es el cómputo,
//! es la memoria.
//!
//! Si cada bloque tiene su propio objetivo y el gradiente no cruza, entrenar
//! una red de N bloques necesita la memoria de **uno**. Ninguna optimización
//! de kernel hace eso.
//!
//! CÓMO SE COBRA, que es la parte fácil de arruinar: no alcanza con desconectar
//! las entradas. Si se suman las pérdidas de todos los bloques y se hace UN
//! backward al final, el grafo sigue entero y no se ahorró nada. El beneficio
//! existe sólo si cada bloque **retropropaga, actualiza y libera** antes de que
//! el siguiente empiece. Por eso el bucle de acá abajo es como es.
//!
//! LA RAZÓN PARA DUDAR, y es fuerte: **las reglas locales rinden peor que
//! backprop en todo intento serio publicado.** No se propone porque vaya a
//! ganar. Se propone porque siempre se las probó sobre arquitecturas diseñadas
//! *para* backprop, nunca sobre un recurrente con decaimiento por canal donde
//! el crédito temporal ya es analítico y local; y porque si empata, el ahorro
//! es estructural. Se mide contra la línea base y se acepta el veredicto.

use crate::nn::{param, Module, RMSNorm};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// Una cabeza de predicción por bloque intermedio.
///
/// El último bloque no lleva: usa la salida real del modelo, así la inferencia
/// queda idéntica y la comparación con la línea base es sobre lo mismo.
///
/// Son andamio de entrenamiento y **no se guardan** en el checkpoint. Cuestan
/// `dim × vocab` cada una; con dim 256 y vocab 256 son 65 K por bloque
/// intermedio, y hay que contarlas al comparar costos aunque no queden en el
/// modelo final.
pub struct Heads {
    norms: Vec<RMSNorm>,
    projs: Vec<Tensor>,
}

impl Heads {
    pub fn new(intermedios: usize, dim: usize, vocab: usize, eps: f32, rng: &mut Rng) -> Self {
        let bound = 1.0 / (dim as f32).sqrt();
        Heads {
            norms: (0..intermedios).map(|_| RMSNorm::new(dim, eps)).collect(),
            projs: (0..intermedios)
                .map(|_| {
                    param(
                        (0..dim * vocab).map(|_| rng.uniform(-bound, bound)).collect(),
                        vec![dim, vocab],
                    )
                })
                .collect(),
        }
    }

    pub fn logits(&self, i: usize, y: &Tensor) -> Tensor {
        ops::matmul(&self.norms[i].forward(y), &self.projs[i])
    }

    pub fn params_mut(&mut self, i: usize) -> Vec<&mut Tensor> {
        let mut out = self.norms[i].parameters_mut();
        out.push(&mut self.projs[i]);
        out
    }

    pub fn count(&self) -> usize {
        self.projs.iter().map(|p| p.data.len()).sum::<usize>()
            + self.norms.iter().map(|n| n.w.data.len()).sum::<usize>()
    }
}

/// Pico de memoria residente del proceso, en MB.
///
/// Es el número que decide qué máquina hace falta. Se lee de `VmHWM`, que es
/// la marca de agua del kernel: no baja aunque después se libere, que es
/// exactamente lo que se quiere saber.
pub fn peak_rss_mb() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return f64::NAN;
    };
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = v.split_whitespace().next().and_then(|k| k.parse::<f64>().ok()) {
                return kb / 1024.0;
            }
        }
    }
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_says_something_plausible() {
        let mb = peak_rss_mb();
        assert!(mb > 0.1 && mb < 1_000_000.0, "pico de memoria absurdo: {mb}");
    }

    #[test]
    fn heads_produce_logits_of_the_right_shape() {
        let mut rng = Rng::new(1);
        let (dim, vocab, s) = (8, 5, 3);
        let h = Heads::new(2, dim, vocab, 1e-5, &mut rng);
        let y = param((0..s * dim).map(|i| i as f32 * 0.1).collect(), vec![s, dim]);
        let l = h.logits(0, &y);
        assert_eq!(vec![s, vocab], l.shape);
    }

    /// El gradiente de una cabeza NO puede llegar a lo que la alimentó si la
    /// entrada vino desconectada. Si esto falla, el crédito no es local y la
    /// medición de memoria mide otra cosa.
    #[test]
    fn a_detached_input_receives_no_gradient() {
        let mut rng = Rng::new(2);
        let (dim, vocab) = (6, 4);
        let h = Heads::new(1, dim, vocab, 1e-5, &mut rng);
        let upstream = param((0..2 * dim).map(|i| (i as f32) * 0.05).collect(), vec![2, dim]);

        let cortado = upstream.detach();
        let l = h.logits(0, &cortado);
        let grads = crate::tensor::autograd::backward(&ops::sum_all(&l));

        assert!(
            grads.get(&upstream.id).is_none(),
            "el gradiente cruzó el corte: el crédito no es local"
        );
    }
}
