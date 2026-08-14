use crate::nn::{param, Linear, Module};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// Ronda 4, hipótesis de GPT: ¿el achatamiento de `alpha` es saturación
/// del sigmoid (medido: |grad| 20-100x más chico donde alpha≈0), no
/// preferencia de la loss? Con esto activo, ClockMem usa
/// `ops::algebraic_sigmoid` (cola polinómica) en vez de `ops::sigmoid`
/// (cola exponencial) para todo lo demás idéntico -- misma arquitectura,
/// mismo rango de `alpha`, mismo `EVA_ALPHA_MAX`, sólo cambia cuánta
/// señal de gradiente sobrevive cerca de los extremos.
fn antisat() -> bool {
    std::env::var("EVA_ALPHA_ANTISAT").is_ok()
}

/// Ronda 4, segunda intervención (a pedido de GPT, tras encontrar que
/// `algebraic_sigmoid` no tocaba la región correcta): temperatura sobre
/// EL MISMO sigmoid, `alpha=sigmoid(z/T)` con T>1. A diferencia de
/// `algebraic_sigmoid`, esto NO cambia la forma de la curva -- estira el
/// mismo sigmoid, así que a un `z` dado (el mismo que ya tenía cualquier
/// canal) le corresponde MÁS derivada, verificado con script antes de
/// tocar código: en z=-3..-5 (donde vive la banda rápida hoy), T=1.3 da
/// ~1.4x-2.4x más derivada que T=1. Elegí T=1.3 explícitamente para eso,
/// no un valor redondo porque sí.
///
/// TRADE-OFF explícito, no escondido: para que "sólo cambie T" sea
/// literal en el código (mismos valores de `log_clock` al arrancar, cero
/// cambio en la inicialización), el `alpha` INICIAL se corre un poco
/// (menos extremo -- con T=1.3, alpha=0.018 al init pasa a ≈0.044). No
/// hay forma de tener EXACTAMENTE el mismo alpha inicial Y más derivada
/// ahí a la vez con una sola familia de reparametrización (es la misma
/// razón por la que `algebraic_sigmoid` con inversa ajustada terminó
/// dándole MENOS señal a la banda rápida, no más). Se opta acá por
/// preservar el `z` (lo que el optimizador realmente ve), no el `alpha`.
fn temperatura() -> f32 {
    std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0)
}

fn alpha_squash(z: &Tensor) -> Tensor {
    if antisat() {
        ops::algebraic_sigmoid(z)
    } else {
        let t = temperatura();
        if t != 1.0 { ops::sigmoid(&ops::scale(z, 1.0 / t)) } else { ops::sigmoid(z) }
    }
}

/// Inversa de `alpha_squash`, sólo para inicializar `log_clock` apuntando
/// al mismo `alpha` objetivo sin importar qué squashing esté activo -- si
/// no, cambiar el squashing también cambiaría el rango inicial y ya no
/// sería una sola variable entre las dos condiciones.
fn alpha_squash_inv(a: f32) -> f32 {
    if antisat() {
        // s(z)=0.5(1+z/sqrt(1+z^2))  =>  z = (2a-1) / (2*sqrt(a(1-a)))
        (2.0 * a - 1.0) / (2.0 * (a * (1.0 - a)).sqrt())
    } else {
        (a / (1.0 - a)).ln()
    }
}

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
                        alpha_squash_inv(a)
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
        let alpha = alpha_squash(&self.log_clock);
        ops::clockmem_from(&q, &k, &v, &g, &alpha, &self.beta, s0)
    }
}

impl Module for ClockMem {
    fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let g = ops::sigmoid(&self.wg.forward(x));
        let alpha = alpha_squash(&self.log_clock);
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
