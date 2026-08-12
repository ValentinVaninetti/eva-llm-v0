//! 8. QUE APUESTE — la cabeza de stake por tramo.
//!
//! El experimento decisivo que quedó definido con Claudio: ¿una cabeza que lee
//! el estado oculto AL ARRANCAR el tramo predice si el tramo va a salir bien,
//! mejor que la vara libre (media de p[argmax] ≈ 0.67) y antes de gastar?
//!
//! La cabeza es una sonda: una proyección lineal del estado oculto a un stake
//! en (0,1), entrenada con `stake_loss` (la op nueva, forward+backward+
//! gradcheck, en el estilo del contrato). El estado oculto entra DETACHED: la
//! sonda no le manda gradiente al modelo. Que el gradiente cruce es la decisión
//! 4, más cara, y se mide recién si acá hay señal.
//!
//! Es andamio como `local::Heads`: no se guarda en el checkpoint, el modelo
//! queda intacto.

use crate::model::EvaModel;
use crate::nn::param;
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// La cabeza de stake: `s = sigmoid(w·h + b)`. D números + 1 bias.
pub struct StakeHead {
    pub w: Tensor,
    pub b: Tensor,
}

impl StakeHead {
    pub fn new(dim: usize, rng: &mut Rng) -> Self {
        let bound = 1.0 / (dim as f32).sqrt();
        StakeHead {
            w: param((0..dim).map(|_| rng.uniform(-bound, bound)).collect(), vec![dim]),
            b: param(vec![0.0], vec![1]),
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w, &mut self.b]
    }

    pub fn count(&self) -> usize {
        self.w.data.len() + self.b.data.len()
    }

    /// El stake de cada tramo de una ventana, sobre el estado oculto real.
    pub fn stakes(&self, hidden: &Tensor, span_len: usize) -> Vec<f32> {
        let d = hidden.shape[1];
        let s = hidden.shape[0];
        let mut out = Vec::new();
        let mut k = 0;
        while (k + 1) * span_len <= s {
            let start = k * span_len;
            let mut z = self.b.data[0];
            for j in 0..d {
                z += self.w.data[j] * hidden.data[start * d + j];
            }
            out.push(1.0 / (1.0 + (-z).exp()));
            k += 1;
        }
        out
    }
}

/// Cuánto salió bien cada tramo de una ventana: la fracción de posiciones
/// donde el argmax del modelo acertó. Es el `bien` del md, y es un dato, no
/// algo diferenciable: entra a la pérdida por `saved_f`.
pub fn bien_por_tramo(logits: &Tensor, targets: &[usize], span_len: usize) -> Vec<f32> {
    let (s, v) = (logits.shape[0], logits.shape[1]);
    let mut out = Vec::new();
    let mut k = 0;
    while (k + 1) * span_len <= s {
        let mut aciertos = 0usize;
        for t in k * span_len..(k + 1) * span_len {
            let row = &logits.data[t * v..(t + 1) * v];
            let argmax = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
                .0;
            if argmax == targets[t] {
                aciertos += 1;
            }
        }
        out.push(aciertos as f32 / span_len as f32);
        k += 1;
    }
    out
}

/// La representación que alimenta la sonda: el estado oculto, detached, en
/// su forma cruda (pre-norma) o la que el modelo usa para predecir
/// (post-norma, RMSNorm + ganancia). La elección se mide porque el modelo
/// predice sobre la segunda; si la señal está, tiene que estar ahí.
fn representacion(model: &EvaModel, hidden: &Tensor, post_norm: bool) -> Tensor {
    if !post_norm {
        return hidden.detach();
    }
    // Se aplica a mano sobre los datos detached: correr `norm_out.forward`
    // dejaría un nodo que apunta al w del modelo, y al backpropagear el
    // stake_loss entrenaría la norma del modelo -- justo lo que la sonda no
    // puede tocar.
    let (s, d) = (hidden.shape[0], hidden.shape[1]);
    let eps = model.cfg.eps;
    let wn = &model.norm_out.w.data;
    let mut out = vec![0.0; s * d];
    for i in 0..s {
        let mut sq = 0.0;
        for j in 0..d {
            let v = hidden.data[i * d + j];
            sq += v * v;
        }
        let inv = 1.0 / (sq / d as f32 + eps).sqrt();
        for j in 0..d {
            out[i * d + j] = hidden.data[i * d + j] * inv * wn[j];
        }
    }
    Tensor::new(out, vec![s, d])
}

/// Entrena la sonda y la mide. Devuelve (r en entrenamiento, r en validación,
/// n de tramos de validación).
///
/// El modelo queda congelado: nunca toca sus parámetros. Sólo `w` y `b` se
/// actualizan, a través del `stake_loss` del grafo.
pub fn entrenar_y_medir(
    model: &EvaModel,
    ds: &crate::data::TextDataset,
    n_train: usize,
    n_val: usize,
    span_len: usize,
    epochs: usize,
    lr: f32,
    seed: u64,
    post_norm: bool,
) -> (f32, f32, usize) {
    let mut rng = Rng::new(seed);
    let mut head = StakeHead::new(model.cfg.dim, &mut rng);
    let mut opt = crate::optim::AdamW::new(lr, 0.0);

    for epoch in 0..epochs {
        let order = ds.shuffled_train_indices(n_train, &mut rng);
        let mut sum = 0.0f32;
        for (i, &wi) in order.iter().enumerate() {
            let (input, target) = ds.window(wi);
            let (logits, hidden) = model.forward_hidden(&input);
            let bien = bien_por_tramo(&logits, &target, span_len);
            // El estado oculto entra desconectado: la sonda no manda gradiente
            // al modelo. Es el punto entero de la medición.
            let h = representacion(model, &hidden, post_norm);
            let loss = ops::stake_loss(&h, &head.w, &head.b, &bien, span_len);
            let grads = crate::tensor::autograd::backward(&loss);
            let mut params = head.params_mut();
            opt.step(&mut params, &grads);
            sum += loss.data[0];
            if i % 500 == 0 {
                println!("  stake epoch {} | paso {} | loss {:.4}", epoch + 1, i, loss.data[0]);
            }
        }
        println!("  stake epoch {} | loss media {:.4}", epoch + 1, sum / order.len().max(1) as f32);
    }

    // Medición sobre las ventanas de entrenamiento (¿memorizó ruido?) y las de
    // validación (las que el modelo nunca vio, el número que vale).
    let (r_train, _) = medir(&head, model, ds, 0, n_train, span_len, post_norm);
    let (r_val, n_val_spans) = medir(&head, model, ds, n_train, n_train + n_val, span_len, post_norm);
    println!("  cabeza: {} params (w+b) | r(stake vs bien) train {:.3} | val {:.3} sobre {n_val_spans} tramos",
        head.count(), r_train, r_val);
    (r_train, r_val, n_val_spans)
}

/// r(stake, bien) de la cabeza sobre las ventanas [desde, hasta).
pub fn medir(
    head: &StakeHead,
    model: &EvaModel,
    ds: &crate::data::TextDataset,
    desde: usize,
    hasta: usize,
    span_len: usize,
    post_norm: bool,
) -> (f32, usize) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for wi in desde..hasta {
        let (input, target) = ds.window(wi);
        let (logits, hidden) = model.forward_hidden(&input);
        let bien = bien_por_tramo(&logits, &target, span_len);
        let h = representacion(model, &hidden, post_norm);
        let stakes = head.stakes(&h, span_len);
        for (x, y) in stakes.iter().zip(&bien) {
            xs.push(*x as f64);
            ys.push(*y as f64);
        }
    }
    (crate::bet::pearson(&xs, &ys), xs.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bien_cuenta_el_argmax_no_la_probabilidad() {
        // Cada fila tiene un único máximo; bien mira dónde cae.
        let logits = param(
            vec![0.0, 1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0],
            vec![4, 3],
        );
        let targets = vec![1usize, 2, 0, 1];
        let bien = bien_por_tramo(&logits, &targets, 2);
        assert_eq!(bien, vec![1.0, 1.0], "ambas filas de cada tramo aciertan");
    }

    #[test]
    fn bien_se_qeda_con_la_fraccion() {
        let logits = param(
            vec![0.0, 1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 1.0],
            vec![4, 3],
        );
        let targets = vec![1usize, 0, 0, 2];
        let bien = bien_por_tramo(&logits, &targets, 2);
        assert_eq!(bien, vec![0.5, 1.0], "el tramo 0 acierta 1 de 2");
    }

    #[test]
    fn los_tramos_no_usan_sobras() {
        // s=5, span=2 → 2 tramos completos; la posición 4 se descarta.
        let mut rng = Rng::new(3);
        let head = StakeHead::new(3, &mut rng);
        let hidden = param((0..15).map(|i| i as f32 * 0.1).collect(), vec![5, 3]);
        assert_eq!(head.stakes(&hidden, 2).len(), 2);
    }

    /// Sanity del pipeline completo: si la señal está en el estado, la sonda
    /// la encuentra. Sin este test, un r≈0 en el modelo real podría ser un
    /// pipeline roto y no un resultado.
    #[test]
    fn la_sonda_aprende_una_senal_que_si_esta() {
        let (d, s, span) = (8usize, 8usize, 4usize);
        let mut rng = Rng::new(7);
        let mut head = StakeHead::new(d, &mut rng);
        let mut opt = crate::optim::AdamW::new(1e-2, 0.0);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..400 {
            let bien = if rng.uniform(0.0, 1.0) > 0.5 { 1.0 } else { 0.0 };
            // El arranque de cada tramo (filas 0 y d) codifica `bien` en la
            // coordenada 0; el resto es ruido.
            let mut data = vec![0.0; s * d];
            data[0] = bien * 4.0 - 2.0;
            data[d] = bien * 4.0 - 2.0;
            let hidden = Tensor::new(data, vec![s, d]);
            let loss = ops::stake_loss(&hidden, &head.w, &head.b, &[bien, bien], span);
            let grads = crate::tensor::autograd::backward(&loss);
            let mut params = head.params_mut();
            opt.step(&mut params, &grads);
            xs.push(head.stakes(&hidden, span)[0] as f64);
            ys.push(bien as f64);
        }
        let r = crate::bet::pearson(&xs, &ys);
        assert!(r > 0.9, "la sonda no aprendió una señal que está: r={r}");
    }
}
