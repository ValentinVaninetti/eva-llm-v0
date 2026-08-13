//! 5. EL CUADERNO ADENTRO — la tabla de k-gramas DENTRO del grafo.
//!
//! En `recall.rs` la mezcla es por FUERA: el modelo corre entero, se guardan
//! los logits, y después se mezclan con la tabla. El modelo nunca se entera de
//! dónde la tabla lo cubre, así que siguió gastando capacidad en memorizar lo
//! que la tabla ya sabe de memoria.
//!
//! Acá la mezcla vive EN el grafo: la pérdida es `mix = (1−λ)p + λq` con `q`
//! la distribución de la tabla. El gradiente que vuelve al modelo está
//! escalado por `(1−λ)·p[tgt]/mix[tgt]`: atenuado donde la tabla acierta (esa
//! capacidad queda libre para generalizar), prendido donde falla. La tabla NO
//! aprende: `q` entra como dato, gradiente solo a logits.
//!
//! Es un andamio de entrenamiento como `local.rs`: no toca el modelo, no se
//! guarda en el checkpoint, y con `λ=0` el entrenamiento es EXACTAMENTE el de
//! hoy (la op `mixed_ce` colapsa a `cross_entropy`, valor y gradiente).
//!
//! REGLAS HEREDADAS DE `recall.rs`:
//! - La tabla se construye SÓLO con bytes de entrenamiento.
//! - El contexto de cada posición son los bytes REALES anteriores del corpus
//!   (lo que tendría un generador en ese punto), no lo que el modelo ve.
//! - λ se elige sobre dev, nunca validación (ya lo barre `cmd_recall`).
//! - El veredicto se toma con el modelo SOLO (bpb sin mezcla): reportar la
//!   mezcla disfraza el número.

use crate::data::TextDataset;
use crate::recall::Recall;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// La tabla ya construida + λ. Vive por afuera del modelo: es andamio.
///
/// Dos modos:
/// - `build` (gate = None): λ constante, `mixed_ce` — el barrido de hoy.
/// - `build_count` (gate = Some(c0)): λ por count, `mixed_ce_count` — el
///   gate que deja el gradiente entero en los hechos raros (count en el
///   piso) y sólo atenúa el boilerplate repetido.
pub struct Cuaderno {
    tabla: Recall,
    lambda: f32,
    gate: Option<f32>,
    seq: usize,
}

impl Cuaderno {
    /// La tabla se arma UNA vez, con las primeras `n_train·seq` ventanas del
    /// corpus — exactamente las ventanas de entrenamiento. Que entre un byte
    /// de validación invalida todo el experimento.
    pub fn build(ds: &TextDataset, n_train: usize, lambda: f32) -> Self {
        Self::nuevo(ds, n_train, lambda, None)
    }

    /// Igual que `build` pero con el gate por count: `λ_i = λ·(1 − c0/c_i)`.
    /// Con `c0 = MIN_COUNT` (2) el piso de la tabla es cross_entropy exacto.
    pub fn build_count(ds: &TextDataset, n_train: usize, lambda: f32, c0: f32) -> Self {
        Self::nuevo(ds, n_train, lambda, Some(c0))
    }

    fn nuevo(ds: &TextDataset, n_train: usize, lambda: f32, gate: Option<f32>) -> Self {
        let n = n_train * ds.seq;
        let train: Vec<usize> = ds.ids[..n].to_vec();
        let tabla = Recall::build(&train);
        Cuaderno { tabla, lambda, gate, seq: ds.seq }
    }

    /// Cuántas entradas tiene, para imprimir lo que "pesa" contra los
    /// parámetros que reemplaza.
    pub fn count(&self) -> usize {
        self.tabla.entries()
    }

    pub fn bytes(&self) -> usize {
        self.tabla.bytes()
    }

    /// La pérdida de una ventana con la mezcla DENTRO del grafo.
    ///
    /// Para cada posición el contexto de la tabla son los bytes reales
    /// anteriores del corpus: `ds.ids[..wi·seq + t + 1]`. El `+1` no es un
    /// capricho: la ventana arranca en el byte siguiente a `input[0]`, así que
    /// al predecir `target[t]` un generador ya vio `input[0..=t]`.
    pub fn loss(&self, logits: &Tensor, ds: &TextDataset, wi: usize, target: &[usize]) -> Tensor {
        let (s, v) = (logits.shape[0], logits.shape[1]);
        debug_assert_eq!(s, self.seq, "logits de {s} posiciones, ventana de {}", self.seq);
        let mut q = vec![0.0f32; s * v];
        let mut hit = vec![0.0f32; s];
        let base = wi * self.seq;
        match self.gate {
            None => {
                for t in 0..s {
                    if let Some(p) = self.tabla.lookup(&ds.ids[..base + t + 1], v) {
                        q[t * v..(t + 1) * v].copy_from_slice(&p);
                        hit[t] = 1.0;
                    }
                }
                ops::mixed_ce(logits, &q, &hit, target, self.lambda)
            }
            Some(c0) => {
                let mut counts = vec![0.0f32; s];
                for t in 0..s {
                    if let Some((p, c)) = self.tabla.lookup_detail(&ds.ids[..base + t + 1], v) {
                        q[t * v..(t + 1) * v].copy_from_slice(&p);
                        hit[t] = 1.0;
                        counts[t] = c as f32;
                    }
                }
                ops::mixed_ce_count(logits, &q, &hit, &counts, target, self.lambda, c0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::param;
    use crate::tensor::autograd::backward;
    use crate::tensor::Tensor;

    fn sum_of(t: &Tensor) -> f32 {
        t.data.iter().sum()
    }

    /// Gradiente numérico sobre logits, con la misma mezcla que el forward.
    fn numeric_logits(x: &mut Tensor, q: &[f32], hit: &[f32], targets: &[usize], lambda: f32) -> Vec<f32> {
        let mut out = vec![0.0; x.data.len()];
        for i in 0..x.data.len() {
            let orig = x.data[i];
            std::sync::Arc::make_mut(&mut x.data)[i] = orig + 1e-3;
            let fp = sum_of(&ops::mixed_ce(x, q, hit, targets, lambda));
            std::sync::Arc::make_mut(&mut x.data)[i] = orig - 1e-3;
            let fm = sum_of(&ops::mixed_ce(x, q, hit, targets, lambda));
            std::sync::Arc::make_mut(&mut x.data)[i] = orig;
            out[i] = (fp - fm) / 2e-3;
        }
        out
    }

    fn check(a: &[f32], b: &[f32], name: &str) {
        assert_eq!(a.len(), b.len(), "len mismatch en {name}");
        for i in 0..a.len() {
            let err = (a[i] - b[i]).abs();
            let scale = (a[i].abs() + b[i].abs()).max(1.0);
            assert!(err / scale < 1e-3, "{name}[{i}]: analitico={} numerico={} err={}", a[i], b[i], err);
        }
    }

    fn q_para(t: usize, v: usize, prob_t: f32) -> Vec<f32> {
        // Distribución de la tabla: masa `prob_t` en el target, el resto
        // repartido uniforme. Vocab 3 con el target en 1.
        let mut q = vec![0.0; v];
        q[t] = prob_t;
        let resto = (1.0 - prob_t) / (v - 1) as f32;
        for j in 0..v {
            if j != t {
                q[j] = resto;
            }
        }
        q
    }

    /// CONTROL: λ=0 tiene que ser EXACTAMENTE cross_entropy, valor y
    /// gradiente, aunque la tabla esté llena de hits. Si no, el cambio de op
    /// cambió algo por accidente y el experimento no mide la hipótesis.
    #[test]
    fn lambda_zero_es_cross_entropy() {
        let mut x = param(vec![0.9, -0.4, 0.2, 1.4, 0.3, -1.1, 0.5, 0.0, 0.8], vec![3, 3]);
        let targets = vec![1usize, 0, 2];
        let mut q = Vec::with_capacity(9);
        for _ in 0..3 {
            q.extend_from_slice(&q_para(1, 3, 0.9));
        }
        let hit = vec![1.0f32; 3];

        let mixed = ops::mixed_ce(&x, &q, &hit, &targets, 0.0);
        let ce = ops::cross_entropy(&x, &targets);
        assert_eq!(&mixed.data[0].to_bits(), &ce.data[0].to_bits(),
            "λ=0 cambió el valor: {} vs {}", mixed.data[0], ce.data[0]);

        let gm = backward(&mixed);
        let gc = backward(&ce);
        let xm = gm.get(&x.id).cloned().unwrap();
        let xc = gc.get(&x.id).cloned().unwrap();
        let iguales = xm.iter().zip(&xc).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(iguales, "λ=0 cambió el gradiente bit a bit");
    }

    /// GRADCHECK 2: λ>0 con hit de verdad — la tabla acierta fuerte en el
    /// target y el gradiente tiene que salir atenuado, no escalado de CE.
    #[test]
    fn gradcheck_mixed_con_hit() {
        let mut x = param(vec![0.9, -0.4, 0.2, 1.4, 0.3, -1.1, 0.5, 0.0, 0.8], vec![3, 3]);
        let targets = vec![1usize, 0, 2];
        let mut q = Vec::with_capacity(9);
        for (t, prob) in targets.iter().zip([0.9, 0.8, 0.7]) {
            q.extend_from_slice(&q_para(*t, 3, prob));
        }
        let hit = vec![1.0f32; 3];
        let lambda = 0.4;

        let loss = ops::mixed_ce(&x, &q, &hit, &targets, lambda);
        let grads = backward(&loss);
        let g = grads.get(&x.id).cloned().unwrap();
        let n = numeric_logits(&mut x, &q, &hit, &targets, lambda);
        check(&g, &n, "hit");
    }

    /// GRADCHECK 3: λ>0 sin hit — factor = 1 y el gradiente es el de CE.
    #[test]
    fn gradcheck_mixed_sin_hit() {
        let mut x = param(vec![0.9, -0.4, 0.2, 1.4, 0.3, -1.1, 0.5, 0.0, 0.8], vec![3, 3]);
        let targets = vec![1usize, 0, 2];
        let q = vec![0.0f32; 9];
        let hit = vec![0.0f32; 3];
        let lambda = 0.4;

        let loss = ops::mixed_ce(&x, &q, &hit, &targets, lambda);
        let grads = backward(&loss);
        let g = grads.get(&x.id).cloned().unwrap();
        let n = numeric_logits(&mut x, &q, &hit, &targets, lambda);
        check(&g, &n, "miss");
    }

    fn numeric_count(
        x: &mut Tensor, q: &[f32], hit: &[f32], counts: &[f32], targets: &[usize], lambda: f32, c0: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0; x.data.len()];
        for i in 0..x.data.len() {
            let orig = x.data[i];
            std::sync::Arc::make_mut(&mut x.data)[i] = orig + 1e-3;
            let fp = sum_of(&ops::mixed_ce_count(x, q, hit, counts, targets, lambda, c0));
            std::sync::Arc::make_mut(&mut x.data)[i] = orig - 1e-3;
            let fm = sum_of(&ops::mixed_ce_count(x, q, hit, counts, targets, lambda, c0));
            std::sync::Arc::make_mut(&mut x.data)[i] = orig;
            out[i] = (fp - fm) / 2e-3;
        }
        out
    }

    /// GRADCHECK 4: `mixed_ce_count` con counts mezclados — piso (2), medio
    /// (50) y alto (500). El gate por count tiene que dar gradiente correcto.
    #[test]
    fn gradcheck_mixed_ce_count() {
        let mut x = param(vec![0.9, -0.4, 0.2, 1.4, 0.3, -1.1, 0.5, 0.0, 0.8], vec![3, 3]);
        let targets = vec![1usize, 0, 2];
        let mut q = Vec::with_capacity(9);
        for t in targets.iter() {
            q.extend_from_slice(&q_para(*t, 3, 0.9));
        }
        let hit = vec![1.0f32; 3];
        let counts = vec![2.0f32, 50.0, 500.0];

        let loss = ops::mixed_ce_count(&x, &q, &hit, &counts, &targets, 0.4, 2.0);
        let grads = backward(&loss);
        let g = grads.get(&x.id).cloned().unwrap();
        let n = numeric_count(&mut x, &q, &hit, &counts, &targets, 0.4, 2.0);
        check(&g, &n, "count");
    }

    /// CONTROL: con el count en el piso (c = c0), el gate da λ_i = 0 y la
    /// posición es EXACTAMENTE cross_entropy, valor y gradiente bit a bit.
    #[test]
    fn count_gate_en_el_piso_es_cross_entropy() {
        let mut x = param(vec![0.9, -0.4, 0.2, 1.4, 0.3, -1.1, 0.5, 0.0, 0.8], vec![3, 3]);
        let targets = vec![1usize, 0, 2];
        let mut q = Vec::with_capacity(9);
        for t in targets.iter() {
            q.extend_from_slice(&q_para(*t, 3, 0.9));
        }
        let hit = vec![1.0f32; 3];
        let counts = vec![2.0f32, 2.0, 2.0];

        let mixed = ops::mixed_ce_count(&x, &q, &hit, &counts, &targets, 0.4, 2.0);
        let ce = ops::cross_entropy(&x, &targets);
        assert_eq!(mixed.data[0].to_bits(), ce.data[0].to_bits(),
            "gate en el piso cambió el valor: {} vs {}", mixed.data[0], ce.data[0]);

        let gm = backward(&mixed);
        let gc = backward(&ce);
        let xm = gm.get(&x.id).cloned().unwrap();
        let xc = gc.get(&x.id).cloned().unwrap();
        let iguales = xm.iter().zip(&xc).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(iguales, "gate en el piso cambió el gradiente bit a bit");
    }

    /// EL MECANISMO: un modelo FRÍO (p_tgt chico, la tabla acierta fuerte).
    /// Con λ constante, el hecho raro (count 2) se atenúa IGUAL que el
    /// boilerplate (count 50) — factor 0.25, el problema del barrido. Con el
    /// gate, el hecho raro conserva TODO el gradiente (CE exacto) y sólo el
    /// boilerplate queda atenuado.
    #[test]
    fn el_gate_preserva_el_gradiente_del_hecho_raro() {
        let x = param(vec![0.0, 1.386], vec![1, 2]); // softmax: p[0]=0.2, p[1]=0.8
        let targets = vec![0usize];
        let q = vec![0.9f32, 0.1]; // la tabla acierta el target
        let hit = vec![1.0f32];
        let lambda = 0.4;

        // Gradiente de CE puro en el logit del target: p_tgt − 1 = −0.8.
        let g_ce = backward(&ops::cross_entropy(&x, &targets));
        let gce = g_ce.get(&x.id).cloned().unwrap();

        // λ constante: el hecho raro se atenúa como el boilerplate.
        let g_cons = backward(&ops::mixed_ce(&x, &q, &hit, &targets, lambda));
        let gcons = g_cons.get(&x.id).cloned().unwrap();

        // Gate: c=2 (piso) → CE exacto; c=50 → atenuado.
        let g_rare = backward(&ops::mixed_ce_count(&x, &q, &hit, &vec![2.0f32], &targets, lambda, 2.0));
        let grare = g_rare.get(&x.id).cloned().unwrap();
        let g_bol = backward(&ops::mixed_ce_count(&x, &q, &hit, &vec![50.0f32], &targets, lambda, 2.0));
        let gbol = g_bol.get(&x.id).cloned().unwrap();

        // El hecho raro con gate == CE bit a bit (gradiente completo).
        assert_eq!(grare[0].to_bits(), gce[0].to_bits(),
            "gate en el piso no devolvió el gradiente de CE: {} vs {}", grare[0], gce[0]);

        // El boilerplate con gate queda atenuado (menos gradiente que CE).
        assert!(gbol[0].abs() < gce[0].abs(),
            "el boilerplate tendría que estar atenuado: {} vs {}", gbol[0], gce[0]);

        // Y el hecho raro recibe MÁS gradiente que el boilerplate, que es el
        // punto del gate. Con λ constante eran indistinguibles.
        assert!(grare[0].abs() > gbol[0].abs() + 1e-4,
            "el gate no separó hecho raro de boilerplate: {} vs {}", grare[0], gbol[0]);
    }
}
