//! AdamW.
//!
//! POR QUÉ ESTE ARCHIVO TIENE PARALELISMO, que no es lo que uno esperaría:
//! medido con `EVA_PROFILE=1` en un entrenamiento de 10.7M de parámetros, el
//! optimizador se llevaba **62.7 s de 310, el 20%**. No por hacer nada raro
//! sino por lo aburrido: recorría todos los parámetros DOS veces, escalar y en
//! un solo hilo, con una raíz cuadrada por elemento. Eran 6.8 ns por parámetro
//! para hacer cuatro cuentas.
//!
//! Tres cambios, en orden de cuánto dieron: repartir los elementos en bandas
//! sobre el pool (son independientes entre sí, no hay nada que sincronizar),
//! fusionar las dos pasadas en una --la mitad de tráfico de memoria-- y sacar
//! las divisiones del lazo.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::pool::Ptr;
use crate::tensor::Tensor;

/// Abajo de esto no se reparte: despertar al pool cuesta microsegundos y un
/// tensor chico se termina antes. Los parámetros que importan (las matrices de
/// dim x ffn) tienen cientos de miles de elementos y caen del lado de arriba.
const SPLIT_FROM: usize = 32_768;

/// Cuántas veces `make_mut` tuvo que COPIAR un parámetro porque el buffer
/// estaba compartido.
///
/// Tiene que ser cero. Si sube, alguien dejó el grafo vivo cuando corre el
/// optimizador y se está copiando el modelo entero en cada paso -- sin error,
/// sólo lento. Es el modo de falla que este proyecto ya sufrió demasiadas
/// veces, así que se cuenta.
static COPIAS: AtomicUsize = AtomicUsize::new(0);

pub fn copias_de_parametros() -> usize {
    COPIAS.load(Relaxed)
}

/// Muta el buffer del tensor, contando si hubo que copiar.
fn mutar(t: &mut Tensor) -> &mut Vec<f32> {
    if std::sync::Arc::strong_count(&t.data) > 1 {
        COPIAS.fetch_add(1, Relaxed);
    }
    std::sync::Arc::make_mut(&mut t.data)
}

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    t: usize,
    state: HashMap<usize, (Vec<f32>, Vec<f32>)>,
}

/// Lo que no cambia dentro de un paso. Se calcula una vez y viaja al lazo.
#[derive(Clone, Copy)]
struct Consts {
    b1: f32,
    b2: f32,
    /// `1 - b1` y `1 - b2`, ya restados.
    om1: f32,
    om2: f32,
    /// Recíprocos de las correcciones de sesgo: dividir por elemento cuando el
    /// divisor es el mismo para los diez millones es tirar ciclos.
    inv_m: f32,
    inv_v: f32,
    lr: f32,
    wd: f32,
    eps: f32,
}

impl AdamW {
    pub fn new(lr: f32, wd: f32) -> Self {
        AdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd, t: 0, state: HashMap::new() }
    }

    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &HashMap<usize, Vec<f32>>) {
        self.t += 1;
        let t = self.t as i32;
        let c = Consts {
            b1: self.beta1,
            b2: self.beta2,
            om1: 1.0 - self.beta1,
            om2: 1.0 - self.beta2,
            inv_m: 1.0 / (1.0 - self.beta1.powi(t)),
            inv_v: 1.0 / (1.0 - self.beta2.powi(t)),
            lr: self.lr,
            wd: self.wd,
            eps: self.eps,
        };

        for p in params.iter_mut() {
            let Some(g) = grads.get(&p.id) else { continue };
            debug_assert_eq!(g.len(), p.data.len(), "grad shape mismatch for param {}", p.id);
            let (m, v) = self
                .state
                .entry(p.id)
                .or_insert_with(|| (vec![0.0; p.data.len()], vec![0.0; p.data.len()]));

            let n = p.data.len();
            let pool = crate::pool::global();
            let chunks = if n < SPLIT_FROM { 1 } else { pool.workers() + 1 };
            if chunks <= 1 {
                // `make_mut` copia SÓLO si el buffer está compartido. En el
                // bucle de entrenamiento el grafo ya se liberó cuando se llega
                // acá, así que el contador está en uno y no copia nada. Si
                // alguna vez copia, es que alguien dejó el grafo vivo -- y se
                // nota como una caída de velocidad, no como un error.
                update(mutar(p), g, m, v, c);
                continue;
            }

            let per = n.div_ceil(chunks);
            let (pp, gp) = (Ptr(mutar(p).as_mut_ptr()), Ptr(g.as_ptr()));
            let (mp, vp) = (Ptr(m.as_mut_ptr()), Ptr(v.as_mut_ptr()));
            pool.run(chunks, move |i| {
                let lo = i * per;
                if lo >= n {
                    return;
                }
                let len = per.min(n - lo);
                // SAFETY: la banda [lo, lo+len) es de esta pieza y de ninguna
                // otra; las bandas no se pisan. Los cuatro buffers tienen el
                // mismo largo `n`, y `run` no vuelve hasta que todas terminaron.
                update(
                    pp.add(lo).as_mut(len),
                    gp.at(lo).as_ref(len),
                    mp.add(lo).as_mut(len),
                    vp.add(lo).as_mut(len),
                    c,
                );
            });
        }
    }
}

/// El paso de AdamW sobre una banda. Una sola pasada por los cuatro buffers.
///
/// Los cuatro `iter` en zip no son estilo: le sacan al compilador los chequeos
/// de rango y lo dejan vectorizar el lazo entero, raíz cuadrada incluida.
fn update(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], c: Consts) {
    for (((p, g), m), v) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
        *m = c.b1 * *m + c.om1 * g;
        *v = c.b2 * *v + c.om2 * g * g;
        let mhat = *m * c.inv_m;
        let vhat = *v * c.inv_v;
        *p -= c.lr * (mhat / (vhat.sqrt() + c.eps) + c.wd * *p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Repartir en bandas no puede cambiar el resultado: cada elemento depende
    /// sólo de su propio m, v y gradiente. Si esto falla, alguna banda se está
    /// pisando con otra.
    #[test]
    fn splitting_gives_the_same_numbers_as_one_thread() {
        let n = SPLIT_FROM * 3 + 17; // a propósito no múltiplo de nada
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();

        let mut serial = vec![0.5f32; n];
        let (mut m, mut v) = (vec![0.0; n], vec![0.0; n]);
        let c = Consts {
            b1: 0.9,
            b2: 0.999,
            om1: 0.1,
            om2: 0.001,
            inv_m: 1.0 / 0.1,
            inv_v: 1.0 / 0.001,
            lr: 1e-3,
            wd: 0.01,
            eps: 1e-8,
        };
        update(&mut serial, &grad, &mut m, &mut v, c);

        let mut par = vec![0.5f32; n];
        let (mut m2, mut v2) = (vec![0.0; n], vec![0.0; n]);
        let pool = crate::pool::global();
        let chunks = pool.workers() + 1;
        let per = n.div_ceil(chunks);
        let (pp, gp) = (Ptr(par.as_mut_ptr()), Ptr(grad.as_ptr()));
        let (mp, vp) = (Ptr(m2.as_mut_ptr()), Ptr(v2.as_mut_ptr()));
        pool.run(chunks, move |i| {
            let lo = i * per;
            if lo >= n {
                return;
            }
            let len = per.min(n - lo);
            update(
                pp.add(lo).as_mut(len),
                gp.at(lo).as_ref(len),
                mp.add(lo).as_mut(len),
                vp.add(lo).as_mut(len),
                c,
            );
        });

        assert_eq!(serial, par, "el reparto en bandas cambió el resultado");
    }
}
