//! Inferencia de a un token, llevando estado.
//!
//! POR QUÉ EXISTE, con el desperdicio medido: `generate` rehacía la pasada
//! completa sobre toda la ventana **para cada token**, y de las 64 filas que
//! calculaba usaba UNA y tiraba 63. Con `seq=64` eso es hasta 64x de trabajo
//! al pedo por token. Encima construía el grafo de autograd entero --con sus
//! clones de entradas por operación-- para descartarlo enseguida.
//!
//! LO QUE LO HACE POSIBLE es la propiedad que distingue a esta arquitectura:
//! el estado de `ClockMem` es de **tamaño fijo**, D números, no importa cuán
//! largo sea el contexto. Un transformer no puede hacer esto: necesita un
//! caché de claves y valores que **crece con cada token**. Acá se implementan
//! los dos, justamente para que esa diferencia se pueda medir en vez de
//! afirmarse.
//!
//! EL PELIGRO, y es el motivo de que el test se escribiera antes que el
//! código: esto es una **segunda implementación de la misma matemática**. Los
//! dos caminos se desincronizan en silencio -- el de entrenamiento queda bien
//! y el rápido calcula otra cosa, sin que nada falle. Por eso
//! `tests::stream_matches_batch` exige que paso a paso dé lo mismo que la
//! pasada completa. Si algún día ese test se pone en amarillo, el que miente
//! es este archivo.

use crate::model::block::Mixer;
use crate::model::EvaModel;
use crate::nn::{Linear, RMSNorm};

/// Estado de un bloque entre un token y el siguiente.
struct BlockState {
    /// Las `k-1` entradas anteriores de la conv, en orden, aplanadas.
    ///
    /// Arranca en ceros y eso **es** el relleno del principio: la conv suma
    /// `w * x[src]` sólo cuando `src >= 0`, y multiplicar por cero da lo
    /// mismo que no sumar. No hace falta un caso especial.
    conv: Vec<f32>,
    mixer: MixerState,
}

enum MixerState {
    /// D números. No crece nunca, por más largo que sea el contexto.
    Clock(Vec<f32>),
    /// Crece con cada token: esta es la diferencia, hecha código.
    Attn { keys: Vec<f32>, vals: Vec<f32> },
}

pub struct Streamer<'a> {
    model: &'a EvaModel,
    blocks: Vec<BlockState>,
    /// Posición absoluta. Sólo la usa la tabla de posiciones de la atención.
    t: usize,
    /// `sigmoid(log_clock)` por bloque, calculado una vez.
    alphas: Vec<Vec<f32>>,
}

impl<'a> Streamer<'a> {
    pub fn new(model: &'a EvaModel) -> Self {
        let d = model.cfg.dim;
        let blocks = model
            .blocks
            .iter()
            .map(|b| BlockState {
                conv: vec![0.0; (b.conv.k - 1) * d],
                mixer: match &b.mixer {
                    Mixer::Clock(_) => MixerState::Clock(vec![0.0; d]),
                    Mixer::Attn(_) => MixerState::Attn { keys: Vec::new(), vals: Vec::new() },
                },
            })
            .collect();
        let alphas = model
            .blocks
            .iter()
            .map(|b| match &b.mixer {
                Mixer::Clock(m) => m.log_clock.data.iter().map(|&v| sigmoid(v)).collect(),
                Mixer::Attn(_) => Vec::new(),
            })
            .collect();
        Streamer { model, blocks, t: 0, alphas }
    }

    /// Consume un token y devuelve los logits del siguiente.
    pub fn next(&mut self, id: usize) -> Vec<f32> {
        let x = self.advance(id);
        let x = rms_norm(&x, &self.model.norm_out);
        matvec(&x, &self.model.head_w.data, self.model.cfg.dim, self.model.cfg.vocab)
    }

    /// Consume un token SIN calcular la proyección de salida.
    ///
    /// Para cuando la estructura ya decidió qué viene: no hay nada que
    /// preguntarle al modelo, pero el estado igual tiene que avanzar o se
    /// desincroniza del texto. Esto es lo que hace que restringir AHORRE en
    /// vez de sólo evitar el error.
    pub fn consume(&mut self, id: usize) {
        self.advance(id);
    }

    /// Logits de sólo algunas columnas, cuando la estructura dejó pocas
    /// opciones. Devuelve `(token, logit)` en el mismo orden que `cols`.
    pub fn next_among(&mut self, id: usize, cols: &[usize]) -> Vec<f32> {
        let x = self.advance(id);
        let x = rms_norm(&x, &self.model.norm_out);
        let d = self.model.cfg.dim;
        let w = &self.model.head_w.data;
        let v = self.model.cfg.vocab;
        cols.iter()
            .map(|&c| (0..d).map(|i| x[i] * w[i * v + c]).sum())
            .collect()
    }

    /// Todo el modelo menos la cabeza: deja el estado listo y devuelve la
    /// representación de la posición.
    fn advance(&mut self, id: usize) -> Vec<f32> {
        let cfg = &self.model.cfg;
        let d = cfg.dim;

        let mut x = self.model.embed.table.data[id * d..(id + 1) * d].to_vec();
        if let Some(pos) = &self.model.pos {
            // La tabla es de largo fijo; más allá se repite la última, que es
            // lo que hacía el recorte de ventana.
            let p = self.t.min(cfg.seq_len - 1);
            for (xi, pi) in x.iter_mut().zip(&pos.data[p * d..(p + 1) * d]) {
                *xi += pi;
            }
        }

        for i in 0..self.model.blocks.len() {
            x = self.block_step(i, &x);
        }
        self.t += 1;
        x
    }

    fn block_step(&mut self, i: usize, x: &[f32]) -> Vec<f32> {
        let b = &self.model.blocks[i];
        let d = self.model.cfg.dim;

        // h = x + conv(norm0(x))
        let n0 = rms_norm(x, &b.norm0);
        let c = self.conv_step(i, &n0);
        let mut h: Vec<f32> = x.iter().zip(&c).map(|(a, b)| a + b).collect();

        // h = h + mixer(norm1(h))
        let n1 = rms_norm(&h, &b.norm1);
        let m = self.mixer_step(i, &n1);
        for (hi, mi) in h.iter_mut().zip(&m) {
            *hi += mi;
        }

        // out = h + glu(norm2(h))
        let n2 = rms_norm(&h, &b.norm2);
        let a = matvec(&n2, &b.glu.w1.data, d, b.glu.w1.shape[1]);
        let g = matvec(&n2, &b.glu.w2.data, d, b.glu.w2.shape[1]);
        let gated: Vec<f32> = a.iter().zip(&g).map(|(x, y)| silu(*x) * y).collect();
        let f = matvec(&gated, &b.glu.w3.data, gated.len(), d);
        for (hi, fi) in h.iter_mut().zip(&f) {
            *hi += fi;
        }
        h
    }

    fn conv_step(&mut self, i: usize, x: &[f32]) -> Vec<f32> {
        let b = &self.model.blocks[i];
        let (d, k) = (self.model.cfg.dim, b.conv.k);
        let hist = &self.blocks[i].conv;
        let mut out = vec![0.0; d];
        for c in 0..d {
            let mut acc = b.conv.b.data[c];
            // `u` recorre el kernel; los primeros k-1 pesos van contra la
            // historia y el último contra el token actual, igual que el `src`
            // de la versión por lotes.
            for u in 0..k - 1 {
                acc += b.conv.w.data[c * k + u] * hist[u * d + c];
            }
            acc += b.conv.w.data[c * k + (k - 1)] * x[c];
            out[c] = acc;
        }
        // Corre la ventana: se va el más viejo, entra el actual.
        let hist = &mut self.blocks[i].conv;
        if k > 1 {
            hist.copy_within(d.., 0);
            hist[(k - 2) * d..].copy_from_slice(x);
        }
        out
    }

    fn mixer_step(&mut self, i: usize, x: &[f32]) -> Vec<f32> {
        let d = self.model.cfg.dim;
        match (&self.model.blocks[i].mixer, &mut self.blocks[i].mixer) {
            (Mixer::Clock(m), MixerState::Clock(s)) => {
                let q = linear(x, &m.wq, d);
                let kk = linear(x, &m.wk, d);
                let v = linear(x, &m.wv, d);
                let g = linear(x, &m.wg, d);
                let beta = m.beta.data[0];
                let alpha = &self.alphas[i];
                let mut out = vec![0.0; d];
                for c in 0..d {
                    s[c] = alpha[c] * s[c] + beta * kk[c] * v[c];
                    out[c] = q[c] * s[c] * sigmoid(g[c]);
                }
                out
            }
            (Mixer::Attn(m), MixerState::Attn { keys, vals }) => {
                let q = linear(x, &m.wq, d);
                keys.extend_from_slice(&linear(x, &m.wk, d));
                vals.extend_from_slice(&linear(x, &m.wv, d));
                // El caché se recorta a la ventana con la que se entrenó, que
                // es lo que hacía el recorte de la versión vieja. Sin tope, la
                // atención mira más lejos de lo que vio nunca en
                // entrenamiento, Y la memoria crece sin límite. ClockMem no
                // necesita este recorte: su olvido está en el mecanismo.
                let tope = self.model.cfg.seq_len;
                if keys.len() / d > tope {
                    keys.drain(..d);
                    vals.drain(..d);
                }
                let n = keys.len() / d;
                let scale = 1.0 / (d as f32).sqrt();

                let mut scores: Vec<f32> = (0..n)
                    .map(|j| {
                        let kj = &keys[j * d..(j + 1) * d];
                        q.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;

                let mut ctx = vec![0.0; d];
                for j in 0..n {
                    let a = scores[j] * inv;
                    for (c, o) in ctx.iter_mut().enumerate() {
                        *o += a * vals[j * d + c];
                    }
                }
                linear(&ctx, &m.wo, d)
            }
            _ => unreachable!("el estado no corresponde al mezclador"),
        }
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn rms_norm(x: &[f32], n: &RMSNorm) -> Vec<f32> {
    let d = x.len();
    let sq: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sq / d as f32 + n.eps).sqrt();
    x.iter().zip(n.w.data.iter()).map(|(v, w)| v * inv * w).collect()
}

fn linear(x: &[f32], l: &Linear, out_d: usize) -> Vec<f32> {
    let mut y = matvec(x, &l.w.data, x.len(), out_d);
    for (yi, bi) in y.iter_mut().zip(l.b.data.iter()) {
        *yi += bi;
    }
    y
}

/// Una fila por una matriz. Usa el mismo `math::matmul` que el entrenamiento
/// --con su AVX2 y su reparto-- en vez de escribir un tercer producto.
fn matvec(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    crate::math::matmul(x, w, 1, k, n, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, EvaConfig, EvaModel};

    fn modelo(arch: Arch) -> EvaModel {
        EvaModel::new(EvaConfig {
            vocab: 32,
            dim: 24,
            ffn_dim: 40,
            blocks: 3,
            conv_kernel: 4,
            eps: 1e-5,
            seq_len: 12,
            arch,
        })
    }

    /// EL TEST QUE JUSTIFICA EL ARCHIVO.
    ///
    /// `stream.rs` es una segunda implementación de la misma matemática, y dos
    /// implementaciones se desincronizan en silencio: la de entrenamiento
    /// queda bien, la rápida calcula otra cosa, y nada falla. Acá se exige que
    /// la última fila de la pasada completa sea la misma que la que devuelve
    /// el paso a paso, para cada posición.
    ///
    /// No se pide igualdad exacta de bits: la pasada por lotes suma en otro
    /// orden que la fila sola, y en punto flotante eso difiere en el último
    /// dígito. Se pide 1e-4 relativo, que es varios órdenes por debajo de
    /// cualquier diferencia que signifique un error de lógica.
    fn stream_iguala_a_lote(arch: Arch) {
        let m = modelo(arch);
        let ids: Vec<usize> = vec![7, 3, 19, 0, 11, 4, 28, 15];

        let mut st = Streamer::new(&m);
        for (i, &id) in ids.iter().enumerate() {
            let paso = st.next(id);

            // La pasada completa sobre el prefijo: su última fila predice lo
            // mismo que el paso a paso después de consumir ese token.
            let lote = m.forward(&ids[..=i]);
            let v = lote.shape[1];
            let ultima = &lote.data[(lote.shape[0] - 1) * v..];

            assert_eq!(ultima.len(), paso.len());
            for (j, (a, b)) in ultima.iter().zip(&paso).enumerate() {
                let err = (a - b).abs() / (a.abs() + b.abs()).max(1.0);
                assert!(
                    err < 1e-4,
                    "posición {i}, logit {j}: lote={a} stream={b} (err {err:.2e})"
                );
            }
        }
    }

    #[test]
    fn stream_matches_batch_clock() {
        stream_iguala_a_lote(Arch::Clock);
    }

    #[test]
    fn stream_matches_batch_attn() {
        stream_iguala_a_lote(Arch::Attn);
    }

    /// La propiedad que hace que todo esto valga la pena: el estado de
    /// ClockMem NO crece con el contexto. Si algún día crece, se perdió la
    /// única ventaja estructural que tenemos sobre un transformer.
    #[test]
    fn clock_state_does_not_grow_with_context() {
        let m = modelo(Arch::Clock);
        let mut st = Streamer::new(&m);
        let tamano = |s: &Streamer| -> usize {
            s.blocks
                .iter()
                .map(|b| {
                    b.conv.len()
                        + match &b.mixer {
                            MixerState::Clock(v) => v.len(),
                            MixerState::Attn { keys, vals } => keys.len() + vals.len(),
                        }
                })
                .sum()
        };
        let inicial = tamano(&st);
        for i in 0..200 {
            st.next(i % 32);
        }
        assert_eq!(inicial, tamano(&st), "el estado creció con el contexto");
    }

    /// Y el contraste, que es el punto de haber implementado las dos: el de la
    /// atención SÍ crece, linealmente. Esto no es un defecto de la
    /// implementación, es la arquitectura.
    #[test]
    fn attention_cache_grows_and_that_is_the_difference() {
        let m = modelo(Arch::Attn);
        let mut st = Streamer::new(&m);
        let cache = |s: &Streamer| -> usize {
            s.blocks
                .iter()
                .map(|b| match &b.mixer {
                    MixerState::Attn { keys, vals } => keys.len() + vals.len(),
                    MixerState::Clock(v) => v.len(),
                })
                .sum()
        };
        assert_eq!(0, cache(&st));
        for i in 0..50 {
            st.next(i % 32);
        }
        // Crece hasta la ventana de entrenamiento y ahí se detiene. Aun con
        // tope, es 12 veces el estado de ClockMem para el mismo modelo -- y
        // sin tope crecería para siempre.
        assert_eq!(3 * 12 * 24 * 2, cache(&st));
        let clock = modelo(Arch::Clock);
        let mut sc = Streamer::new(&clock);
        for i in 0..50 {
            sc.next(i % 32);
        }
        let clock_state: usize = sc
            .blocks
            .iter()
            .map(|b| match &b.mixer {
                MixerState::Clock(v) => v.len(),
                MixerState::Attn { keys, vals } => keys.len() + vals.len(),
            })
            .sum();
        assert!(cache(&st) > 10 * clock_state, "la diferencia de memoria se perdió");
    }
}
