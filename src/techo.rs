//! TECNO RETROSPECTIVO — el primer número del cómputo por influencia.
//!
//! ESTO EXISTE PARA DECIDIR SI EL CÓMPUTO CONDICIONAL TIENE ESPACIO EN EVA:
//!
//! > ¿Cuánto se puede ahorrar, como máximo, saltándose bloques que no cambian
//! > la respuesta?
//!
//! Se corre el modelo completo sobre texto NO VISTO y después se repite con un
//! bloque afuera por vez (la corriente residual pasa por el bloque como si no
//! existiera). Para cada variante se mide:
//!
//! * bits/byte — la calidad;
//! * cambio de predicción — cuántas posiciones cambian de argmax;
//! * tiempo de pared — el costo físico, medido (la regla de la casa);
//! * pesos del bloque — el tráfico que se deja de leer.
//!
//! El techo es lo que habría ahorrado una regla ideal que ya conociera el
//! efecto de cada bloque: no es lo que un centinela lograría, sino cuánto se
//! puede ganar como máximo. Si quitar bloques casi no perjudica la calidad,
//! hay espacio real para un mecanismo que los salte; si quitar cualquiera
//! destruye la calidad, la línea muere acá sin escribir el planificador.
//!
//! Nada se ajusta contra este número: es una medición sobre validación, con el
//! mismo corte contiguo que usó `train`.

use std::time::Instant;

use crate::data::TextDataset;
use crate::model::EvaModel;
use crate::nn::Module;

/// `p[target]` y el argmax de una fila de logits, con softmax estable.
fn softmax_p(logits: &[f32], tgt: usize) -> (f32, usize) {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut p_tgt = 0.0f32;
    let mut argmax = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - mx).exp();
        sum += e;
        if i == tgt {
            p_tgt = e;
        }
        if e > best {
            best = e;
            argmax = i;
        }
    }
    (p_tgt / sum, argmax)
}

/// Una variante con el bloque `idx` omitido.
pub struct BlockRes {
    pub idx: usize,
    /// bits/byte de la variante sobre validación.
    pub bits_per_byte: f32,
    /// Δ bits/byte contra el modelo completo.
    pub delta_bits: f32,
    /// Δ relativo (0.01 = 1%): la métrica que decide.
    pub rel_delta: f32,
    /// Fracción de posiciones donde cambió el argmax contra el completo.
    pub pred_change: f32,
    /// Tiempo de pared total de la variante, medido.
    pub time_total_s: f64,
    /// Lo que el bloque cuesta de tiempo = completo − variante.
    pub block_time_s: f64,
    /// Bytes de parámetros del bloque (lo que se deja de leer).
    pub weight_bytes: usize,
}

pub struct TecReport {
    pub n_pos: usize,
    pub n_val: usize,
    pub full_bits_per_byte: f32,
    pub full_time_s: f64,
    pub total_weight_bytes: usize,
    pub blocks: Vec<BlockRes>,
}

/// Resultado de omitir un conjunto explícito en UNA pasada real.
pub struct ComboRes {
    pub skips: Vec<usize>,
    pub full_bits_per_byte: f32,
    pub bits_per_byte: f32,
    pub delta_bits: f32,
    pub rel_delta: f32,
    pub pred_change: f32,
    pub full_time_s: f64,
    pub time_total_s: f64,
    pub weight_bytes: usize,
}

fn pasada(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, skips: &[usize]) -> (f64, Vec<usize>, f64) {
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let mut nats = 0.0f64;
    let mut argmax = Vec::with_capacity((to - from) * seq);
    let t0 = Instant::now();
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_skips(&input, skips);
        for t in 0..seq {
            let (p, am) = softmax_p(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
            nats -= (p as f64).max(1e-12).ln();
            argmax.push(am);
        }
    }
    (nats, argmax, t0.elapsed().as_secs_f64())
}

/// Mide una combinación real, no la suma de ablations individuales.
pub fn medir_combinacion(
    model: &EvaModel, ds: &TextDataset, from: usize, to: usize, skips: &[usize],
) -> Result<ComboRes, String> {
    if skips.is_empty() || skips.iter().any(|&i| i >= model.blocks.len()) {
        return Err("la combinación tiene que omitir al menos un bloque existente".into());
    }
    if skips.windows(2).any(|w| w[0] >= w[1]) {
        return Err("los bloques omitidos tienen que venir ordenados y sin repetidos".into());
    }
    let n_pos = (to - from) * model.cfg.seq_len;
    let (nats_full, full_argmax, full_time_s) = pasada(model, ds, from, to, &[]);
    let (nats, argmax, time_total_s) = pasada(model, ds, from, to, skips);
    let full_bits = (nats_full / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;
    let bits = (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;
    let changes = argmax.iter().zip(&full_argmax).filter(|(a, b)| a != b).count();
    let weight_bytes = skips.iter().map(|&i| {
        model.blocks[i].parameters().iter().map(|t| t.numel()).sum::<usize>() * 4
    }).sum();
    Ok(ComboRes {
        skips: skips.to_vec(), full_bits_per_byte: full_bits, bits_per_byte: bits,
        delta_bits: bits - full_bits, rel_delta: (bits - full_bits) / full_bits.max(1e-9),
        pred_change: changes as f32 / n_pos.max(1) as f32,
        full_time_s, time_total_s, weight_bytes,
    })
}

/// Mide el techo sobre las ventanas `ds[from..to]`.
///
/// Una pasada por variante: primero el modelo completo (la línea base), y
/// después una por cada bloque omitido. El tiempo de cada variante se mide de
/// pared; lo que el bloque cuesta es la diferencia contra el completo, sobre
/// el MISMO texto.
pub fn medir(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) -> Result<TecReport, String> {
    if model.blocks.is_empty() {
        return Err("el modelo no tiene bloques: no hay techo que medir".into());
    }
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let n_val = to - from;
    let n_pos = n_val * seq;

    // Pasada completa: la línea base.
    let mut nats_full = 0.0f64;
    let mut full_argmax = Vec::with_capacity(n_pos);
    let t0 = Instant::now();
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_skip(&input, None);
        for t in 0..seq {
            let (p, am) = softmax_p(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
            nats_full -= (p as f64).max(1e-12).ln();
            full_argmax.push(am);
        }
    }
    let full_time_s = t0.elapsed().as_secs_f64();
    let full_bits = (nats_full / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;

    // Una pasada por bloque omitido.
    let mut blocks = Vec::with_capacity(model.blocks.len());
    for idx in 0..model.blocks.len() {
        let mut nats = 0.0f64;
        let mut changes = 0usize;
        let t0 = Instant::now();
        for wi in from..to {
            let rel = wi - from;
            let (input, target) = ds.window(wi);
            let (logits, _) = model.forward_skip(&input, Some(idx));
            for t in 0..seq {
                let (p, am) = softmax_p(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
                nats -= (p as f64).max(1e-12).ln();
                changes += (am != full_argmax[rel * seq + t]) as usize;
            }
        }
        let time_s = t0.elapsed().as_secs_f64();
        let bpb = (nats / n_pos.max(1) as f64 / std::f64::consts::LN_2) as f32;
        let weight_bytes: usize =
            model.blocks[idx].parameters().iter().map(|t: &&crate::tensor::Tensor| t.numel()).sum::<usize>() * 4;
        blocks.push(BlockRes {
            idx,
            bits_per_byte: bpb,
            delta_bits: bpb - full_bits,
            rel_delta: (bpb - full_bits) / full_bits.max(1e-9),
            pred_change: changes as f32 / n_pos.max(1) as f32,
            time_total_s: time_s,
            block_time_s: full_time_s - time_s,
            weight_bytes,
        });
    }

    Ok(TecReport {
        n_pos,
        n_val,
        full_bits_per_byte: full_bits,
        full_time_s,
        total_weight_bytes: model.param_count() * 4,
        blocks,
    })
}

/// Cuánto habría ahorrado una regla ideal que conociera el efecto real y
/// saltara todo bloque con pérdida relativa de bits/byte `≤ umbral`.
///
/// Devuelve (bloques saltados, % de tiempo ahorrado, % de pesos no leídos,
/// bits/byte proyectados si se saltaran todos los elegidos). La pérdida
/// proyectada es la SUMA de las pérdidas individuales: correr la combinación
/// junta podría dar más o menos, y eso no se asume — es el techo, no la
/// implementación.
pub fn regla_ideal(report: &TecReport, umbral: f32) -> (usize, f32, f32, f32) {
    let mut saltar = 0usize;
    let mut t = 0.0f64;
    let mut w = 0usize;
    let mut perdida = 0.0f32;
    for b in &report.blocks {
        if b.rel_delta <= umbral {
            saltar += 1;
            t += b.block_time_s;
            w += b.weight_bytes;
            perdida += b.delta_bits;
        }
    }
    (
        saltar,
        (100.0 * t / report.full_time_s.max(1e-12)) as f32,
        (100.0 * w as f64 / report.total_weight_bytes.max(1) as f64) as f32,
        report.full_bits_per_byte + perdida,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Arch;
    use crate::model::EvaConfig;

    fn mini_model() -> EvaModel {
        EvaModel::new(EvaConfig {
            vocab: 256,
            dim: 8,
            ffn_dim: 16,
            blocks: 2,
            conv_kernel: 3,
            eps: 1e-5,
            seq_len: 6,
            arch: Arch::Clock,
        })
    }

    fn mini_ds() -> TextDataset {
        TextDataset::from_str("hola mundo, esta es una prueba de texto para el techo. ", 6)
    }

    #[test]
    fn skip_none_es_exactamente_forward_hidden() {
        let m = mini_model();
        let ids = vec![1usize, 2, 3, 4, 5, 6];
        let (a, ha) = m.forward_hidden(&ids);
        let (b, hb) = m.forward_skip(&ids, None);
        assert_eq!(a.data, b.data, "logits tienen que ser idénticos");
        assert_eq!(ha.data, hb.data, "hidden tiene que ser idéntico");
    }

    #[test]
    fn skip_un_bloque_cambia_solo_ese_bloque() {
        let m = mini_model();
        let ids = vec![1usize, 2, 3, 4, 5, 6];
        // El skip 0 y el skip 1 tienen que diferir del completo (si el bloque
        // no hace nada, no hay techo que medir), y pueden diferir entre sí.
        let (full, _) = m.forward_skip(&ids, None);
        let (s0, _) = m.forward_skip(&ids, Some(0));
        let (s1, _) = m.forward_skip(&ids, Some(1));
        assert_ne!(full.data, s0.data, "quitar el bloque 0 no puede no cambiar nada");
        assert_ne!(full.data, s1.data, "quitar el bloque 1 no puede no cambiar nada");
    }

    #[test]
    fn softmax_matchea_el_calculo_a_mano() {
        let logits = [0.0f32, 2.0, 1.0];
        let s = 1.0 + 2.0f32.exp() + 1.0f32.exp();
        let (p, am) = softmax_p(&logits, 0);
        assert!((p - 1.0 / s).abs() < 1e-5, "p[verdad] {p}");
        assert_eq!(am, 1, "el argmax es el índice 1");
    }

    #[test]
    fn medir_devuelve_un_resultado_por_bloque() {
        let m = mini_model();
        let ds = mini_ds();
        let n = ds.num_windows();
        assert!(n >= 2, "el dataset de prueba tiene que partirse");
        let rep = medir(&m, &ds, 0, n).unwrap();
        assert_eq!(rep.blocks.len(), m.blocks.len());
        assert_eq!(rep.n_pos, n * m.cfg.seq_len);
        assert!(rep.full_bits_per_byte > 0.0, "bits/byte del completo");
        for b in &rep.blocks {
            assert!(b.weight_bytes > 0);
            assert!(b.pred_change >= 0.0 && b.pred_change <= 1.0, "pred_change {}", b.pred_change);
        }
    }

    #[test]
    fn combinacion_rechaza_indices_invalidos_y_mide_un_conjunto() {
        let m = mini_model();
        let ds = mini_ds();
        let n = ds.num_windows();
        assert!(medir_combinacion(&m, &ds, 0, n, &[0, 0]).is_err(), "no acepta repetidos");
        assert!(medir_combinacion(&m, &ds, 0, n, &[2]).is_err(), "no acepta fuera de rango");
        let r = medir_combinacion(&m, &ds, 0, n, &[0, 1]).expect("mide combinación válida");
        assert_eq!(r.skips, vec![0, 1]);
        assert!(r.bits_per_byte.is_finite());
        assert!(r.time_total_s >= 0.0);
        assert!(r.weight_bytes > 0);
    }

    #[test]
    fn regla_ideal_ordena_bien() {
        let m = mini_model();
        let ds = mini_ds();
        let n = ds.num_windows();
        let rep = medir(&m, &ds, 0, n).unwrap();
        // Con umbral 0 nadie califica; con umbral enorme todos.
        let (cero, _, _, _) = regla_ideal(&rep, 0.0);
        let (todos, _, _, _) = regla_ideal(&rep, f32::MAX);
        assert_eq!(cero, 0);
        assert_eq!(todos, rep.blocks.len());
    }
}
