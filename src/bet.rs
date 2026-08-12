//! 8. QUE APUESTE — el primer número que lo mata.
//!
//! > ¿La confianza que el modelo **declara** rastrea lo que realmente acierta,
//! > sobre texto que **no vio**?
//!
//! Se corre el modelo sobre la porción de validación (el mismo corte contiguo
//! que usó `train`) y, para cada predicción de token, se registra:
//!
//! * `conf`    = p(token que elegiría) — qué tan seguro *dijo* estar
//! * `correct` = si ese token era el real
//!
//! Se agrupan por deciles de confianza declarada y se mira si el acierto sube
//! con ella. **Si la correlación no aparece, el principio organizador muere
//! acá** y nos ahorramos los otros seis.
//!
//! El modo de falla a vigilar —que aprenda a apostar siempre bajo— se ve en el
//! margen: si el margen (p1 − p2) no separa a los que aciertan de los que no,
//! la confianza es cobardía uniforme y tampoco sirve para apostar.
//!
//! Nada se ajusta contra este número: es una medición. La validación se toca
//! una sola vez.

use crate::data::TextDataset;
use crate::model::EvaModel;

#[derive(Clone, Copy, Default)]
pub struct Bin {
    pub n: usize,
    pub conf: f64,
    pub hits: usize,
}

/// Agregado de la calibración sobre una porción de texto.
///
/// Los bins se arman sobre la confianza **declarada** (p del token que el
/// modelo elegiría). Un bin vacío no cuenta en nada de lo de abajo.
pub struct Calibration {
    n_bins: usize,
    bins: Vec<Bin>,
    n: usize,
    hits: usize,
    m_sum: f64,
    m2_sum: f64,
    m_hits: f64,
    pt_sum: f64,
    pt_hits: f64,
}

impl Calibration {
    pub fn new(n_bins: usize) -> Self {
        Calibration {
            n_bins: n_bins.max(1),
            bins: vec![Bin::default(); n_bins.max(1)],
            n: 0,
            hits: 0,
            m_sum: 0.0,
            m2_sum: 0.0,
            m_hits: 0.0,
            pt_sum: 0.0,
            pt_hits: 0.0,
        }
    }

    pub fn add(&mut self, conf: f32, margin: f32, correct: bool, p_target: f32) {
        let mut b = (conf * self.n_bins as f32) as usize;
        if b >= self.n_bins {
            b = self.n_bins - 1;
        }
        self.bins[b].n += 1;
        self.bins[b].conf += conf as f64;
        self.n += 1;
        self.m_sum += margin as f64;
        self.m2_sum += margin as f64 * margin as f64;
        self.pt_sum += p_target as f64;
        if correct {
            self.hits += 1;
            self.bins[b].hits += 1;
            self.m_hits += margin as f64;
            self.pt_hits += p_target as f64;
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    pub fn bins(&self) -> &[Bin] {
        &self.bins
    }

    pub fn bin_conf(&self, b: usize) -> f32 {
        (self.bins[b].conf / self.bins[b].n.max(1) as f64) as f32
    }

    pub fn bin_accuracy(&self, b: usize) -> f32 {
        self.bins[b].hits as f32 / self.bins[b].n.max(1) as f32
    }

    pub fn bin_n(&self, b: usize) -> usize {
        self.bins[b].n
    }

    pub fn accuracy(&self) -> f32 {
        self.hits as f32 / self.n.max(1) as f32
    }

    /// Expected Calibration Error ponderado por bin: |acierto − conf media|.
    pub fn ece(&self) -> f32 {
        let mut ece = 0.0f64;
        for b in &self.bins {
            if b.n == 0 {
                continue;
            }
            let acc = b.hits as f64 / b.n as f64;
            let mc = b.conf / b.n as f64;
            ece += (acc - mc).abs() * b.n as f64;
        }
        (ece / self.n.max(1) as f64) as f32
    }

    /// Correlación entre confianza media y acierto a través de los bins.
    ///
    /// Es el número que decide: si la confianza rastrea el acierto, tiene que
    /// ser claramente positivo.
    pub fn bin_corr(&self) -> f32 {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for b in &self.bins {
            if b.n == 0 {
                continue;
            }
            xs.push(b.conf / b.n as f64);
            ys.push(b.hits as f64 / b.n as f64);
        }
        pearson(&xs, &ys)
    }

    /// Acierto del bin más seguro contra el del menos seguro.
    pub fn top_vs_bottom(&self) -> Option<(f32, f32)> {
        let mut top: Option<(f64, f64)> = None;
        let mut bottom: Option<(f64, f64)> = None;
        for b in &self.bins {
            if b.n == 0 {
                continue;
            }
            let conf = b.conf / b.n as f64;
            let acc = b.hits as f64 / b.n as f64;
            top = Some(match top {
                Some((c, a)) if c >= conf => (c, a),
                _ => (conf, acc),
            });
            bottom = Some(match bottom {
                Some((c, a)) if c <= conf => (c, a),
                _ => (conf, acc),
            });
        }
        match (top, bottom) {
            (Some((_, t)), Some((_, b))) => Some((t as f32, b as f32)),
            _ => None,
        }
    }

    /// Correlación puntual-biserial entre el margen (p1 − p2) y el acierto.
    ///
    /// Si el margen no separa a los que aciertan de los que no, la confianza es
    /// cobardía uniforme: el modo de falla que la apuesta tiene que vigilar.
    pub fn margin_r(&self) -> f32 {
        let n = self.n.max(1) as f64;
        let p = self.hits as f64 / n;
        let m1 = self.m_hits / self.hits.max(1) as f64;
        let m0 = (self.m_sum - self.m_hits) / (self.n - self.hits).max(1) as f64;
        let s = ((self.m2_sum - self.m_sum * self.m_sum / n) / (self.n.max(2) - 1) as f64).sqrt();
        if s <= 0.0 {
            return 0.0;
        }
        ((m1 - m0) / s * (p * (1.0 - p)).sqrt()) as f32
    }

    /// Probabilidad que le dio a la verdad: cuando acierta vs cuando no.
    ///
    /// Si cuando acierta le da p baja, no está "seguro" ni en los aciertos.
    pub fn pt_calibration(&self) -> (f32, f32) {
        let yes = self.pt_hits / self.hits.max(1) as f64;
        let no = (self.pt_sum - self.pt_hits) / (self.n - self.hits).max(1) as f64;
        (yes as f32, no as f32)
    }
}

pub fn pearson(xs: &[f64], ys: &[f64]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let (sx, sy) = (xs.iter().sum::<f64>(), ys.iter().sum::<f64>());
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let syy: f64 = ys.iter().map(|y| y * y).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let den = ((n * sxx - sx * sx) * (n * syy - sy * sy)).sqrt();
    if den <= 0.0 {
        0.0
    } else {
        ((n * sxy - sx * sy) / den) as f32
    }
}

/// Clasifica una fila de logits contra el byte real.
///
/// Devuelve `(conf, margen, correcto, p_target)`:
/// * `conf` = p del token que el modelo elegiría (su apuesta)
/// * `margen` = p1 − p2, qué tan clara está la elección
/// * `correcto` = si el token elegido era el real
/// * `p_target` = p que le dio a la verdad
pub fn classify_row(row: &[f32], target: usize) -> (f32, f32, bool, f32) {
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    let mut p_target = 0.0f32;
    for (i, &v) in row.iter().enumerate() {
        let e = (v - mx).exp();
        sum += e;
        if i == target {
            p_target = e;
        }
        if e > top1 {
            top2 = top1;
            top1 = e;
            argmax = i;
        } else if e > top2 {
            top2 = e;
        }
    }
    (top1 / sum, (top1 - top2) / sum, argmax == target, p_target / sum)
}

/// Una observación por posición de texto.
pub struct TokenObs {
    pub conf: f32,
    pub margin: f32,
    pub correct: bool,
    pub p_target: f32,
    /// Largo (norma L2) del estado oculto pre-norma en esa posición.
    pub hidden_norm: f32,
}

/// Mide la calibración del modelo sobre las ventanas `ds[from..to]`.
pub fn measure(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, n_bins: usize) -> Calibration {
    let mut cal = Calibration::new(n_bins);
    for o in scan(model, ds, from, to) {
        cal.add(o.conf, o.margin, o.correct, o.p_target);
    }
    cal
}

/// Una pasada sobre `ds[from..to]`, una observación por posición.
///
/// Es la pasada única: la calibración por token y el análisis por tramo se
/// agregan sobre lo que esto devuelve, para que no sean dos implementaciones
/// del mismo softmax.
pub fn scan(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) -> Vec<TokenObs> {
    let mut out = Vec::new();
    let vocab = model.cfg.vocab;
    let dim = model.cfg.dim;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, hidden) = model.forward_hidden(&input);
        for t in 0..input.len() {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let (conf, margin, correct, pt) = classify_row(row, target[t]);
            let h = &hidden.data[t * dim..(t + 1) * dim];
            let norm = h.iter().map(|v| (v * v) as f64).sum::<f64>().sqrt() as f32;
            out.push(TokenObs { conf, margin, correct, p_target: pt, hidden_norm: norm });
        }
    }
    out
}

/// Resultado del análisis por tramo para un largo de tramo.
pub struct SpanAnalysis {
    pub span_len: usize,
    pub n: usize,
    pub bien_global: f32,
    /// Correlación de cada predictor con `bien`, sobre los tramos.
    pub r_media: f32,
    pub r_min: f32,
    /// geomean de p[verdad]: la "probabilidad del tramo" real, pero usa la
    /// respuesta — es el TECHO (oráculo), no la vara.
    pub r_geomean: f32,
    /// geomean de p[argmax]: la confianza del modelo en lo que él mismo diría,
    /// disponible al generar — esta es la VARA usable.
    pub r_geomean_argmax: f32,
    pub r_magnitud: f32,
}

/// Correlación entre los tramos.
fn span_r(xs: &[f64], ys: &[f64]) -> f32 {
    pearson(xs, ys)
}

/// Parten las observaciones en tramos de `span_len` dentro de cada ventana de
/// `seq` y mide si el acierto del tramo (`bien`) se predice con:
///
/// * `media` — promedio de p[argmax] (la agregación más simple, usable)
/// * `min` — el eslabón más débil del tramo
/// * `geomean` — media geométrica de p[verdad]: la "probabilidad del tramo"
///   real que le da el modelo a lo que escribió — TECHO, usa la respuesta
/// * `geomean_argmax` — media geométrica de p[argmax]: la confianza en lo que
///   el modelo diría, usable al generar — VARA honesta
/// * `magnitud` — largo del vector oculto AL ARRANCAR el tramo (la señal del
///   4, disponible antes de que el tramo exista)
///
/// Si ninguno rastrea `bien` sobre texto que no vio, una cabeza de stake
/// entrenada tampoco lo va a hacer: la apuesta se cierra acá.
pub fn span_analysis(obs: &[TokenObs], seq: usize, span_len: usize) -> SpanAnalysis {
    let mut xs_media = Vec::new();
    let mut xs_min = Vec::new();
    let mut xs_geomean = Vec::new();
    let mut xs_geomean_argmax = Vec::new();
    let mut xs_mag = Vec::new();
    let mut ys = Vec::new();
    let mut n = 0usize;
    let mut hits = 0usize;

    // Los tramos no cruzan ventanas: cada ventana tiene sus posiciones y el
    // contexto no se mezcla entre una y la siguiente.
    let mut i = 0;
    while i < obs.len() {
        let hasta = (i / seq + 1) * seq;
        let mut t = i;
        while t + span_len <= hasta {
            let mut media = 0.0f64;
            let mut min = f64::INFINITY;
            let mut loggeo = 0.0f64;
            let mut loggeo_argmax = 0.0f64;
            let mut ok = 0usize;
            for o in &obs[t..t + span_len] {
                media += o.conf as f64;
                min = min.min(o.conf as f64);
                loggeo += (o.p_target as f64).max(1e-9).ln();
                loggeo_argmax += (o.conf as f64).max(1e-9).ln();
                ok += o.correct as usize;
            }
            let m = media / span_len as f64;
            let geo = (loggeo / span_len as f64).exp();
            let geo_argmax = (loggeo_argmax / span_len as f64).exp();
            let mag = obs[t].hidden_norm as f64;
            let bien = ok as f64 / span_len as f64;
            xs_media.push(m);
            xs_min.push(min);
            xs_geomean.push(geo);
            xs_geomean_argmax.push(geo_argmax);
            xs_mag.push(mag);
            ys.push(bien);
            n += 1;
            hits += ok;
            t += span_len;
        }
        i = hasta;
    }

    SpanAnalysis {
        span_len,
        n,
        bien_global: hits as f32 / (n * span_len).max(1) as f32,
        r_media: span_r(&xs_media, &ys),
        r_min: span_r(&xs_min, &ys),
        r_geomean: span_r(&xs_geomean, &ys),
        r_geomean_argmax: span_r(&xs_geomean_argmax, &ys),
        r_magnitud: span_r(&xs_mag, &ys),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_obs(conf: f32, p_target: f32, correct: bool, hidden_norm: f32) -> TokenObs {
        TokenObs { conf, margin: 0.0, correct, p_target, hidden_norm }
    }

    #[test]
    fn classify_row_conocido() {
        // logits [0, 2, 1] → e = [1, e², e]; top1 es el índice 1.
        let (conf, margin, correct, pt) = classify_row(&[0.0, 2.0, 1.0], 0);
        let s = 1.0 + (2.0f32).exp() + 1.0f32.exp();
        assert!((conf - (2.0f32).exp() / s).abs() < 1e-5, "conf {conf}");
        assert!((margin - ((2.0f32).exp() - 1.0f32.exp()) / s).abs() < 1e-5, "margin {margin}");
        assert!(!correct, "el argmax es el índice 1, no el 0");
        assert!((pt - 1.0 / s).abs() < 1e-5, "p_target {pt}");
    }

    #[test]
    fn calibrado_perfecto_da_ece_cero() {
        // En cada bin, acierto == conf media. ECE tiene que dar 0.
        let mut c = Calibration::new(2);
        for _ in 0..3 {
            c.add(0.25, 0.0, false, 0.2);
        }
        c.add(0.25, 0.0, true, 0.3);
        for _ in 0..3 {
            c.add(0.75, 0.0, true, 0.8);
        }
        c.add(0.75, 0.0, false, 0.7);
        assert!(c.ece() < 1e-6, "ece {}", c.ece());
    }

    #[test]
    fn confianza_que_rastrea_da_correlacion_positiva() {
        let mut c = Calibration::new(4);
        // Cuatro bins colineales con acierto = conf + 0.05 (n=10 por bin):
        // bin0: 1/10, bin1: 4/10, bin2: 7/10, bin3: 9/10.
        let plan = [(0.05, 1), (0.35, 4), (0.65, 7), (0.95, 9)];
        for (conf, hits) in plan {
            for _ in 0..10 - hits {
                c.add(conf, 0.0, false, conf);
            }
            for _ in 0..hits {
                c.add(conf, 0.0, true, conf);
            }
        }
        assert!(c.bin_corr() > 0.99, "r = {}", c.bin_corr());
        let (top, bottom) = c.top_vs_bottom().unwrap();
        assert!(top > bottom, "top {top} debería superar a bottom {bottom}");
    }

    #[test]
    fn confianza_que_miente_da_correlacion_negativa() {
        let mut c = Calibration::new(4);
        // Colineal al revés: acierto = 1 − conf (n=20 por bin).
        let plan = [(0.05, 19), (0.35, 13), (0.65, 7), (0.95, 1)];
        for (conf, hits) in plan {
            for _ in 0..20 - hits {
                c.add(conf, 0.0, false, conf);
            }
            for _ in 0..hits {
                c.add(conf, 0.0, true, conf);
            }
        }
        assert!(c.bin_corr() < -0.999, "r = {}", c.bin_corr());
    }

    #[test]
    fn el_margen_separa_a_quienes_aciertan() {
        // Los que aciertan tienen margen alto; los que no, margen bajo.
        let mut c = Calibration::new(2);
        for _ in 0..20 {
            c.add(0.8, 0.6, true, 0.8);
            c.add(0.4, 0.05, false, 0.3);
        }
        assert!(c.margin_r() > 0.9, "r = {}", c.margin_r());
        let (yes, no) = c.pt_calibration();
        assert!(yes > no, "p_verdad cuando acierta {yes} debería superar a cuando no {no}");
    }

    #[test]
    fn confianza_uno_va_al_ultimo_bin() {
        let mut c = Calibration::new(3);
        c.add(1.0, 0.0, true, 1.0);
        assert_eq!(c.bins[2].n, 1);
        assert_eq!(c.bins[2].hits, 1);
    }

    #[test]
    fn bins_vacios_no_cuentan() {
        let mut c = Calibration::new(10);
        for _ in 0..10 {
            c.add(0.95, 0.0, true, 0.9);
        }
        assert_eq!(c.n(), 10);
        assert_eq!(c.accuracy(), 1.0);
        assert!(c.ece() >= 0.0);
        let (top, bottom) = c.top_vs_bottom().unwrap();
        assert_eq!(top, 1.0);
        assert_eq!(bottom, 1.0);
    }

    #[test]
    fn top_vs_bottom_usa_la_media_no_el_conteo() {
        // Muchos con conf baja, uno solo con conf alta: la comparación tiene
        // que ir por la MEDIA de confianza, no por la suma del bin.
        let mut c = Calibration::new(10);
        for _ in 0..100 {
            c.add(0.1, 0.0, false, 0.1); // acc 0, suma 10
        }
        c.add(0.9, 0.0, true, 0.9); // acc 1, suma 0.9
        let (top, bottom) = c.top_vs_bottom().unwrap();
        assert_eq!(top, 1.0, "el bin más seguro es el de conf 0.9");
        assert_eq!(bottom, 0.0);
    }

    #[test]
    fn tramos_no_cruzan_ventanas_y_los_predictores_rastrean() {
        // seq 6 → dos ventanas de 6 posiciones; span_len 3 → 4 tramos, y
        // ninguno puede cruzar de una ventana a la otra.
        let plan = [(0.0, 0.01), (1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 2.0 / 3.0), (1.0, 0.99)];
        let mut obs = Vec::new();
        for &(bien, pt) in &plan {
            for k in 0..3 {
                let correct = (k as f32) < bien * 3.0;
                obs.push(mk_obs(pt, pt, correct, bien + 0.5));
            }
        }
        let a = span_analysis(&obs, 6, 3);
        assert_eq!(a.n, 4);
        assert!((a.bien_global - 0.5).abs() < 0.01, "bien global {}", a.bien_global);
        assert!(a.r_geomean > 0.99, "r_geomean {}", a.r_geomean);
        assert!(a.r_geomean_argmax > 0.99, "r_geomean_argmax {}", a.r_geomean_argmax);
        assert!(a.r_magnitud > 0.99, "r_magnitud {}", a.r_magnitud);
        assert!(a.r_media > 0.99, "r_media {}", a.r_media);
    }

    #[test]
    fn geomean_con_p_cero_sigue_siendo_finito() {
        // p[verdad] = 0 en un tramo: la media geométrica tiene que quedar
        // finita (piso), no explotar a infinito.
        let mut obs = Vec::new();
        for _ in 0..2 {
            obs.push(mk_obs(0.5, 0.0, true, 1.0));
            obs.push(mk_obs(0.5, 0.5, true, 1.0));
        }
        let a = span_analysis(&obs, 4, 2);
        assert!(a.r_geomean.is_finite(), "r_geomean {}", a.r_geomean);
        assert!(a.r_geomean_argmax.is_finite(), "r_geomean_argmax {}", a.r_geomean_argmax);
    }
}
