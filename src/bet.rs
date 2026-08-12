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

fn pearson(xs: &[f64], ys: &[f64]) -> f32 {
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

/// Mide la calibración del modelo sobre las ventanas `ds[from..to]`.
pub fn measure(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, n_bins: usize) -> Calibration {
    let mut cal = Calibration::new(n_bins);
    let vocab = model.cfg.vocab;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        for t in 0..input.len() {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let (conf, margin, correct, pt) = classify_row(row, target[t]);
            cal.add(conf, margin, correct, pt);
        }
    }
    cal
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
