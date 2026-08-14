//! 8. THAT IT BETS -- the first number that kills it.
//!
//! > Does the confidence the model **declares** track what it actually gets
//! > right, on text it **did not see**?
//!
//! The model is run over the validation slice (the same contiguous cut
//! `train` used) and, for each token prediction, we record:
//!
//! * `conf`    = p(token it would pick) -- how sure it *said* it was
//! * `correct` = whether that token was the real one
//!
//! These are grouped by declared-confidence deciles and we check whether
//! accuracy rises with it. **If the correlation doesn't show up, the
//! organizing principle dies right here** and we save ourselves the other
//! six.
//!
//! The failure mode to watch for -- learning to always bet low -- shows up
//! in the margin: if the margin (p1 - p2) doesn't separate the ones it gets
//! right from the ones it doesn't, the confidence is uniform cowardice and
//! isn't usable for betting either.
//!
//! Nothing is tuned against this number: it's a measurement. Validation is
//! touched exactly once.

use crate::data::TextDataset;
use crate::model::EvaModel;

#[derive(Clone, Copy, Default)]
pub struct Bin {
    pub n: usize,
    pub conf: f64,
    pub hits: usize,
}

/// Calibration aggregate over a slice of text.
///
/// Bins are built on the **declared** confidence (p of the token the model
/// would pick). An empty bin doesn't count toward anything below.
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

    /// Expected Calibration Error weighted per bin: |accuracy - mean conf|.
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

    /// Correlation between mean confidence and accuracy across bins.
    ///
    /// This is the number that decides: if confidence tracks accuracy, it
    /// has to be clearly positive.
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

    /// Accuracy of the most confident bin against the least confident one.
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

    /// Point-biserial correlation between margin (p1 - p2) and accuracy.
    ///
    /// If the margin doesn't separate the ones it gets right from the ones
    /// it doesn't, the confidence is uniform cowardice: the failure mode the
    /// bet has to watch for.
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

    /// Probability it gave to the truth: when correct vs when not.
    ///
    /// If even when correct it assigns low p, it isn't "sure" even on the
    /// hits.
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

/// Classifies one row of logits against the real byte.
///
/// Returns `(conf, margin, correct, p_target)`:
/// * `conf` = p of the token the model would pick (its bet)
/// * `margin` = p1 - p2, how clear-cut the choice is
/// * `correct` = whether the chosen token was the real one
/// * `p_target` = p it gave to the truth
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

/// One observation per text position.
pub struct TokenObs {
    pub conf: f32,
    pub margin: f32,
    pub correct: bool,
    pub p_target: f32,
    /// Length (L2 norm) of the pre-norm hidden state at that position.
    pub hidden_norm: f32,
}

/// Measures the model's calibration over windows `ds[from..to]`.
pub fn measure(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, n_bins: usize) -> Calibration {
    let mut cal = Calibration::new(n_bins);
    for o in scan(model, ds, from, to) {
        cal.add(o.conf, o.margin, o.correct, o.p_target);
    }
    cal
}

/// One pass over `ds[from..to]`, one observation per position.
///
/// This is the single pass: per-token calibration and per-span analysis are
/// both aggregated from what this returns, so they aren't two
/// implementations of the same softmax.
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

/// Result of the per-span analysis for one span length.
pub struct SpanAnalysis {
    pub span_len: usize,
    pub n: usize,
    pub good_global: f32,
    /// Correlation of each predictor with `good`, over the spans.
    pub r_media: f32,
    pub r_min: f32,
    /// geomean of p[truth]: the real "span probability", but it uses the
    /// answer -- it's the CEILING (oracle), not the bar.
    pub r_geomean: f32,
    /// geomean of p[argmax]: the model's confidence in what it would say
    /// itself, available at generation time -- this is the usable BAR.
    pub r_geomean_argmax: f32,
    pub r_magnitude: f32,
}

/// Correlation across spans.
fn span_r(xs: &[f64], ys: &[f64]) -> f32 {
    pearson(xs, ys)
}

/// Splits the observations into `span_len`-long spans within each `seq`
/// window and checks whether span accuracy (`good`) is predicted by:
///
/// * `mean` -- average of p[argmax] (the simplest aggregation, usable)
/// * `min` -- the weakest link of the span
/// * `geomean` -- geometric mean of p[truth]: the real "span probability"
///   the model gives to what it wrote -- CEILING, uses the answer
/// * `geomean_argmax` -- geometric mean of p[argmax]: confidence in what the
///   model would say, usable at generation time -- honest BAR
/// * `magnitude` -- length of the hidden vector AT THE START of the span
///   (signal #4, available before the span even exists)
///
/// If none of them track `good` on text the model hasn't seen, a trained
/// stake head won't either: the bet closes right here.
pub fn span_analysis(obs: &[TokenObs], seq: usize, span_len: usize) -> SpanAnalysis {
    let mut xs_media = Vec::new();
    let mut xs_min = Vec::new();
    let mut xs_geomean = Vec::new();
    let mut xs_geomean_argmax = Vec::new();
    let mut xs_mag = Vec::new();
    let mut ys = Vec::new();
    let mut n = 0usize;
    let mut hits = 0usize;

    // Spans never cross windows: each window has its own positions and
    // context doesn't mix between one and the next.
    let mut i = 0;
    while i < obs.len() {
        let end = (i / seq + 1) * seq;
        let mut t = i;
        while t + span_len <= end {
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
            let good = ok as f64 / span_len as f64;
            xs_media.push(m);
            xs_min.push(min);
            xs_geomean.push(geo);
            xs_geomean_argmax.push(geo_argmax);
            xs_mag.push(mag);
            ys.push(good);
            n += 1;
            hits += ok;
            t += span_len;
        }
        i = end;
    }

    SpanAnalysis {
        span_len,
        n,
        good_global: hits as f32 / (n * span_len).max(1) as f32,
        r_media: span_r(&xs_media, &ys),
        r_min: span_r(&xs_min, &ys),
        r_geomean: span_r(&xs_geomean, &ys),
        r_geomean_argmax: span_r(&xs_geomean_argmax, &ys),
        r_magnitude: span_r(&xs_mag, &ys),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_obs(conf: f32, p_target: f32, correct: bool, hidden_norm: f32) -> TokenObs {
        TokenObs { conf, margin: 0.0, correct, p_target, hidden_norm }
    }

    #[test]
    fn classify_row_known_case() {
        // logits [0, 2, 1] -> e = [1, e^2, e]; top1 is index 1.
        let (conf, margin, correct, pt) = classify_row(&[0.0, 2.0, 1.0], 0);
        let s = 1.0 + (2.0f32).exp() + 1.0f32.exp();
        assert!((conf - (2.0f32).exp() / s).abs() < 1e-5, "conf {conf}");
        assert!((margin - ((2.0f32).exp() - 1.0f32.exp()) / s).abs() < 1e-5, "margin {margin}");
        assert!(!correct, "the argmax is index 1, not 0");
        assert!((pt - 1.0 / s).abs() < 1e-5, "p_target {pt}");
    }

    #[test]
    fn perfect_calibration_gives_zero_ece() {
        // In each bin, accuracy == mean conf. ECE has to come out to 0.
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
    fn confidence_that_tracks_gives_positive_correlation() {
        let mut c = Calibration::new(4);
        // Four bins collinear with accuracy = conf + 0.05 (n=10 per bin):
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
        assert!(top > bottom, "top {top} should beat bottom {bottom}");
    }

    #[test]
    fn confidence_that_lies_gives_negative_correlation() {
        let mut c = Calibration::new(4);
        // Collinear the other way around: accuracy = 1 - conf (n=20 per bin).
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
    fn the_margin_separates_who_gets_it_right() {
        // The ones that get it right have high margin; the ones that don't,
        // low margin.
        let mut c = Calibration::new(2);
        for _ in 0..20 {
            c.add(0.8, 0.6, true, 0.8);
            c.add(0.4, 0.05, false, 0.3);
        }
        assert!(c.margin_r() > 0.9, "r = {}", c.margin_r());
        let (yes, no) = c.pt_calibration();
        assert!(yes > no, "p_truth when correct {yes} should beat when not {no}");
    }

    #[test]
    fn confidence_one_goes_to_the_last_bin() {
        let mut c = Calibration::new(3);
        c.add(1.0, 0.0, true, 1.0);
        assert_eq!(c.bins[2].n, 1);
        assert_eq!(c.bins[2].hits, 1);
    }

    #[test]
    fn empty_bins_do_not_count() {
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
    fn top_vs_bottom_uses_the_mean_not_the_count() {
        // Many with low conf, a single one with high conf: the comparison
        // has to go by the MEAN confidence, not the bin's sum.
        let mut c = Calibration::new(10);
        for _ in 0..100 {
            c.add(0.1, 0.0, false, 0.1); // acc 0, sum 10
        }
        c.add(0.9, 0.0, true, 0.9); // acc 1, sum 0.9
        let (top, bottom) = c.top_vs_bottom().unwrap();
        assert_eq!(top, 1.0, "the most confident bin is the one at conf 0.9");
        assert_eq!(bottom, 0.0);
    }

    #[test]
    fn spans_do_not_cross_windows_and_predictors_track() {
        // seq 6 -> two windows of 6 positions; span_len 3 -> 4 spans, and
        // none of them can cross from one window to the other.
        let plan = [(0.0, 0.01), (1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 2.0 / 3.0), (1.0, 0.99)];
        let mut obs = Vec::new();
        for &(good, pt) in &plan {
            for k in 0..3 {
                let correct = (k as f32) < good * 3.0;
                obs.push(mk_obs(pt, pt, correct, good + 0.5));
            }
        }
        let a = span_analysis(&obs, 6, 3);
        assert_eq!(a.n, 4);
        assert!((a.good_global - 0.5).abs() < 0.01, "good global {}", a.good_global);
        assert!(a.r_geomean > 0.99, "r_geomean {}", a.r_geomean);
        assert!(a.r_geomean_argmax > 0.99, "r_geomean_argmax {}", a.r_geomean_argmax);
        assert!(a.r_magnitude > 0.99, "r_magnitude {}", a.r_magnitude);
        assert!(a.r_media > 0.99, "r_media {}", a.r_media);
    }

    #[test]
    fn geomean_with_p_zero_stays_finite() {
        // p[truth] = 0 in one span: the geometric mean has to stay finite
        // (a floor), not blow up to infinity.
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
