//! Write-magnitude probe for the error-gated write (EVA_WRITE_ERROR).
//!
//! Measures, over a slice of the corpus, whether the model's own internal
//! error signal already separates positions where the token is NEW from
//! positions where it REPEATS something already in the window, and what the
//! gate would do to the write magnitude in each class.
//!
//! The gate lives inside the normal forward (`ops::clockmem_gated_from`).
//! With `EVA_WRITE_ERROR=1 EVA_WRITE_ERROR_TRACE=1` that forward records,
//! per ClockMem block and per position, the pair (e_t, r_t). This command
//! sets those variables itself so the measurement is deterministic and
//! always measures the gate ON (it reads `EVA_WRITE_ERROR_P` if you set it,
//! default 1.0).
//!
//! The gate-OFF baseline needs no forward pass: with the plain recurrence
//! the write magnitude is the block's `beta` at every position, identical
//! for both classes by construction. The separation the gate could add --
//! "si es novedad, escribe más" -- is exactly the `r` column of this probe.
//!
//! "Novelty" is defined per window: position t is NEW if input[t] has not
//! appeared at any earlier position of the same window, REPEATED otherwise.
//! A position is classified by its token, not by the model's output.

use crate::data::TextDataset;
use crate::model::block::Mixer;
use crate::model::EvaModel;
use crate::tensor::ops;

#[derive(Default)]
struct Stats {
    count: usize,
    sum_e: f64,
    sum_r: f64,
    sum_w: f64,
    sat: usize,
    zero: usize,
}

impl Stats {
    fn add(&mut self, e: f32, r: f32, beta: f32) {
        self.count += 1;
        self.sum_e += e as f64;
        self.sum_r += r as f64;
        self.sum_w += (beta * r) as f64;
        if r >= 0.999 {
            self.sat += 1;
        }
        if r < 1e-4 {
            self.zero += 1;
        }
    }
}

fn print_row(name: &str, st: &Stats) {
    let n = st.count.max(1) as f64;
    println!(
        "  {name:<9} {:>7}  {:>9.4}  {:>9.4}  {:>17.4}  {:>7.1}  {:>8.1}",
        st.count,
        st.sum_e / n,
        st.sum_r / n,
        st.sum_w / n,
        100.0 * st.sat as f64 / n,
        100.0 * st.zero as f64 / n,
    );
}

/// Runs the measurement and prints the report.
///
/// `from`/`to` are window indices into `ds` (half-open). Each window runs
/// independently (fresh state, like training and eval), so `e_ref` is per
/// window and "novelty" is per window.
pub fn run(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) {
    std::env::set_var("EVA_WRITE_ERROR", "1");
    std::env::set_var("EVA_WRITE_ERROR_TRACE", "1");
    let p = std::env::var("EVA_WRITE_ERROR_P")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);

    let mut betas = Vec::new();
    for b in &model.blocks {
        if let Mixer::Clock(m) = &b.mixer {
            betas.push(m.beta.data[0]);
        }
    }
    if betas.is_empty() {
        println!("wprobe: no ClockMem blocks in this model; nothing to measure");
        return;
    }
    let n_clock = betas.len();
    let mean_beta: f32 = betas.iter().sum::<f32>() / n_clock as f32;
    let s = model.cfg.seq_len;
    let vocab = model.cfg.vocab;

    let mut new = Stats::default();
    let mut rep = Stats::default();
    let mut all = Stats::default();

    for wi in from..to {
        let (input, _target) = ds.window(wi);
        let classes = novelty(&input, vocab);
        ops::write_trace_reset();
        let _ = model.forward(&input);
        let tr = ops::write_trace_take();
        assert_eq!(
            tr.len(),
            n_clock * classes.len() * 2,
            "wprobe: expected {} trace values ({} blocks x {} positions x 2), got {}; \
             is another EVA_READ_* toggle replacing the plain write?",
            n_clock * classes.len() * 2,
            n_clock,
            classes.len(),
            tr.len()
        );
        let mut off = 0;
        for bi in 0..n_clock {
            let beta = betas[bi];
            for &is_new in classes.iter() {
                let e = tr[off];
                let r = tr[off + 1];
                off += 2;
                all.add(e, r, beta);
                if is_new {
                    new.add(e, r, beta);
                } else {
                    rep.add(e, r, beta);
                }
            }
        }
    }

    println!();
    println!(
        "wprobe: {} windows x {} positions, {} ClockMem blocks, gate ON (EVA_WRITE_ERROR=1, p={p})",
        to - from,
        s,
        n_clock
    );
    println!("novelty = first occurrence of the token in its window; e_ref = per-window EMA (eta 0.1)");
    println!();
    println!("  class     positions  mean e      mean r    mean write beta*r    sat r%   zero r%");
    print_row("new", &new);
    print_row("repeated", &rep);
    print_row("all", &all);
    println!();
    println!("  gate OFF baseline: mean beta over blocks = {mean_beta:.4} at every position;");
    println!("  both classes write the same magnitude by construction -- the r column above");
    println!("  is the separation the gate adds.");
}

fn novelty(ids: &[usize], vocab: usize) -> Vec<bool> {
    let mut seen = vec![false; vocab];
    let mut out = Vec::with_capacity(ids.len());
    for &t in ids {
        out.push(!seen[t]);
        seen[t] = true;
    }
    out
}
