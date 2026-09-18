//! Free number on offer: do ClockMem's `alpha`s (the
//! per-channel forgetting) show a real temporal hierarchy after training,
//! or do they all converge to something similar? Just reading an
//! already-trained checkpoint -- zero new code, zero training.
//!
//! IMPORTANT CORRECTION to how the question was originally framed:
//! dispersion is NOT something that would "emerge" from nothing.
//! `ClockMem::new` (src/model/clock.rs) seeds `log_clock` with a
//! geometric spectrum DESIGNED on purpose, from alpha~=0.01 (fast channel)
//! to alpha~=0.9999 (slow channel) -- so there's hierarchy by
//! INITIALIZATION, not by discovery. The honest, falsifiable question
//! isn't "does hierarchy show up?" (it's already there), it's: **does
//! training PRESERVE it (or sharpen it), or does it FLATTEN it toward a
//! single value, throwing away the initial design?** If it flattens, the
//! gradient is saying the hierarchy of scales wasn't contributing. If it's
//! preserved or sharpened, the model is really using it.

use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::save::load_model;

fn main() {
    let temp: f32 = std::env::var("EVA_ALPHA_TEMP")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let model = load_model(&weights).expect("could not load the checkpoint");

    println!("eva alpha_dispersion: {} blocks, dim {}\n", model.blocks.len(), model.cfg.dim);

    for (bi, block) in model.blocks.iter().enumerate() {
        let Mixer::Clock(clock) = &block.mixer else {
            println!("block {bi}: not ClockMem (different arch), skipped");
            continue;
        };
        // alpha = squash(log_clock), same as in forward() -- respects
        // EVA_ALPHA_ANTISAT because the checkpoint might have trained with
        // algebraic_sigmoid instead of sigmoid (Round 4, the hypothesis).
        let antisat = std::env::var("EVA_ALPHA_ANTISAT").is_ok();
        let alphas: Vec<f32> = clock.log_clock.data.iter().map(|&lc| {
            // BUG FIXED (2026-08-24): this probe ignored EVA_ALPHA_TEMP and
            // reported sigmoid(z) on models that use sigmoid(z/T). Every alpha
            // reading on a checkpoint trained with temperature was therefore
            // showing values the model does NOT use. Caught by a control:
            // reading the same checkpoint with and without the variable gave
            // identical output, when it had to differ.
            let lc = lc / temp;
            if antisat { 0.5 * (1.0 + lc / (1.0 + lc * lc).sqrt()) } else { 1.0 / (1.0 + (-lc).exp()) }
        }).collect();
        let mut sorted = alphas.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean = alphas.iter().sum::<f32>() / n as f32;
        let var = alphas.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32;
        let stdev = var.sqrt();

        // effective memory ~ 1/(1-alpha) tokens (approx. from the geometric series).
        let eff_mem = |a: f32| 1.0 / (1.0 - a).max(1e-6);

        println!("=== block {bi} ===");
        println!("  alpha: min {:.4} (mem~{:.0} tok)  p25 {:.4}  median {:.4}  p75 {:.4}  max {:.4} (mem~{:.0} tok)",
            sorted[0], eff_mem(sorted[0]),
            sorted[n / 4],
            sorted[n / 2],
            sorted[3 * n / 4],
            sorted[n - 1], eff_mem(sorted[n - 1]));
        println!("  mean {mean:.4}  std dev {stdev:.4}  (theoretical geometric init: alpha_min=0.01, alpha_max=0.9999)");

        // Breakdown by effective-memory band, to see whether a real
        // hierarchy (fast/medium/slow) survives or it collapses into one
        // band.
        let bands = [
            ("fast (mem<10 tok)", 0.0f32, 0.9f32),
            ("medium (10-100 tok)", 0.9, 0.99),
            ("slow (100-1000 tok)", 0.99, 0.999),
            ("very slow (>1000 tok)", 0.999, 1.0),
        ];
        print!("  breakdown by band:");
        for (name, lo, hi) in bands {
            let c = alphas.iter().filter(|&&a| a >= lo && a < hi).count();
            print!("  {name}={c}({:.0}%)", 100.0 * c as f32 / n as f32);
        }
        println!("\n");
    }

    println!("=== VERDICT ===");
    println!("  If the standard deviation is small (channels converged to something");
    println!("  similar) and almost everything falls in a single band: training");
    println!("  FLATTENED the initial spectrum -- the hierarchy of scales did not");
    println!("  hold up as useful, at this scale/corpus.");
    println!("  If the deviation is large and there's real mass across several");
    println!("  bands: the hierarchy survived (or sharpened) -- the model IS using");
    println!("  different time scales per channel, and that's where the question");
    println!("  would continue (does that accumulated history add something a");
    println!("  \"current\" state alone wouldn't give?).");
}
