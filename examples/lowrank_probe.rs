//! Task #58: did the low-rank write's parameters actually MOVE, or do they
//! just exist? `po` starts at exactly zero, so its norm is the cleanest
//! possible read of "this path is alive": if it is still 0 after training,
//! the whole low-rank branch contributed nothing and any result from the run
//! is about the element-wise path alone.
//!
//! USAGE: cargo run --release --example lowrank_probe -- <checkpoint>
//! (EVA_WRITE_LOWRANK must be set to the SAME r used to train, or the
//! checkpoint loads without these parameters and the probe says so.)

use eva_llm_v0::save::load_model;

fn stats(v: &[f32]) -> (f32, f32) {
    let n = v.len().max(1) as f32;
    let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let amax = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    (l2 / n.sqrt(), amax)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: lowrank_probe <checkpoint>");
    let model = load_model(&path).expect("could not load checkpoint");
    let named = model.named_parameters();
    let mut found = 0;
    for (name, t) in &named {
        let tail = name.rsplit('.').next().unwrap_or("");
        if !matches!(tail, "pk" | "pv" | "pq" | "po" | "log_clock_m") {
            continue;
        }
        found += 1;
        let (rms, amax) = stats(&t.data);
        if tail == "log_clock_m" {
            // Report the decay itself, not its logit: sigmoid(z), the number
            // that says how long a binding survives in that slot.
            let a: Vec<f32> = t.data.iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
            let (lo, hi) = a.iter().fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
            println!("{:<28} n={:<6} alpha_m range [{:.4}, {:.4}]", name, t.data.len(), lo, hi);
        } else {
            println!("{:<28} n={:<6} rms={:.6}  |max|={:.6}{}", name, t.data.len(), rms, amax,
                if tail == "po" && amax == 0.0 { "   <-- STILL ZERO: the low-rank path never learned" } else { "" });
        }
    }
    if found == 0 {
        println!("no low-rank parameters in this checkpoint -- was EVA_WRITE_LOWRANK set, both to train and now?");
    }
}
