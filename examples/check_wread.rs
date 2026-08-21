//! Inspect the trained `wread` of the windowed checkpoint: did the model move
//! the window weights off the delta init (w[0,c]=1, rest 0)?
//!
//! USAGE: EVA_READ_WIN=K cargo run --release --example check_wread -- <weights>

use eva_llm_v0::save::load_model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = load_model(&args[1]).unwrap();
    let mut found = 0usize;
    for (name, t) in model.named_parameters() {
        if name.ends_with(".wread") {
            found += 1;
            let (k, d) = (t.shape[0], t.shape[1]);
            let data = &t.data;
            let mut per_j_abs = vec![0.0f64; k];
            let mut max_abs = 0.0f32;
            for j in 0..k {
                let mut s = 0.0f64;
                for c in 0..d {
                    let a = data[j * d + c].abs();
                    s += a as f64;
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                per_j_abs[j] = s / d as f64;
            }
            let sum_rest: f64 = per_j_abs[1..].iter().sum();
            println!(
                "{}  [{}x{}]  mean|w[0]|={:.4}  mean|w[j>=1]|={:.4}  sum|w[j>=1]|={:.3}  max|any|={:.4}",
                name, k, d, per_j_abs[0], sum_rest / (k as f64 - 1.0), sum_rest, max_abs
            );
            // Show the 8 largest |w[j]| means over j (averaged over c).
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| per_j_abs[b].partial_cmp(&per_j_abs[a]).unwrap());
            let top: Vec<String> = order[..8.min(k)]
                .iter()
                .map(|&j| format!("j={} ({:.4})", j, per_j_abs[j]))
                .collect();
            println!("    top taps: {}", top.join(", "));
        }
    }
    if found == 0 {
        println!("NO wread params found (EVA_READ_WIN not set at load?)");
    }
}
