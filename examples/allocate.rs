//! How much does it REALLY cost to allocate the gradient map on every step?
//!
//! `backward` creates a new `Vec` for every tensor that receives a
//! gradient and throws it away when it's done: 40.7 MB per step. The
//! hypothesis is that reusing those buffers saves time. But `vec![0.0; n]`
//! might be asking the system for pages that are ALREADY ZEROED, in which
//! case allocating comes out almost free and reusing saves nothing.
//!
//! This settles it before building anything: it replicates exactly the
//! sizes the real model asks for and times them.
use eva_llm_v0::model::{Arch, EvaConfig, EvaModel};

fn main() {
    let cfg = EvaConfig {
        vocab: 256, dim: 256, ffn_dim: 512, blocks: 16,
        conv_kernel: 5, eps: 1e-5, seq_len: 256, arch: Arch::Clock,
    };
    let m = EvaModel::new(cfg);
    let sizes: Vec<usize> = m.parameters().iter().map(|p| p.data.len()).collect();
    let total: usize = sizes.iter().sum();
    println!("  {} buffers, {:.1} MB total", sizes.len(), total as f64 * 4.0 / 1_048_576.0);

    // Allocate and drop, like backward does on every step.
    let mut best = f64::INFINITY;
    for _ in 0..50 {
        let t0 = std::time::Instant::now();
        let map: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0.0f32; n]).collect();
        std::hint::black_box(&map);
        drop(map);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms < best { best = ms; }
    }
    println!("  allocate + drop everything:   {best:8.3} ms   (best of 50)");

    // Reuse: the buffers already exist, just zero them out.
    let mut reuse: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0.0f32; n]).collect();
    let mut best_r = f64::INFINITY;
    for _ in 0..50 {
        let t0 = std::time::Instant::now();
        for v in reuse.iter_mut() {
            v.fill(0.0);
        }
        std::hint::black_box(&reuse);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms < best_r { best_r = ms; }
    }
    println!("  reuse (just zeroing):          {best_r:8.3} ms   (best of 50)");
    println!("\n  difference per step: {:.3} ms", best - best_r);
    println!("  the full step measures ~350 ms, so this is {:.2}%",
        100.0 * (best - best_r) / 350.0);
}
