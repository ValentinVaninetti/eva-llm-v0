//! Micro-benchmark of the windowed ClockMem read at real scale.
//!
//! Loads a checkpoint with EVA_READ_WIN=K set (so wread gets created),
//! then times forward+backward on the first few training windows.
//!
//! USAGE: EVA_READ_WIN=257 cargo run --release --example bench_win -- <weights> <data> <iters>

use std::time::Instant;

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::EvaConfig;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::autograd::backward;
use eva_llm_v0::tensor::ops;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (weights, data) = (args[1].clone(), args[2].clone());
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);

    let mcfg = EvaConfig {
        dim: 512,
        ffn_dim: 1024,
        blocks: 5,
        vocab: 4096,
        seq_len: 512,
        ..Default::default()
    };
    let model = load_model(&weights).unwrap();
    let ds = TextDataset::from_file(&data, mcfg.seq_len).unwrap();

    println!("model: {} params (expect +wread if EVA_READ_WIN set)", model.param_count());

    // Warmup.
    let (input, target) = ds.window(0);
    let logits = model.forward(&input);
    let loss = ops::cross_entropy(&logits, &target);
    backward(&loss);

    let mut fwd_us = 0u128;
    let mut bwd_us = 0u128;
    for i in 0..iters {
        let (input, target) = ds.window(i);
        let t0 = Instant::now();
        let logits = model.forward(&input);
        let loss = ops::cross_entropy(&logits, &target);
        fwd_us += t0.elapsed().as_micros();
        let t1 = Instant::now();
        backward(&loss);
        bwd_us += t1.elapsed().as_micros();
    }
    println!(
        "forward+loss  avg {:.1} ms   backward avg {:.1} ms   total avg {:.1} ms  (over {} iters)",
        fwd_us as f64 / iters as f64 / 1000.0,
        bwd_us as f64 / iters as f64 / 1000.0,
        (fwd_us + bwd_us) as f64 / iters as f64 / 1000.0,
        iters
    );
}
