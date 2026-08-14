//! How much work gets saved when the structure has already decided.
//!
//! Generates the SAME output with and without the constraint and compares.
//! The template mimics the shape real, useful output takes in an actual
//! system: a fixed wrapper with holes where the model actually decides.
use eva_llm_v0::constrain::{Free, Skeleton};
use eva_llm_v0::gen::generate_shaped;
use eva_llm_v0::rng::Rng;
use eva_llm_v0::save::load_model;

fn main() {
    let path = std::env::args().nth(1).expect("usage: forma <weights>");
    let model = load_model(&path).expect("load the model");
    let prompt: Vec<usize> = "The ".bytes().map(|b| b as usize).collect();
    // The template's length, not 300: if hundreds of bytes get generated
    // past the template, the rest goes unconstrained and the measured
    // proportion says nothing about what happens when the output IS the
    // structured thing.
    let n = 100;

    let run = |shape: &mut dyn eva_llm_v0::constrain::Constraint, label: &str| {
        // WARM UP BEFORE MEASURING. Without this the first condition pays
        // for cold memory and the second one collects the advantage: the
        // first measurement gave 18% savings where the mechanism only
        // explains 0.2%, and that mismatch was the error, not the result.
        let mut rng = Rng::new(7);
        let _ = generate_shaped(&model, &prompt, n, 0.8, 40, &mut rng, shape);
        // BEST OF 20. With a single run this gave +18% and then -75% for
        // the same mechanism: the machine's variance is bigger than the
        // effect.
        let mut best = f64::INFINITY;
        let mut c = eva_llm_v0::gen::Counts::default();
        for _ in 0..20 {
            let mut rng = Rng::new(7);
            let t0 = std::time::Instant::now();
            let (_, cc) = generate_shaped(&model, &prompt, n, 0.8, 40, &mut rng, shape);
            let t = t0.elapsed().as_secs_f64() * 1e3;
            if t < best { best = t; c = cc; }
        }
        let ms = best;
        println!(
            "  {label:<12} {ms:7.1} ms | forced {:3} partial {:3} free {:3} \
             | columns {} of {} ({:.0}% less)",
            c.forced, c.partial, c.free, c.columns, c.columns_unrestricted,
            100.0 * (1.0 - c.columns as f64 / c.columns_unrestricted as f64),
        );
        ms
    };

    println!("generating {n} bytes with the same model:");
    let free = run(&mut Free, "free");

    // A structured-answer-style template: fixed wrapper, two holes. Small
    // holes, like the real ones: what the model chooses is little and the
    // wrapper is a lot.
    let mut template = Skeleton::new(
        &[r#"{"say":""#, r#"","citing":["#, "]}"],
        &[(70, b'"'), (3, b']')],
    );
    let with = run(&mut template, "with template");

    println!("\n  time difference: {:.1}%", 100.0 * (1.0 - with / free));

    // How much the head ACTUALLY weighs in this model, measured and not
    // assumed. Without this, any extrapolation to large vocabularies is
    // hot air.
    let d = model.cfg.dim;
    let v = model.cfg.vocab;
    let x = vec![0.5f32; d];
    let mut out = vec![0.0f32; v];
    let mut head = f64::INFINITY;
    for _ in 0..2000 {
        let t0 = std::time::Instant::now();
        eva_llm_v0::math::matmul(&x, &model.head_w.data, 1, d, v, &mut out);
        let t = t0.elapsed().as_secs_f64() * 1e3;
        if t < head { head = t; }
    }
    let per_token = free / n as f64;
    println!("\n  head alone: {:.4} ms | full step: {:.4} ms | the head is {:.1}%",
        head, per_token, 100.0 * head / per_token);
    println!("  with a 32000 vocabulary the head would cost {:.1}x more and would dominate",
        32000.0 / v as f64);
}
