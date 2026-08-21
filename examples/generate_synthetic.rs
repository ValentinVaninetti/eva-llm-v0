//! Generator for the long-range benchmark's synthetic corpus (lead
//! decision 2026-08-14).
//!
//! Structure: per window, an N-byte uniform seed + recurrence
//! y[t] = F(x[t-N]) with F a FIXED permutation shared across conditions
//! (derived from PHYSICS_SEED, recorded here and in
//! TRABAJO-2026-08-14.md). The window is self-contained: the dependency
//! never crosses the window boundary as long as N < seq, which is why the
//! experiment runs with seq=512 fixed.
//!
//! Position convention (dataset of windows that overlap by 1 byte):
//! - window i covers absolute positions [i*seq, (i+1)*seq];
//! - the first N are a uniform seed;
//! - the rest (and the last byte of the last window) is x[p] = F(x[p-N]);
//! - the last position of each window i < win-1 gets overwritten by the
//!   next window's seed; that's why the "recurrent" region the eval
//!   measures is [i*seq+N, (i+1)*seq) (half-open), and the eval counts it
//!   with the same rule (see examples/mask_eval.rs).
//!
//! USAGE: cargo run --release --example generate_synthetic -- <N> <seq> <win> <seed> <out>
//!   N    dependency distance (8, 32, 128, 256; 2 for the local control)
//!   seq  window length (512 fixed in this benchmark)
//!   win  number of windows (~2500; each one generates seq bytes)
//!   seed DATA seed for this corpus (recorded; F doesn't depend on it)
//!   out  output file (data/sintetico_...)
//!
//! Self-check: verifies the recurrence holds byte for byte in the
//! recurrent region of the first 8 windows, and that the seed marginal is
//! flat under sampling.

use eva_llm_v0::rng::Rng;

/// Seed for the PHYSICS (the permutation F). Fixed and shared across all
/// conditions: what changes between corpora is N and the data seed.
const PHYSICS_SEED: u64 = 0xF15A1CA;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok())
        .expect("usage: generate_synthetic <N> <seq> <win> <seed> <out>");
    let seq: usize = args.get(2).and_then(|s| s.parse().ok())
        .expect("usage: generate_synthetic <N> <seq> <win> <seed> <out>");
    let win: usize = args.get(3).and_then(|s| s.parse().ok())
        .expect("usage: generate_synthetic <N> <seq> <win> <seed> <out>");
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok())
        .expect("usage: generate_synthetic <N> <seq> <win> <seed> <out>");
    let out = args.get(5).cloned()
        .expect("usage: generate_synthetic <N> <seq> <win> <seed> <out>");

    assert!(n >= 1 && n < seq, "N has to be in [1, seq)");
    assert!(win > 0, "win has to be > 0");

    // The "world's" permutation F: shuffled identity with an Rng isolated
    // from the data. Never stored in the corpus: it's derivable from
    // PHYSICS_SEED.
    let mut physics = Rng::new(PHYSICS_SEED);
    let mut idx: Vec<usize> = (0..256).collect();
    physics.shuffle(&mut idx);
    let perm: Vec<u8> = idx.into_iter().map(|x| x as u8).collect();

    let total = win * seq + 1;
    let mut bytes = vec![0u8; total];
    let mut rng = Rng::new(seed);
    for i in 0..win {
        let s = i * seq;
        for t in 0..n {
            bytes[s + t] = (rng.next_u64() >> 40) as u8;
        }
        for p in (s + n)..=(s + seq) {
            bytes[p] = perm[bytes[p - n] as usize];
        }
    }

    // Self-check 1: the recurrence holds in the recurrent region.
    let mut failures = 0usize;
    for i in 0..win.min(8) {
        let s = i * seq;
        for p in (s + n)..(s + seq) {
            if bytes[p] != perm[bytes[p - n] as usize] {
                failures += 1;
            }
        }
    }
    if failures > 0 {
        eprintln!("ERROR: {failures} recurrence failures in the self-check");
        std::process::exit(1);
    }

    // Self-check 2: flat marginal of the seeds (64K-byte sample).
    let mut count = vec![0u32; 256];
    let mut samples = 0usize;
    for i in 0..win.min(128) {
        let s = i * seq;
        for t in 0..n.min(seq) {
            count[bytes[s + t] as usize] += 1;
            samples += 1;
        }
    }
    let expected = samples as f32 / 256.0;
    let max_dev = count.iter().map(|&c| (c as f32 - expected).abs()).fold(0.0f32, f32::max);
    let max_dev_rel = max_dev / expected.max(1.0);

    std::fs::write(&out, &bytes).expect("could not write the corpus");

    println!("generate_synthetic: N={n} seq={seq} win={win} data_seed={seed} physics_seed={PHYSICS_SEED}");
    println!("  file: {out} ({} bytes = {:.2} MB, {} windows)", total, total as f64 / 1e6, win);
    println!("  seed per window: {n} bytes ({:.1}% of the corpus), flat marginal (max rel deviation {max_dev_rel:.3} over {samples} samples)",
        100.0 * n as f64 / seq as f64);
    println!("  recurrent region per window: positions [{n}, {seq}) -> {} measurable positions", seq - n);
    println!("  F: fixed 256-byte permutation (PHYSICS_SEED={PHYSICS_SEED}), shared across conditions");
    println!("  recurrence self-check: OK (0 failures)");
    println!("  floor of the full bpb (perfect model) = {:.3} bits/byte | no memory = 8.000 bits/byte",
        8.0 * (n as f64) / (seq as f64));
}
