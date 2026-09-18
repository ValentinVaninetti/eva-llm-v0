//! Masked eval for the long-range synthetic benchmark.
//!
//! The metric that decides the thread: CONDITIONAL bpb over the recurrent
//! positions t >= N of each window. Without the mask the seed floor (N bytes
//! per window, unpredictable for any model) contaminates the N -> error curve
//! and makes N=256 look harder than it is even with perfect memory.
//!
//! Recurrent region of window i, same rule as the generator: absolute
//! [i*seq+N, (i+1)*seq), or relative targets t in [N-1, seq-2]. That region is
//! bpb 0 with perfect memory and bpb 8 without. The last target of each window
//! (t=seq-1, the first byte of the next) is excluded as a boundary.
//!
//! USAGE: cargo run --release --example mask_eval -- <weights> <data> <N>
//!   weights checkpoint trained with --seq 512 --val 0.1
//!   data    corpus from generate_synthetic
//!   N       the same dependency distance the generator used
//!   The val cut is the contiguous last 10%, as in training.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::save::load_model;

fn main() {
    let weights = std::env::args().nth(1).expect("usage: mask_eval <weights> <data> <N>");
    let data = std::env::args().nth(2).expect("usage: mask_eval <weights> <data> <N>");
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok())
        .expect("usage: mask_eval <weights> <data> <N>");

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    assert!(n >= 1 && n < seq, "N has to be in [1, seq)");
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n_win = ds.num_windows();
    assert!(n_win > 0, "the corpus is too small for seq {seq}");
    let n_val = (((n_win as f32) * 0.1).round() as usize).clamp(1, n_win / 2);
    let n_train = n_win - n_val;

    println!("eva mask_eval: {weights} | {data} | N={n} seq={seq}");
    println!("  windows: {n_train} train, {n_val} val (contiguous cut)");
    println!("  recurrent region per window: targets [{}, {}] -> {} positions", n - 1, seq - 2, seq - n);
    println!("  floor: 0 bits/byte with perfect memory, 8.000 with none\n");

    let (full, rec, seed, accv) = measure(&model, &ds, n_train, n_win, n);
    println!("=== VALIDATION (unseen text) ===");
    println!("  full bpb (everything)           {full:.4}   (contaminated by seeds)");
    println!("  RECURRENT bpb (t>=N, masked)     {rec:.4}   <-- THE NUMBER");
    println!("  recurrent top-1 acc             {accv:.3}%");
    println!("  seed bpb (t<N, real floor)       {seed:.4}   (has to be ~8: unpredictable)");

    let (full_tr, rec_tr, seed_tr, acc_tr) = measure(&model, &ds, 0, n_train, n);
    println!("\n=== TRAINING (reference, memorization) ===");
    println!("  full bpb                {full_tr:.4}");
    println!("  recurrent bpb            {rec_tr:.4}");
    println!("  recurrent top-1 acc     {acc_tr:.3}%");
    println!("  seed bpb                 {seed_tr:.4}");
}

/// Nats per segment over windows [from, to), no backward.
/// Returns (full_bpb, recurrent_bpb, seed_bpb, recurrent_acc).
fn measure(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, n: usize) -> (f32, f32, f32, f32) {
    let seq = ds.seq;
    let mut sum = [0.0f64; 3];
    let mut cnt = [0usize; 3];
    let (mut ok, mut tot) = (0usize, 0usize);
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        let (s, v) = (logits.shape[0], logits.shape[1]);
        assert_eq!(s, input.len());
        for i in 0..s {
            let row = &logits.data[i * v..(i + 1) * v];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut esum = 0.0f64;
            for &x in row {
                esum += ((x - m) as f64).exp();
            }
            let per = -((row[target[i]] - m) as f64 - esum.ln());
            sum[0] += per;
            cnt[0] += 1;
            if i >= n - 1 && i < seq - 1 {
                sum[1] += per;
                cnt[1] += 1;
                let mut best = 0usize;
                for j in 1..v {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                if best == target[i] {
                    ok += 1;
                }
                tot += 1;
            } else if i < n - 1 {
                sum[2] += per;
                cnt[2] += 1;
            }
        }
    }
    let b = |s: f64, c: usize| -> f32 { (s / c.max(1) as f64 / std::f64::consts::LN_2) as f32 };
    (b(sum[0], cnt[0]), b(sum[1], cnt[1]), b(sum[2], cnt[2]), 100.0 * ok as f32 / tot.max(1) as f32)
}
