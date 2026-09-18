//! QUERY-SWAP: does the model actually USE the identity of the key it is
//! being queried with?
//!
//! Every model identifies the right candidate SET (92-99.9% of top-1
//! predictions are a value bound somewhere in the window) and then mostly
//! guesses within it. Top-1 cannot separate (a) it reads the query key but
//! retrieval is noisy, from (b) it IGNORES the key and emits "some value in
//! play here", which already earns partial credit. This can, with no training:
//! take a real query position `t` asking key A, and re-run the SAME window with
//! byte `t` replaced by another already-bound key B. Everything before `t` is
//! bit-identical, so the only change is which key is asked.
//!
//!   binds       -> P(val_A) collapses and P(val_B) rises
//!   ignores key -> both distributions stay nearly the same
//!
//! READ THE DIRECTIONAL NUMBER, NOT THE RATIO (learned the hard way). The
//! total-variation ratio answers "did the output MOVE", not "did it move toward
//! the right answer" -- swapping A for B replaces a byte with another of the
//! same family, which perturbs the distribution carrying no identity
//! information at all. Measured: a checkpoint showed ratio 1.45x (reproducible
//! at 3x the sample, so not noise) while `P(val_A | ask A) - P(val_A | ask B)`
//! was +0.0005. It moved 45% more than for a filler byte and pointed nowhere.
//! The signed probability difference is the one that tells binding from motion.
//!
//! The CONTROL makes those numbers readable at all: the same measurement
//! perturbing a FILLER byte just before `t`. Filler is noise the model should
//! be free to ignore, so it calibrates how much this model's output moves when
//! an irrelevant byte changes. If swapping the KEY moves it no more than
//! swapping filler, key identity is being ignored -- and that no longer depends
//! on what counts as a "big" change.
//!
//! USAGE: cargo run --release --example query_swap -- <weights> <data> [windows] [swaps_per_window]


use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::save::load_model;

const KEY_HI: usize = 32;
const FILLER_LO: usize = 64;

fn softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn total_variation(a: &[f32], b: &[f32]) -> f32 {
    0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>()
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for j in 1..row.len() {
        if row[j] > row[best] { best = j; }
    }
    best
}

fn main() {
    let weights = std::env::args().nth(1).expect("usage: query_swap <weights> <data> [windows] [swaps_per_window]");
    let data = std::env::args().nth(2).expect("usage: query_swap <weights> <data> [windows] [swaps_per_window]");
    let n_windows = std::env::args().nth(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(150);
    let per_win = std::env::args().nth(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);

    let model: EvaModel = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n_win = ds.num_windows();
    // Validation split only (contiguous tail), same convention as training.
    let n_val = (((n_win as f32) * 0.1).round() as usize).clamp(1, n_win / 2);
    let n_train = n_win - n_val;
    let to = n_win.min(n_train + n_windows);

    let mut pairs = 0usize;
    let mut orig_right = 0usize;   // top-1 == val_A before the swap
    let mut swap_follows = 0usize; // top-1 == val_B after the swap (tracked the new key)
    let mut swap_sticks = 0usize;  // top-1 still == val_A after the swap (ignored the swap)
    let mut d_pa = 0.0f64;         // P(val_A | ask A) - P(val_A | ask B)
    let mut d_pb = 0.0f64;         // P(val_B | ask B) - P(val_B | ask A)
    let mut tv_key = 0.0f64;       // total variation when swapping the KEY
    let mut tv_filler = 0.0f64;    // ... when perturbing a FILLER byte (control)
    let mut n_filler = 0usize;

    for wi in n_train..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        let v = logits.shape[1];

        // Bindings of this window, recomputed from raw bytes alone.
        let mut bound: [Option<usize>; 32] = [None; 32];
        let mut seen: [bool; 32] = [false; 32];
        let mut done = 0usize;

        for t in 0..seq.min(input.len()) {
            if done >= per_win { break; }
            let b = input[t];
            if b >= KEY_HI { continue; }
            if !seen[b] {
                seen[b] = true;
                bound[b] = Some(target[t]);
                continue;
            }
            // `t` is a genuine query for key `b`. Find another bound key
            // whose value DIFFERS, otherwise the swap tests nothing.
            let val_a = match bound[b] { Some(x) => x, None => continue };
            let other = (0..KEY_HI).find(|&k| k != b && bound[k].is_some() && bound[k] != Some(val_a));
            let (kb, val_b) = match other { Some(k) => (k, bound[k].unwrap()), None => continue };

            let row_o = softmax(&logits.data[t * v..(t + 1) * v]);

            let mut swapped = input.clone();
            swapped[t] = kb;
            let lg_s = model.forward(&swapped);
            let row_s = softmax(&lg_s.data[t * v..(t + 1) * v]);

            pairs += 1;
            if argmax(&row_o) == val_a { orig_right += 1; }
            let am_s = argmax(&row_s);
            if am_s == val_b { swap_follows += 1; }
            if am_s == val_a { swap_sticks += 1; }
            d_pa += (row_o[val_a] - row_s[val_a]) as f64;
            d_pb += (row_s[val_b] - row_o[val_b]) as f64;
            tv_key += total_variation(&row_o, &row_s) as f64;

            // CONTROL: perturb the nearest filler byte before t instead.
            if let Some(fp) = (0..t).rev().find(|&i| input[i] >= FILLER_LO) {
                let mut pert = input.clone();
                pert[fp] = FILLER_LO + ((input[fp] + 37 - FILLER_LO) % (256 - FILLER_LO));
                let lg_f = model.forward(&pert);
                let row_f = softmax(&lg_f.data[t * v..(t + 1) * v]);
                tv_filler += total_variation(&row_o, &row_f) as f64;
                n_filler += 1;
            }
            done += 1;
        }
    }

    let n = pairs.max(1) as f64;
    println!("eva query_swap: {weights}");
    println!("  {pairs} query positions swapped, over windows {n_train}..{to} (validation split)\n");
    println!("  top-1 == val_A before the swap        {:6.2}%", 100.0 * orig_right as f64 / n);
    println!("  top-1 == val_B after  the swap        {:6.2}%   <- followed the new key", 100.0 * swap_follows as f64 / n);
    println!("  top-1 still val_A after the swap      {:6.2}%   <- ignored the swap", 100.0 * swap_sticks as f64 / n);
    println!();
    println!("  --- DIRECTIONAL (this is the one that decides) ---");
    println!("  P(val_A | ask A) - P(val_A | ask B)   {:+.4}", d_pa / n);
    println!("  P(val_B | ask B) - P(val_B | ask A)   {:+.4}", d_pb / n);
    println!("    (~0 means the answer does not depend on WHICH key is being asked)");
    println!();
    let dir = ((d_pa / n) + (d_pb / n)) / 2.0;
    let verdict = if dir > 0.5 {
        "READS THE QUERY (strongly)"
    } else if dir > 0.05 {
        "reads the query, partially"
    } else {
        "DOES NOT READ THE QUERY -- answers the same no matter who is asked"
    };
    println!("  VERDICT: {verdict}   (mean directional shift {:+.4})", dir);
    println!();
    println!("  --- magnitude of motion (scale only -- see the header: this");
    println!("      number can be >1 with zero directional content) ---");
    println!("  total variation, swapping the KEY     {:.4}", tv_key / n);
    if n_filler > 0 {
        let tf = tv_filler / n_filler as f64;
        println!("  total variation, perturbing FILLER    {:.4}   <- control, n={n_filler}", tf);
        println!("  ratio KEY/FILLER                      {:.2}x", (tv_key / n) / tf.max(1e-9));
        println!("    (<=1 means changing WHICH KEY is asked moves the output no more");
        println!("     than changing an irrelevant noise byte)");
    }
}
