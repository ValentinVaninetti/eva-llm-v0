//! Follow-up to the open question left after the N-sweep:
//! R2's gate settles at s~0.53 in every condition tested (block or channel,
//! N=32/128/256). Is that a genuine optimum (opening further would hurt),
//! or a gradient-magnitude artifact (opening further would help or be
//! neutral, but the gradient on `z` was too small in the training budget
//! to get there)?
//!
//! Same method already used for `alpha` last night: hand-perturb the
//! trained parameter (no retraining, no gradient), re-measure bpb. If
//! forcing the gate open (or closed) changes bpb by a lot, ~0.53 is a real
//! optimum. If forcing it open gives similar-or-better bpb than the
//! trained value, gradient starvation is the more likely story.
//!
//! USAGE: cargo run --release --example gate_perturb -- <weights> <data> <N>
//!   requires the SAME EVA_READ_* envs used to train/eval that checkpoint
//!   (EVA_READ_WIN, EVA_READ_DF_N, EVA_READ_R2=1, EVA_READ_R2_CHANNEL=1 if applicable).

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::save::load_model;
use std::sync::Arc;

fn main() {
    let weights = std::env::args().nth(1).expect("usage: gate_perturb <weights> <data> <N>");
    let data = std::env::args().nth(2).expect("usage: gate_perturb <weights> <data> <N>");
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok())
        .expect("usage: gate_perturb <weights> <data> <N>");

    let mut model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n_win = ds.num_windows();
    let n_val = (((n_win as f32) * 0.1).round() as usize).clamp(1, n_win / 2);
    let n_train = n_win - n_val;

    println!("eva gate_perturb: {weights} | {data} | N={n}\n");

    // Report the trained z, per block, before touching anything.
    for (bi, b) in model.blocks.iter().enumerate() {
        if let Mixer::Clock(c) = &b.mixer {
            if let Some(z) = &c.gate_z {
                let mean: f32 = z.data.iter().sum::<f32>() / z.data.len() as f32;
                println!("  block {bi}: trained z mean = {mean:.4} (n={})", z.data.len());
            }
        }
    }
    println!();

    let baseline = measure_recurrent_bpb(&model, &ds, n_train, n_win, n);
    println!("baseline (trained z, untouched):        bpb={:.4} acc={:.3}%", baseline.0, baseline.1);

    // Forced z values: very closed (near floor), the trained region, wide
    // open, and beyond-open (in case sigmoid saturation itself matters).
    for &z_forced in &[-3.0f32, -1.0, 0.0, 1.0, 3.0, 6.0] {
        set_all_gate_z(&mut model, z_forced);
        let (bpb, acc) = measure_recurrent_bpb(&model, &ds, n_train, n_win, n);
        println!("forced z={z_forced:+.1} (all blocks/channels): bpb={bpb:.4} acc={acc:.3}%");
    }
}

fn set_all_gate_z(model: &mut EvaModel, z_value: f32) {
    for b in model.blocks.iter_mut() {
        if let Mixer::Clock(c) = &mut b.mixer {
            if let Some(z) = &mut c.gate_z {
                let n = z.data.len();
                z.data = Arc::new(vec![z_value; n]);
            }
        }
    }
}

/// Same masked-bpb definition as mask_eval.rs, recurrent region only.
fn measure_recurrent_bpb(model: &EvaModel, ds: &TextDataset, from: usize, to: usize, n: usize) -> (f32, f32) {
    let seq = ds.seq;
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    let (mut ok, mut tot) = (0usize, 0usize);
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        let (s, v) = (logits.shape[0], logits.shape[1]);
        for i in 0..s {
            if i < n - 1 || i >= seq - 1 {
                continue;
            }
            let row = &logits.data[i * v..(i + 1) * v];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut esum = 0.0f64;
            for &x in row {
                esum += ((x - m) as f64).exp();
            }
            let per = -((row[target[i]] - m) as f64 - esum.ln());
            sum += per;
            cnt += 1;
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
        }
    }
    ((sum / cnt.max(1) as f64 / std::f64::consts::LN_2) as f32, 100.0 * ok as f32 / tot.max(1) as f32)
}
