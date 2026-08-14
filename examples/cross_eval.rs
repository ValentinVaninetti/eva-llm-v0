//! Fixing my own mistake: `prosa250.txt` is EXACTLY the prefix of the
//! first 250,000 bytes of `prosa.txt` (same md5). Comparing the "FINAL
//! VALIDATION" of a run over prosa250 against a run over the full
//! prosa.txt is NOT a valid comparison: each one defines its val as the
//! LAST 10% of ITS OWN file, so they land on different spans of text --
//! and worse, prosa250's val (bytes ~225k-250k) falls INSIDE the training
//! set of the run over the full prosa.txt (which only reserves val from
//! ~370k onward). There's no clean way to compare the two checkpoints as
//! they are without evaluating both on the SAME span.
//!
//! This script evaluates a given checkpoint over the LAST val_frac of a
//! given corpus -- exactly what `train.rs::eval_loss` does internally, but
//! exposed for any checkpoint/corpus pair. Usage: run the small checkpoint
//! (250 KB) over the val split of the full `prosa.txt` (which is 100%
//! unknown to it, it never trained on it) and compare THAT number against
//! the 2.699 bpb the large run already reported on its own val -- there it
//! really is the same span of text for both.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::ops;

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    let mut sum = 0.0f32;
    for wi in n_train..n {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        sum += ops::cross_entropy(&logits, &target).data[0];
    }
    let loss = sum / n_val as f32;
    let bpb = loss / std::f32::consts::LN_2;
    println!("eva cross_eval: {weights} over the val split of {data} ({n_val} windows)");
    println!("  loss {loss:.4} | {bpb:.3} bits/byte");
}
