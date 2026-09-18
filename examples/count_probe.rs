//! Does the context's count get lost in `lookup()`? What we noted
//! (recall.rs:117): `hits.iter()` sums everything before normalizing, and
//! the caller only receives the normalized distribution -- without the
//! count or which order answered.
//!
//! Tested with data: two synthetic contexts, one seen 500 times and
//! another exactly 2 (MIN_COUNT's floor), both with a single continuation.
//! If the count got dropped, the caller would see the SAME distribution
//! (certainty 1.0) and wouldn't be able to tell solid memory apart from a
//! 2-sample guess.
//!
//! Only uses the public API (`build`, `lookup`): doesn't touch `src/`,
//! doesn't load a model, doesn't use the GPU -- zero collision with
//! the sweep.

use eva_llm_v0::recall::Recall;

fn main() {
    let mut bytes: Vec<usize> = Vec::new();
    let a: [usize; 4] = [1, 2, 3, 4];
    let b: [usize; 4] = [7, 8, 9, 10];
    for _ in 0..500 {
        bytes.extend_from_slice(&a);
        bytes.push(9);
    }
    for _ in 0..2 {
        bytes.extend_from_slice(&b);
        bytes.push(11);
    }

    let table = Recall::build(&bytes);
    println!("eva count_probe: table built over {} bytes", bytes.len());

    let q_a = table.lookup(&a, 256).expect("A should be a hit (500 occurrences)");
    let q_b = table.lookup(&b, 256).expect("B should be a hit (2 occurrences, the floor)");

    println!("context A (seen 500 times):   p(9)  = {:.3}", q_a[9]);
    println!("context B (seen 2 times):     p(11) = {:.3}", q_b[11]);

    if (q_a[9] - q_b[11]).abs() < 1e-6 {
        println!("\n=== CONFIRMED ===");
        println!("Same certainty (1.0) for 500 samples and 2 samples. The count gets");
        println!("lost in lookup(): the caller cannot tell solid memory apart from a");
        println!("2-sample guess -- and a per-count lambda needs the count.");
    } else {
        println!("\n=== NOT lost: the counts are distinguishable ===");
    }
}
