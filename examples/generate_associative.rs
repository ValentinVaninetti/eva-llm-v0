//! Generator for the ASSOCIATIVE RECALL benchmark, written after
//! entity-retrieval on the Quijote showed that neither T=1, T=1.3 nor a
//! fixed-slow-decay SSM ever beat the no-memory reference at long range
//! (PAPER-DRAFT.md 4.7-4.9).
//!
//! This is CONTENT addressing, not POSITION addressing. The earlier synthetic
//! benchmark (`generate_synthetic.rs`) uses y[t] = F(x[t-N]) with a fixed,
//! globally-shared lag and permutation: the model always knows how far back to
//! look, and F is memorized once across the whole training set. Real recall
//! needs the opposite -- the same name reappears at any distance and its
//! binding has to be re-derived from context every time.
//!
//! The standard associative-recall / induction-head task (MQAR family: Olsson
//! et al. 2022; Fu et al. 2023 H3/Hyena; Arora et al. 2023 Based/MQAR) adapted
//! to a byte-level LM:
//!
//!   KEY    [0, 32)     VALUE  [32, 64)     FILLER [64, 256), random gaps
//!
//! Per window, bindings are FRESH (no global table): emit (KEY, VALUE) pairs
//! drawn from `active_keys`, spaced by filler runs. A key's FIRST appearance
//! binds a random, unpredictable value (analogous to `cold` in
//! entity_retrieval). Every later occurrence is a query: the key alone is
//! written, and the next byte is the value bound earlier -- predictable ONLY by
//! recalling that key's binding, at a random distance not known in advance.
//!
//! USAGE: cargo run --release --example generate_associative -- <active_keys> <seq> <win> <seed> <out>
//!   active_keys  distinct keys reused per window (<=32; try 8)
//!   seq          window length (512, as in the rest of the benchmarks)
//!   win          number of windows
//!   seed         data seed: bindings and fillers depend on it. The KEY/VALUE/
//!                FILLER ranges are fixed constants -- there is no global F.
//!   out          output file

use eva_llm_v0::rng::Rng;

pub const KEY_LO: u16 = 0;
pub const KEY_HI: u16 = 32;
pub const VAL_LO: u16 = 32;
pub const VAL_HI: u16 = 64;
pub const FILLER_LO: u16 = 64;
pub const FILLER_HI: u16 = 256;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let active_keys: u8 = args.get(1).and_then(|s| s.parse().ok())
        .expect("usage: generate_associative <active_keys> <seq> <win> <seed> <out> [filler_min] [filler_max]");
    let seq: usize = args.get(2).and_then(|s| s.parse().ok())
        .expect("usage: generate_associative <active_keys> <seq> <win> <seed> <out> [filler_min] [filler_max]");
    let win: usize = args.get(3).and_then(|s| s.parse().ok())
        .expect("usage: generate_associative <active_keys> <seq> <win> <seed> <out> [filler_min] [filler_max]");
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok())
        .expect("usage: generate_associative <active_keys> <seq> <win> <seed> <out> [filler_min] [filler_max]");
    let out = args.get(5).cloned()
        .expect("usage: generate_associative <active_keys> <seq> <win> <seed> <out> [filler_min] [filler_max]");
    let filler_min: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(4);
    let filler_max: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(40);

    assert!(active_keys >= 1 && (active_keys as u16) <= (KEY_HI - KEY_LO), "active_keys must be in [1, 32]");
    assert!(win > 0 && seq > 8, "win > 0 and seq > 8 required");
    assert!(filler_min >= 1 && filler_min <= filler_max, "need 1 <= filler_min <= filler_max");

    let mut rng = Rng::new(seed);
    let mut bytes = vec![0u8; win * seq];

    let mut total_queries = 0u64;
    let mut total_bindings = 0u64;
    let mut gap_min = usize::MAX;
    let mut gap_max = 0usize;
    let mut gap_sum = 0u64;

    for w in 0..win {
        let base = w * seq;
        let mut bound: [Option<u16>; 32] = [None; 32];
        let mut last_pos: [Option<usize>; 32] = [None; 32];
        let mut pos = 0usize; // offset within window
        while pos < seq {
            let key = (rng.next_u64() % active_keys as u64) as u16;
            let is_binding = bound[key as usize].is_none();
            let value = if is_binding {
                let v = VAL_LO + (rng.next_u64() % (VAL_HI - VAL_LO) as u64) as u16;
                bound[key as usize] = Some(v);
                total_bindings += 1;
                v
            } else {
                let v = bound[key as usize].unwrap();
                let gap = pos - last_pos[key as usize].unwrap();
                gap_min = gap_min.min(gap);
                gap_max = gap_max.max(gap);
                gap_sum += gap as u64;
                total_queries += 1;
                v
            };
            last_pos[key as usize] = Some(pos);

            bytes[base + pos] = (KEY_LO + key) as u8;
            pos += 1;
            if pos >= seq {
                break;
            }
            bytes[base + pos] = value as u8;
            pos += 1;
            if pos >= seq {
                break;
            }

            let filler_len = filler_min + (rng.next_u64() as usize % (filler_max - filler_min + 1));
            for _ in 0..filler_len {
                if pos >= seq {
                    break;
                }
                bytes[base + pos] = (FILLER_LO + (rng.next_u64() % (FILLER_HI - FILLER_LO) as u64) as u16) as u8;
                pos += 1;
            }
        }
    }

    // Self-check: recompute bindings/queries from the raw bytes alone (no
    // side information beyond the fixed byte ranges) and verify every
    // query byte matches the binding established earlier in that window.
    let mut checked = 0u64;
    let mut failures = 0u64;
    for w in 0..win.min(50) {
        let base = w * seq;
        let mut bound: [Option<u8>; 32] = [None; 32];
        let mut pos = 0usize;
        while pos + 1 < seq {
            let b = bytes[base + pos];
            if (b as u16) >= KEY_LO && (b as u16) < KEY_HI {
                let key = b as usize;
                let value = bytes[base + pos + 1];
                match bound[key] {
                    None => bound[key] = Some(value),
                    Some(expected) => {
                        checked += 1;
                        if value != expected {
                            failures += 1;
                        }
                    }
                }
                pos += 2;
            } else {
                pos += 1;
            }
        }
    }

    std::fs::write(&out, &bytes).expect("could not write output file");

    println!("eva generate_associative: {out}");
    println!("  active_keys={active_keys} seq={seq} windows={win} seed={seed}");
    println!("  KEY=[{KEY_LO},{KEY_HI}) VALUE=[{VAL_LO},{VAL_HI}) FILLER=[{FILLER_LO},{FILLER_HI}) filler_len=[{filler_min},{filler_max}]");
    println!("  bindings={total_bindings} queries={total_queries} (ratio queries/bindings={:.2})",
        total_queries as f64 / total_bindings.max(1) as f64);
    println!("  gap (bytes since the key's previous occurrence): min={gap_min} max={gap_max} mean={:.1}",
        gap_sum as f64 / total_queries.max(1) as f64);
    println!("  self-check (first 50 windows, recomputed from raw bytes only): {checked} query positions checked, {failures} failures");
    assert_eq!(failures, 0, "self-check failed: some query byte does not match its window's earlier binding");
}
