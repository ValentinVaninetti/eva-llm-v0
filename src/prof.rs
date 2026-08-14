//! A stopwatch per label, to know where the time goes.
//!
//! WHY THIS EXISTS: twice in the same day, intuition picked the wrong
//! bottleneck. First "the shader needs optimizing" when it was the memory
//! flags (tiling gave 1.13x, the flag gave 2.5x), then "the bottleneck is
//! ClockMem" based on indirect evidence. Guessing is expensive and
//! measuring is cheap.
//!
//! There's no `perf` on this machine, so: atomic counters per label, zero
//! cost when it's off, and a report at the end.
//!
//! ```text
//! EVA_PROFILE=1 cargo run --release -- train --data ...
//! ```
//!
//! Measures accumulated WALL TIME, so if something runs on several threads
//! it adds up to more than the clock. That's intentional: what we're after
//! is where the effort goes, not how long the run took.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// The labels are fixed: a `HashMap` with a lock on the hot path would end
/// up measuring the lock itself.
#[derive(Clone, Copy)]
pub enum P {
    Matmul,
    ClockFwd,
    ClockBwd,
    Backward,
    Optim,
}

impl P {
    const ALL: [(P, &'static str); 5] = [
        (P::Matmul, "matmul"),
        (P::ClockFwd, "clockmem fwd"),
        (P::ClockBwd, "clockmem bwd"),
        (P::Backward, "backward (all)"),
        (P::Optim, "optimizer"),
    ];
}

static NANOS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static HITS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("EVA_PROFILE").is_ok())
}

/// Times however long `body` takes. With profiling off it's just a call and
/// nothing more.
pub fn time<R>(p: P, body: impl FnOnce() -> R) -> R {
    if !on() {
        return body();
    }
    let t0 = Instant::now();
    let r = body();
    let i = p as usize;
    NANOS[i].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    HITS[i].fetch_add(1, Ordering::Relaxed);
    r
}

/// Final report. `wall` is how long it actually took, to scale the
/// percentages.
pub fn report(wall: std::time::Duration) {
    if !on() {
        return;
    }
    let total = wall.as_secs_f64();
    eprintln!("\n-- where the time went --");
    for (p, name) in P::ALL {
        let i = p as usize;
        let s = NANOS[i].load(Ordering::Relaxed) as f64 / 1e9;
        let n = HITS[i].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        eprintln!(
            "  {name:<16} {s:8.2} s  {:5.1}%  ({n} calls, {:.1} us each)",
            100.0 * s / total,
            s * 1e6 / n as f64,
        );
    }
    eprintln!("  {:<16} {total:8.2} s", "wall clock");
}
