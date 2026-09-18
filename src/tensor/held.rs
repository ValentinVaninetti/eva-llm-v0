//! How much memory the autograd graph holds, measured rather than estimated.
//!
//! WHY THIS EXISTS: backpropagation has to hold onto whatever the forward
//! pass produced, and THAT is why training needs expensive hardware -- not
//! the compute, the memory. Before trying to bring it down you need to know
//! how much there is, separated from the parameters and from the optimizer
//! state, which cost the same either way.
//!
//! The process's peak (`VmHWM`) doesn't work for this: it mixes everything
//! together and is also a watermark of the allocator, so it counts
//! fragmentation too. Here we count the bytes the graph has ALIVE at each
//! instant.
//!
//! And ONE DETAIL OF OUR CASE that changes the size of the prize: `finalize`
//! **clones the inputs** of every operation into `saved_v`. In other words we
//! don't hold activations: we hold **copies** of activations. The per-op
//! breakdown below is what tells you which ones over-clone.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;

/// The mechanism, kept separate from the global instance.
///
/// WHY SEPARATE: the first version had the counters as loose statics and the
/// test asserted directly on them. Rust tests run in parallel and almost all
/// of them build graphs, so any other test would move the counter between
/// the read and the assertion. **Flaky by construction**, and it showed up
/// once in the full suite before anyone went looking for it. With the
/// mechanism split out, the test uses its own instance and there's no race.
pub struct Counter {
    live: AtomicUsize,
    peak: AtomicUsize,
    nodes: AtomicUsize,
    nodes_peak: AtomicUsize,
    grads_peak: AtomicUsize,
    /// Bytes accumulated per operation: not what's currently alive but the
    /// total that ever passed through, to know who holds the most over the
    /// whole training run.
    by_op: Mutex<Vec<(&'static str, usize, usize)>>,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    pub const fn new() -> Self {
        Counter {
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            nodes: AtomicUsize::new(0),
            nodes_peak: AtomicUsize::new(0),
            grads_peak: AtomicUsize::new(0),
            by_op: Mutex::new(Vec::new()),
        }
    }

    pub fn enter(&self, op: &'static str, bytes: usize) {
        let now = self.live.fetch_add(bytes, Relaxed) + bytes;
        self.peak.fetch_max(now, Relaxed);
        let n = self.nodes.fetch_add(1, Relaxed) + 1;
        self.nodes_peak.fetch_max(n, Relaxed);

        let mut t = self.by_op.lock().unwrap();
        match t.iter_mut().find(|(o, _, _)| *o == op) {
            Some((_, b, c)) => {
                *b += bytes;
                *c += 1;
            }
            None => t.push((op, bytes, 1)),
        }
    }

    pub fn leave(&self, bytes: usize) {
        self.live.fetch_sub(bytes, Relaxed);
        self.nodes.fetch_sub(1, Relaxed);
    }

    pub fn live(&self) -> usize {
        self.live.load(Relaxed)
    }

    pub fn peak(&self) -> usize {
        self.peak.load(Relaxed)
    }
}

static G: Counter = Counter::new();

/// A node entered the graph.
pub fn enter(op: &'static str, bytes: usize) {
    G.enter(op, bytes);
}

/// A node was freed.
pub fn leave(bytes: usize) {
    G.leave(bytes);
}

pub fn peak_mb() -> f64 {
    G.peak() as f64 / 1_048_576.0
}

pub fn live_mb() -> f64 {
    G.live() as f64 / 1_048_576.0
}

pub fn nodes_peak() -> usize {
    G.nodes_peak.load(Relaxed)
}

/// How much the gradient map takes up at its highest point.
///
/// `backward` creates a new `Vec` for EVERY tensor that receives a
/// gradient -- not just for parameters -- and throws it away at the end of
/// the step.
pub fn grads(bytes: usize) {
    G.grads_peak.fetch_max(bytes, Relaxed);
}

/// Report: how much the graph holds and who's holding it.
pub fn report(params: usize) {
    let mut t = G.by_op.lock().unwrap().clone();
    t.sort_by_key(|(_, b, _)| std::cmp::Reverse(*b));
    let total: usize = t.iter().map(|(_, b, _)| *b).sum();

    let weights = params * 4;
    let adam = params * 8;
    println!("\n-- what holds the training memory --");
    println!("  parameters            {:8.1} MB   (unavoidable)", weights as f64 / 1_048_576.0);
    println!("  AdamW state           {:8.1} MB   (unavoidable today)", adam as f64 / 1_048_576.0);
    println!("  GRAPH, own peak       {:8.1} MB", peak_mb());
    println!("  live nodes, peak      {:8}", nodes_peak());
    println!("  graph alive now       {:8.1} MB   (0 if everything was freed)", live_mb());
    println!("  GRADIENT MAP          {:8.1} MB   <- a new Vec per tensor, every step",
        G.grads_peak.load(Relaxed) as f64 / 1_048_576.0);

    println!("\n-- who holds, accumulated over the whole training run --");
    for (op, b, c) in t.iter().take(8) {
        println!("  {op:<16} {:9.1} MB in {c:>7} nodes   {:4.0}%",
            *b as f64 / 1_048_576.0, 100.0 * *b as f64 / total.max(1) as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On its OWN instance, not the global one: the previous version
    /// asserted on the shared counter while the rest of the suite was
    /// building graphs in parallel. It was flaky by construction and showed
    /// up once.
    #[test]
    fn what_goes_in_comes_out() {
        let c = Counter::new();
        c.enter("test", 1000);
        assert_eq!(1000, c.live());
        c.leave(1000);
        assert_eq!(0, c.live(), "the counter ended up unbalanced");
    }

    #[test]
    fn the_peak_does_not_go_down() {
        let c = Counter::new();
        c.enter("test", 5_000_000);
        c.leave(5_000_000);
        assert_eq!(5_000_000, c.peak(), "the peak went down when freeing");
        assert_eq!(0, c.live());
    }

    /// And the property the old test did NOT check: that it survives
    /// concurrency. The production counter is touched by several threads if
    /// nodes ever get created in parallel, and a lost addition would ruin
    /// the whole measurement.
    #[test]
    fn it_survives_many_threads() {
        let c = std::sync::Arc::new(Counter::new());
        let mut threads = Vec::new();
        for _ in 0..8 {
            let c = c.clone();
            threads.push(std::thread::spawn(move || {
                for _ in 0..2000 {
                    c.enter("concurrent", 64);
                    c.leave(64);
                }
            }));
        }
        for h in threads {
            h.join().unwrap();
        }
        assert_eq!(0, c.live(), "additions or subtractions were lost between threads");
        assert!(c.peak() >= 64);
    }
}
