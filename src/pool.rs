//! A persistent worker pool for data-parallel kernels.
//!
//! WHY IT EXISTS: spawning threads per `matmul` call cost more than the
//! multiplication for small shapes. Threads here are created once and parked.
//!
//! THE BUG THIS FILE WAS REWRITTEN AROUND (2026-08-11), because the shape of
//! it is easy to rebuild by accident: the worker waited on a condvar with `if`
//! instead of `while`, and every finishing worker called `notify_all`. So a
//! worker that had already parked was woken by its peers, re-read the job that
//! was never cleared, and **ran it again** -- measured 23 to 31 re-runs of each
//! chunk for a single submission. Harmless-looking (a matmul band recomputed is
//! the same band) until the caller returns and frees its buffers: the stragglers
//! then write into freed memory. That is the SIGSEGV.
//!
//! Three properties keep it from coming back, and each one is tested:
//!   1. A condvar wakeup is a hint, never a fact: every wait re-checks `gen`.
//!   2. Workers signal the submitter on a SEPARATE condvar, so finishing work
//!      cannot wake a peer. It is also faster -- no thundering herd.
//!   3. `run` clears the job while holding the lock before returning, so no
//!      pointer captured by a closure can outlive the call that published it.

use std::cell::Cell;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

type Job = Arc<dyn Fn(usize) + Send + Sync>;

struct State {
    /// Bumped once per submission. A worker compares it against what it last
    /// saw; equal means "nothing new for me", whatever woke it up.
    gen: usize,
    chunks: usize,
    /// Acks, including workers that had no piece to run. `run` waits for all
    /// of them, so the pool is fully idle before it returns.
    done: usize,
    job: Option<Job>,
    stop: bool,
}

struct Ctl {
    m: Mutex<State>,
    /// Submitter -> workers. Only `run` signals this one.
    work: Condvar,
    /// Workers -> submitter. Only the last worker signals this one.
    idle: Condvar,
}

pub struct ThreadPool {
    ctl: Arc<Ctl>,
    /// Serializes submissions. Two threads calling `run` at once would
    /// overwrite each other's job while workers were reading it.
    submit: Mutex<()>,
    workers: Vec<JoinHandle<()>>,
}

thread_local! {
    /// True while this thread is inside a job. A job that calls `run` again
    /// would wait for workers that are busy with the outer job -- a deadlock.
    /// Nested calls run inline instead.
    static IN_JOB: Cell<bool> = const { Cell::new(false) };
}

/// Runs `body` with the nested-call guard raised, and lowers it after.
fn in_job<R>(body: impl FnOnce() -> R) -> R {
    IN_JOB.with(|f| f.set(true));
    let r = body();
    IN_JOB.with(|f| f.set(false));
    r
}

impl ThreadPool {
    pub fn new() -> Self {
        let ctl = Arc::new(Ctl {
            m: Mutex::new(State { gen: 0, chunks: 0, done: 0, job: None, stop: false }),
            work: Condvar::new(),
            idle: Condvar::new(),
        });
        // One less than the core count: the submitting thread runs a piece too.
        let n = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1).saturating_sub(1);
        let workers = (0..n)
            .map(|w| {
                let ctl = ctl.clone();
                std::thread::Builder::new()
                    .name(format!("eva-pool-{w}"))
                    .spawn(move || worker(ctl, w, n))
                    .expect("spawn pool worker")
            })
            .collect();
        ThreadPool { ctl, submit: Mutex::new(()), workers }
    }

    /// Number of background threads. The caller participates as well, so the
    /// useful chunk count is `workers() + 1`.
    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Runs `f` over `chunks` pieces and blocks until every one has finished.
    ///
    /// The caller executes piece 0 itself; workers take `1..chunks`. When this
    /// returns, no thread is still touching anything `f` captured -- which is
    /// what makes it sound for `f` to close over raw pointers into the caller's
    /// buffers.
    pub fn run<F>(&self, chunks: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        if chunks == 0 {
            return;
        }
        // A job that calls back into the pool cannot wait for the pool.
        if IN_JOB.with(|f| f.get()) {
            for t in 0..chunks {
                f(t);
            }
            return;
        }

        let nworkers = self.workers.len();
        let chunks = chunks.min(nworkers + 1);
        if chunks == 1 || nworkers == 0 {
            in_job(|| f(0));
            return;
        }

        let _one_at_a_time = self.submit.lock().unwrap();
        let job: Job = Arc::new(f);

        let mut st = self.ctl.m.lock().unwrap();
        st.job = Some(job.clone());
        st.chunks = chunks;
        st.done = 0;
        st.gen = st.gen.wrapping_add(1);
        drop(st);
        self.ctl.work.notify_all();

        in_job(|| job(0));

        let mut st = self.ctl.m.lock().unwrap();
        while st.done < nworkers {
            st = self.ctl.idle.wait(st).unwrap();
        }
        // Cleared under the lock: after this, a worker woken by anything at all
        // finds no job to re-run. This is the line that turns the old
        // use-after-free into an impossibility rather than a race we lost.
        st.job = None;
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let mut st = self.ctl.m.lock().unwrap();
        st.stop = true;
        drop(st);
        self.ctl.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(ctl: Arc<Ctl>, w: usize, nworkers: usize) {
    let mut seen = 0usize;
    IN_JOB.with(|f| f.set(true));
    loop {
        let job = {
            let mut st = ctl.m.lock().unwrap();
            // `while`, not `if`. A wakeup only means "look again".
            while !st.stop && st.gen == seen {
                st = ctl.work.wait(st).unwrap();
            }
            if st.stop {
                return;
            }
            seen = st.gen;
            // Workers past the chunk count still fall through to the ack below:
            // `run` counts every worker, not every piece.
            if w + 1 < st.chunks { st.job.clone() } else { None }
        };

        if let Some(job) = job {
            job(w + 1);
        }

        let mut st = ctl.m.lock().unwrap();
        st.done += 1;
        let last = st.done == nworkers;
        drop(st);
        // Only the last one signals, and only the submitter is listening.
        if last {
            ctl.idle.notify_one();
        }
    }
}

/// Puntero crudo que es `Send + Sync`, para repartir un buffer en bandas.
///
/// Existe por una restricción del pool y no por gusto: `run` pide `Send + Sync
/// + 'static`, así que un `&mut [f32]` de quien llama no entra. Es sano
/// SÓLO porque `run` no vuelve hasta que terminó cada pieza -- nada de lo que
/// se arma acá sobrevive al préstamo que lo originó. Si eso cambia, esto se
/// vuelve un use-after-free.
#[derive(Clone, Copy)]
pub struct Ptr<T>(pub T);

unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

impl Ptr<*const f32> {
    pub fn as_ref(self, len: usize) -> &'static [f32] {
        unsafe { std::slice::from_raw_parts(self.0, len) }
    }

    pub fn at(self, off: usize) -> Ptr<*const f32> {
        Ptr(unsafe { self.0.add(off) })
    }
}

impl Ptr<*mut f32> {
    pub fn add(self, off: usize) -> Ptr<*mut f32> {
        Ptr(unsafe { self.0.add(off) })
    }

    pub fn as_mut(self, len: usize) -> &'static mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0, len) }
    }
}

static POOL: OnceLock<ThreadPool> = OnceLock::new();

pub fn global() -> &'static ThreadPool {
    POOL.get_or_init(ThreadPool::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The regression that matters: each piece runs EXACTLY once per call.
    /// The old pool ran them 23-31 times and then wrote into freed memory.
    #[test]
    fn every_piece_runs_exactly_once() {
        let pool = ThreadPool::new();
        let chunks = pool.workers() + 1;
        let hits: Arc<Vec<AtomicUsize>> =
            Arc::new((0..chunks).map(|_| AtomicUsize::new(0)).collect());

        for _ in 0..200 {
            for h in hits.iter() {
                h.store(0, Ordering::Relaxed);
            }
            let seen = hits.clone();
            pool.run(chunks, move |t| {
                seen[t].fetch_add(1, Ordering::Relaxed);
            });
            for (t, h) in hits.iter().enumerate() {
                assert_eq!(1, h.load(Ordering::Relaxed), "pieza {t} no corrió exactamente una vez");
            }
        }
    }

    /// After `run` returns, nothing may still be executing. If a straggler
    /// existed it would be writing into the caller's frame -- the SIGSEGV.
    #[test]
    fn nothing_runs_after_run_returns() {
        let pool = ThreadPool::new();
        let live = Arc::new(AtomicUsize::new(0));

        for _ in 0..200 {
            let inside = live.clone();
            pool.run(pool.workers() + 1, move |_| {
                inside.fetch_add(1, Ordering::Relaxed);
            });
            let after = live.load(Ordering::Relaxed);
            std::thread::yield_now();
            assert_eq!(after, live.load(Ordering::Relaxed), "quedó trabajo corriendo tras run()");
        }
    }

    /// Fewer pieces than workers: the extra workers must still acknowledge, or
    /// `run` waits forever.
    #[test]
    fn fewer_chunks_than_workers_does_not_hang() {
        let pool = ThreadPool::new();
        let hits = Arc::new(AtomicUsize::new(0));
        let seen = hits.clone();
        pool.run(1, move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(1, hits.load(Ordering::Relaxed));
    }

    /// A job that calls the pool again must not deadlock waiting for itself.
    #[test]
    fn nested_run_falls_back_to_inline() {
        let pool = ThreadPool::new();
        let hits = Arc::new(AtomicUsize::new(0));
        let seen = hits.clone();
        pool.run(pool.workers() + 1, move |_| {
            let inner = seen.clone();
            crate::pool::global().run(4, move |_| {
                inner.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(4 * (pool.workers() + 1), hits.load(Ordering::Relaxed));
    }
}
