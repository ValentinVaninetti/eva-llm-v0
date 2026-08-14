# evallm — what this is and how we work

Plain-language overview of the project: what we're testing, how the team
works together day to day, and an honest list of what we found and what we
didn't. Technical detail lives in `README.md`; day-by-day raw log lives in
the dated `TRABAJO-*.md` files (Spanish, kept private — working notes, not
polished writeups).

## What this is

`evallm` is a language model written from scratch in Rust — no ML framework,
no CUDA, only `std` plus a hand-written Vulkan GPU backend. The bet behind it:
commercial LLMs get more capable mostly by getting bigger, and a lot of that
cost is not inherent to the problem — it's business logic and off-the-shelf
architecture choices that were never re-examined once they worked well enough
at scale. This project tests a narrower claim: that some of that cost can be
removed by rethinking the architecture (how the model keeps memory over time)
rather than by adding more parameters or more hardware.

The core experimental idea is **ClockMem**: instead of the usual
quadratic-cost attention mechanism, each channel of the model's internal
state has its own *learned* decay rate — some channels are meant to forget
fast (local syntax, formatting), others to hold onto information for a long
time (topic, entities), each channel picking its own timescale during
training. Full architecture and numbers are in `README.md`.

## How we work: four collaborators, no single author

This isn't one person's code. Four collaborators, each with a different role,
share this repository:

- **Valentín** — owns the project, makes the calls that matter (what to try,
  what to trust, when to stop), runs the actual hardware.
- **Dante** (OpenCode) — writes code, runs experiments, often leads a given
  day's work.
- **GPT** — reviews results and proposes what to measure next; has no direct
  access to the repo or the machines, so it's briefed and answered through
  written messages relayed by whoever's available.
- **Claude ("Claudio")** — writes code, runs experiments, relays messages
  to/from GPT, keeps the shared log honest.

The coordination mechanism is a **shared, append-only, chronological log**
(the dated `TRABAJO-*.md` files) — everyone writes down what they ran, what
they found, and what they're doing next, in the order it happened, without
editing what someone else wrote. That's the entire protocol: no ticket
system, no meetings. Whoever's active reads the log to see where things
stand and picks up the next open thread.

A few rules, learned the hard way, that hold the process together:

- **Measure before building.** Don't optimize what you haven't profiled;
  intuition about where the slow part is has been wrong repeatedly in this
  project, expensively, every time it wasn't checked first.
- **"Code that touches X" is not "X works, measured."** Those are two
  different claims and they don't get conflated in the log.
- **One variable at a time.** Changing two things and seeing an improvement
  tells you nothing about which one mattered.
- **A negative result closes a line of work** until a genuinely new number
  reopens it — no re-litigating a measured result on a hunch.
- **Announce before touching `src/` or a file someone else has open.** Avoids
  silently clobbering someone's in-progress work.

## What we found

- **The bottleneck people assumed (matrix multiplication) was wrong.**
  Profiling under load showed the optimizer step and autograd bookkeeping
  (allocations, not arithmetic) ate far more time than expected; two boring,
  unglamorous fixes there cut total training time by ~47% without touching
  the experimental architecture at all.

- **ClockMem beats plain attention on the same budget**, on the (limited)
  test done so far: same block, same everything except the time-mixing
  mechanism, ClockMem wins by ~2% on held-out text across three seeds with no
  overlap between the two methods' results, and runs faster besides. Caveat,
  stated plainly: this was tested at a short context length (64 tokens) and
  on one small corpus — the exact regime where attention has the least room
  to show its actual advantage (long-range recall). That comparison, at long
  range, is unresolved and is what's being tested right now.

- **A real bug in the core mechanism, found by measuring, not by
  suspicion.** The design promise was "each channel picks its own memory
  timescale." Measured after training, that was false in practice: 98% of
  channels always collapsed to short memory (under 10 tokens), regardless of
  checkpoint, corpus, or context length. Tracing the actual gradient during
  training (not just staring at the final numbers) showed why: it's a
  mechanical side effect of the squashing function used to keep the decay
  parameter in range — gradient signal to the "slow" channels was roughly
  20x weaker than to the "fast" ones, near a saturation edge of that
  function, independent of what the training objective actually wanted.
  A one-line fix (a temperature term in that squashing function) rebalanced
  the spectrum — measured reproducible bit-for-bit across three independent
  runs on two different machines, two different backends (GPU and CPU-only),
  two different operating systems — at no cost in accuracy on this corpus.
  Whether that rebalanced spectrum actually *helps* on data that needs
  long-range memory is the open question the long-range synthetic benchmark
  (in progress as of this writing) is built to answer.

- **Generation got 10–20x faster** by carrying model state across tokens
  instead of recomputing the whole window per token — and, unlike a
  transformer's key/value cache, that state has fixed size regardless of how
  long the generated sequence gets.

## What we tried and it didn't pan out

Documented as closed, not hidden — a negative result is still a result:

- Injecting the k-gram lookup table's prediction as a training-time bias
  (hoping it would act as a regularizer): measured net negative.
- A different squashing function (`algebraic_sigmoid`) meant to fix the same
  gradient-starvation problem as the temperature fix above: initially looked
  promising, but closer numeric verification showed it helps in the wrong
  region of the parameter space — it was archived as inconclusive rather than
  quietly dropped, and led directly to designing the temperature fix that did
  work.
- Adding learned absolute position embeddings to the attention baseline, to
  make the comparison "fairer" to it: made attention's results worse across
  all three seeds, not better — the causal convolution already gave it
  positional structure; the extra parameters just diluted it.

## What's still open

- Whether the rebalanced memory spectrum from the temperature fix actually
  pays for itself on data with real long-range dependencies (the synthetic
  benchmark built specifically to test this is running now).
- The same performance numbers, measured on the actual target hardware (an
  AMD RX 580), not just the NVIDIA card used for development so far.
- Reducing autograd's own memory-allocation overhead, now the largest
  remaining chunk of training time that isn't raw arithmetic.

## A note on how this document is kept

Going forward: **all source code and all `.md` documentation pushed to this
repository is written in English** — comments, identifiers, commit-adjacent
docs, everything. The one exception is the dated `TRABAJO-*.md` working logs,
which stay as informal day-to-day notes in Spanish between collaborators and
aren't meant to be polished or public-facing.
