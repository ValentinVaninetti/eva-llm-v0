# evallm

An LLM written from scratch in Rust. No dependencies beyond `std`, no CUDA, no
framework: tensors, autograd, optimiser, tokeniser and a Vulkan compute backend
are all in this repo, and every piece is meant to be replaceable.

It exists to answer one question with measurements instead of opinions: **what
actually limits memory in a small language model?**

## The architecture

`EvaClock` — a byte-level transformer variant where attention is replaced by
**ClockMem**, a per-channel decaying state:

```
cur[c] = alpha[c] * cur[c] + beta * k[t,c] * v[t,c]
out    = q * cur * g
```

Each channel is an independent accumulator with its own learned decay. It is
O(sequence) instead of O(sequence²), and position is encoded for free in the
decay itself.

13.4M parameters, 256-symbol byte vocabulary, 5 blocks, trained on consumer
hardware (a GTX 1650 and an RX 580).

## What was measured

**On real text.** 1.716 bits per byte on the full *Don Quijote*, 8 seeds,
mean 1.7235 with a range of 0.027. It writes Spanish: "Sancho", "el rucio",
"encomendar".

**The central finding.** Its failure at associative recall is **not a memory
failure, it is a binding failure.** With 8 keys in context it identifies which
values are candidates 99.91% of the time — and then returns the value belonging
to the *wrong key* 76.5% of the time. It did not forget anything. It cannot tie
key to value.

**Where the binding lives.** Ablating the read of individual channels localises
the entire capability of a 5.4M-parameter model to **4 channels out of 256**.
Turning off any other 8 channels changes nothing; turning off channels 0-3
destroys it. That kills every explanation based on capacity.

**It is an optimisation lottery.** Same architecture, same corpus, same initial
weights — only the order in which windows are visited differs. The binding
behaviour appears in **2 runs out of 17**, and when it appears it reaches
99.77%. The element-wise write *can* represent a binding; gradient descent
almost never finds it. **The problem is optimisation, not representation.**

**And it does not transfer.** Eight seeds on real text show no such lottery:
the phenomenon is a property of the synthetic benchmark. What real text *does*
show, consistently across all eight seeds, is that in-context memory measurably
helps between roughly 32 and 256 bytes and switches off beyond 256.

## The record

**`PAPER-DRAFT.md` is the whole thing**, and it is the most useful file here.
Every number, how it was obtained, and — deliberately kept — four correction
banners marking sections whose conclusions were later shown to be wrong,
including one where a surprisal was read with the wrong sign and inverted a
result from positive to null.

Superseded sections are not deleted. The sequence of what was believed, and on
what evidence, is part of the record.

## Running it

```sh
cargo build --release
cargo test --release              # 87 tests

# any UTF-8 text file works; the numbers above are on Don Quijote,
# public domain, from Project Gutenberg.
./target/release/eva_llm_v0 train --data data/quijote.txt --epochs 10 --out eva.weights
./target/release/eva_llm_v0 gen   --weights eva.weights --prompt "En un lugar"
./target/release/eva_llm_v0 info  --weights eva.weights

./target/release/eva_llm_v0 help  # every flag, with its default
```

`data/` is not in the repository: the corpora are large and freely available,
and the generated benchmarks are reproducible with
`cargo run --release --example generate_associative` and `generate_synthetic`,
whose headers give the exact arguments.

`logs/` is: 85 raw training and evaluation logs. They are the artifacts
`PAPER-DRAFT.md` verifies its figures against, which is why they are committed
unedited rather than summarised.

Diagnostics live in `examples/`, each stating at its top the question it
answers and the number that would kill it: `query_swap` (does the model read
the query at all, or just emit a value that is in play?), `entity_retrieval`
(does memory help on real text, by distance?), `mask_eval` (conditional bpb on
the long-range synthetic benchmark), `state_probe` (is the failure storage or
read?), `alpha_dispersion` (the decay spectrum). The per-channel read ablation
that localises the binding to 4 channels is `EVA_READ_MASK=c1,c2,...`, which
any of them honours, since it lives in the forward.

## What this is not

It is not competitive with a commercial model and it is not trying to be. It is
a controlled instrument for one question, small enough that a single person can
measure every part of it.

See `ROADMAP.md` for what is open.
