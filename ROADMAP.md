# Roadmap

Ordered by what a measurement says, not by what sounds interesting. An item only
moves when a number moves it; a null result closes a line until a new number
reopens it.

## The open question

**Why does gradient descent find the binding 2 times out of 17?**

Everything measured points here. The element-wise write *can* represent a
key→value binding — it reaches 99.77% when it finds one — so the limit is not
representational. What decides the lottery is still unknown.

Three axes have been eliminated:

| axis | verdict |
|---|---|
| channels per key | **refuted** — two configs with identical channels-per-key differ by 21 points |
| model width | does not predict anything |
| number of keys | does not predict it either |
| the output gate | **discarded** — removing it gave 0 of 12 against a base rate of 2 of 17 |
| data ordering | **it is the lottery** (11.8%) |
| **initialisation** | **never tested** — was impossible until `EVA_INIT_SEED` existed |

Initialisation is the one structural axis left untouched, and the first attempt
to measure it is void: a wrapper copied the step-2500 checkpoint instead of the
step-5000 one in all twelve runs. Detected because twelve results were identical
to four decimals. Ten machine-hours lost, axis still open.

## Next

**1. The initialisation axis.** Vary weights with the seed fixed. Twelve short
runs. The protocol is cheap because the outcome is decided in the first quarter
of training: at step 5000 the winner is already separated from the blind runs,
which makes runs 4x cheaper and was verified 4 of 4.

**2. Measure #58.** Low-rank outer-product write, implemented and gradient-checked
behind `EVA_WRITE_LOWRANK`, **not yet measured**. Its criterion is fixed in
advance: thresholds derived from the dispersion of the same bucket across seeds,
with staged stopping rules so the likely cost is one hour, not eight. A first
stage only checks whether the new path is used at all — if `M` stays inert, the
answer is "on real text the gradient does not use it" and that is publishable.

**3. Group the kernel arguments.** The forward/backward kernels take up to
twelve loose tensors (`clockmem_inject_learn_g`: q, k, v, g, alpha, beta,
alpha_read, inv_beta, z, floor, n, s0). `clippy::too_many_arguments` is
switched off for the package with the reason written in `Cargo.toml`, which is
a declared debt, not a fix: the signatures are mechanical and easy to pass in
the wrong order. Grouping them into a struct is safe to do only with a
characterisation test over the whole forward, since the tests cover the lib but
not the diagnostics in `examples/`.

**4. An external reference.** Everything so far compares ClockMem against
baselines built *in this same codebase*, and the draft says so plainly. A
standard transformer of matched parameter count on the same corpus has never
been run. Until it is, "better than this reference baseline" is the honest
claim, not "better than attention".

## Known limits

**The synthetic-to-real bridge is not demonstrated.** The binding work lives on
a benchmark of 2 to 8 keys. On real text the same measurements give null. Do
not read the toy results as statements about language.

**The final checkpoint is not the best one.** The winning run scored 99.77% at
step 20,000 and 92.18% at the end, 249 steps later, for 0.006 bpb. Every figure
in the draft comes from a last-step checkpoint and may therefore understate the
peak.

**One machine at a time.** Comparable runs go to the same machine in series.
Splitting a 2x2 across two machines once cost the entire experiment to a
confound.
