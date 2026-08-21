# Where does a gated linear-recurrent memory lose long-range information? A causal dissection

**Status: working draft, not for external circulation.** This document is being built incrementally as results land, by a small multi-agent team (see Acknowledgments). Every number in this draft has been independently verified against the raw training/eval logs and checkpoints in this repository before being written down here — see the "Verification" note at the end of each experiment. Sections marked **[OPEN]** are not yet settled and should not be read as conclusions.

## Abstract

We study a small (13M-parameter) from-scratch language model, EvaClock, whose recurrent memory mechanism (ClockMem) gives every channel an independently learned decay rate, intended to let the network discover its own mix of short- and long-range memory. We show that this promise fails silently in two independent, mechanistically distinct ways, both diagnosable and both fixable without adding capacity:

1. **A parametrization trap**, not a training failure: the decay parameter's own gradient is proportional to `alpha·(1-alpha)`, so channels pushed toward fast decay lose the gradient needed to ever reconsider. Ruled out three alternative explanations (inter-window persistence, corpus size, context length) by direct measurement before accepting this one. A one-line temperature reparametrization restores gradient to the affected region and produces a distributed memory spectrum, reproduced bit-for-bit across two machines, two backends (GPU/CPU), and two operating systems, at no measured cost in accuracy on the corpus tested.

2. **A transport failure downstream of a working memory.** On a synthetic long-range recall task, we show with a "capacity oracle" that (a) the recurrent state provably retains the needed information arbitrarily far back, and (b) an exact linear readout of that information exists and is representable — yet the model still fails end-to-end. We localize the loss to a single multiplicative gate (`q·read·g`) and show, via a sequence of increasingly permissive interventions, that a *learnable* route recovers the missing capability only once the parametrization is structurally restricted to match the true write dynamics (2 parameters/channel) rather than left as a free window (K parameters/channel) — and that reintroducing the gate is safe only if it is architecturally prevented from fully closing.

A secondary, cross-cutting observation: in two independent experiments, the trained model did not converge to the "canonical" (oracle-matching) solution when a family of equally-good solutions existed — it settled on an arbitrary member of that family instead. We flag this as a phenomenon worth measuring directly, not yet as a result.

Applied to real text (a full novel, *Don Quijote*), the temperature fix shows no in-distribution benefit — global bpb, and a targeted entity-retrieval metric built specifically to isolate long-range recall from general language modeling, are both null. Its one positive real-text result is narrower than hoped: under state carried across windows the model was never trained for (an out-of-distribution regime), the fixed model remains stable while the unfixed one collapses — reproduced independently on two machines in a full cross-validated control. A follow-up baseline (a fixed, non-learned slow decay rate on every channel, built inside this same codebase rather than porting an external SSM) shows the same carry-stability, ruling out that it is specific to the learned-and-compressed mechanism of the temperature fix — any sufficiently slow recurrence gets it for free. In no condition tested, including this baseline, does carried state at long range ever outperform the same model's own no-memory reference: stability is not usefulness, in either mechanism.

The real-text null results motivated a reframing: the synthetic benchmark used throughout Section 4 tests purely *positional* recall (a fixed, globally-known lag), not the *content-addressed* recall that entity recognition in real text actually requires. A new synthetic benchmark built for content-addressed recall (in-context key→value binding at random, unknown-in-advance distances — the standard associative-recall/induction-head task family) settles two things. With enough *unique* training data, ClockMem solves it: 99.85% held-out accuracy at short range and still 97.7% at 256 bytes, with both no-leakage controls holding — an outcome that contradicts a design claim asserted in this codebase's own source comments, and that a smaller-data version of the same experiment had badly underestimated. A full-attention baseline on the identical task reaches only ~47%, flat across distance. Classifying what each model actually predicts turns that number into a mechanism: attention places 89.7% of its predictions on one of the two values genuinely bound in that window — it locates the candidate set nearly perfectly — but then splits them 47/42, a coin flip between correct candidates. It retrieves without binding. ClockMem at 2 keys, by contrast, confuses the two keys in exactly 0.00% of 5,967 queries. But the capability does not scale: at 8 keys, matched for data and distance, ClockMem drops to 31.8% and falls into attention's failure mode — it identifies the candidate set even better (99.91%) while picking the wrong key's value 76.5% of the time. The 8-key failure is therefore specifically a *binding* failure, not degraded memory or an unlearned task, and it appears exactly where a partition of 512 channels across keys would run out of room — which is what an element-wise (rather than outer-product) write permits and an outer-product write would not require. A convergence run is in flight to establish whether that cross-key confusion shrinks with more training.

## 1.1 Scope, relative to the closest precedent found so far

(See Section 6 for the precedent itself.) The boundary, stated plainly before the rest of this draft is read:

```
SSM precedent (decay / temporal resolution):
    can a recurrent state RETAIN information and RESOLVE a delayed-copy task?

this work:
    decay  +  addressable read  +  output gate  +  gradient route
        -> can a recurrent state retain, recover, AND actually USE long-range
           memory, without the interface that reads it destroying the signal?
```

Our sharpest result is not that long-range recall is achievable (the precedent already suggests decay parametrization alone can get there for a plain SSM read). It is the chain showing that *achievability of storage/readout and achievability of a learnable interface to that readout are separate problems*: an oracle read into the standard multiplicative gate fails (~8.0 bpb) exactly as often as it succeeds once the gate is removed (0.035) or replaced by a structured, floored, still-gated route (also ≈0.035-0.036). As of this draft, this specific interface-level failure mode has not been found in the precedent's abstract — but see the honest caveats on that in Section 6, including that a head-to-head benchmark against a modern SSM baseline has not yet been run.

## 1. Motivation

[TODO — Valentín/team to write the framing paragraph: why efficiency-via-architecture instead of efficiency-via-scale, what EvaClock is testing. Draft skeleton left here so it isn't forgotten; see `README.md` for the informal version to adapt.]

## 2. Architecture

EvaClock stacks blocks of: causal depthwise conv1d (local mixing) → **ClockMem** (recurrent multi-scale memory) → SwiGLU FFN. ClockMem's recurrence, per channel:

```
s_t = alpha ⊙ s_{t-1} + beta · (k_t ⊙ v_t)     (state update)
o_t = q_t ⊙ s_t ⊙ g_t                           (gated read)
alpha = sigmoid(log_clock)                       (per-channel learned decay, in (0,1))
```

Compute is O(S·D); stateful inference is O(D), independent of sequence length. Full derivation and the attention-vs-ClockMem head-to-head comparison (ClockMem wins by 2.2% bpb at seq=64 on prose, three seeds, non-overlapping) are in `README.md` and not repeated here.

> **CORRECTION (2026-08-19), on what "three seeds" means throughout this draft.** Until this date `EvaModel::new` initialized from a hard-coded constant (`Rng::new(0xE7A1)`); the `--seed` flag fed the training-window ordering and sample generation, never the initial parameters. Every run reported here as a distinct "seed" therefore **started from identical weights** and differed in the order the training windows were visited. This does not invalidate the reported numbers — three orderings producing non-overlapping distributions is still evidence, and the three trained models genuinely differ — but "three seeds" invites the reader to infer robustness to *initialization*, which was never tested. The claims should be read as "three data orderings". Initialization is now variable via `EVA_INIT_SEED` (unset reproduces `0xE7A1` exactly, verified against the historical loss trace), so the stronger claim is measurable; it has not been measured yet.

## 3. Finding 1: the clock flattens itself, and it isn't the objective that wants it to

### 3.1 The observation

After training (any checkpoint, any corpus size tested, any context length tested), 98% of channels settle into effective memory under 10 tokens, regardless of block. Only a handful of channels per block retain long memory. This contradicts the architecture's design promise ("each channel picks its own timescale").

### 3.2 Three ruled-out explanations

Before proposing a mechanism, three plausible causes were tested directly and each was rejected by measurement, not argument:

| candidate cause | test | result | alpha spectrum (fast / medium / slow) |
|---|---|---|---|
| no cross-window memory (`--persist` off) | train with `--persist` on vs off, same recipe/seed | **no effect** | 98/2/~0% both conditions; mean alpha 0.216 vs 0.217 |
| insufficient corpus | 250KB vs 411KB corpus, same recipe | **no effect** | 98/2/~0% both; mean alpha 0.216-0.217 both |
| insufficient context per step | seq ∈ {64, 128, 256, 512}, 1 epoch each (seq=512 gets 8x fewer gradient steps than seq=64: 439 vs 3515) | **no effect, despite the confound working against this being null** | 98/2/~0% at all four; mean alpha 0.216, std 0.249-0.250 at all four |

The seq-length result is notable for surviving its own confound: seq=512 trains with 8x fewer optimizer steps than seq=64, so if the collapse were sensitive to training regime at all, less training should move the spectrum *further* from convergence, not leave it identical. Identical results under an 8x step-count disparity is evidence *against* a training-regime explanation, not merely an absence of evidence for one.

**Verification:** numbers reproduced from `TRABAJO-2026-08-12.md` (persistence: "CLAUDIO: pregunta 1 de GPT respondida"; corpus: "achatado con 411 KB"; seq-length: "seq 64/128/256/512 terminó"). Read directly from the dated log by Claude during this drafting session, 2026-08-15.

### 3.3 Mechanism

With the ruled-out list narrowing the search to parametrization or objective, direct measurement of the real training gradient (not a proxy) settled it:

```
correlation(alpha, signed gradient of log_clock)      = -0.001   (no bias toward fast)
correlation(alpha, |gradient| of log_clock)            = +0.538   (fast channels get almost no signal)
```

Hand-perturbing individual channels (±0.5x / 2x alpha, no retraining) moves bpb negligibly channel-by-channel — the loss is close to indifferent to where the clock sits. The explanation needs no assumption that the objective "prefers" fast memory: `d(alpha)/d(log_clock) = alpha·(1-alpha)`, the sigmoid derivative. At alpha≈0.01 this factor is ~0.01; at alpha≈0.3 it is ~0.21 — roughly 20x more learning signal in the slow region purely from the shape of the squashing curve. Near alpha=0, the channel's own gradient-based learning path for `alpha` shuts itself off. This is a **saturation trap** in the parametrization, not a decision made by training.

### 3.4 Fix and reproducibility

One-line change: `alpha = sigmoid(z/T)` with T=1.3 instead of T=1 — same range, same meaning, only stretches where useful gradient lives. Composed from existing ops (`scale` + `sigmoid`), no new differentiable primitive needed.

| | fast (<10 tok) | medium (10-100) | slow (>100) | bpb |
|---|---|---|---|---|
| T=1 (baseline) | 98% | 2% | ~0% | 2.594 |
| T=1.3 | ~16% | ~54% | ~30% | **2.588** |

Reproduced bit-for-bit (identical loss trajectory, identical per-step trace) across three independent runs: same machine (GPU) twice under different concurrent load and thermal state, and a second machine (different CPU architecture, no GPU, different OS/glibc version), transferring only the pre-compiled binary.

**What this does not yet show:** that the distributed spectrum is useful, not merely free — on this corpus (250KB of general technical prose) it gives the same bpb, not better. Whether it pays off on data with genuine long-range dependencies is the subject of Section 4.

**Verification:** figures reproduced from `README.md` ("El reloj se achataba solo...") and `logs/traza_temp13.log` (committed trace artifact), cross-checked against three separate training log files by Claude, 2026-08-14/15.

## 4. Finding 2: storage exists, transport is where it dies

### 4.1 Setup

A synthetic benchmark isolates long-range recall cleanly: `y[t] = F(x[t-N])`, a fixed byte permutation F applied to the input N steps back, embedded in windows of `seq=512`. Evaluation is bpb/top-1 accuracy restricted to positions `t >= N` (the only positions where the label is determined by something outside the immediate local context; positions `t < N` are provably unpredictable and excluded from the reported number). Tested at N ∈ {8, 32, 128, 256}; this section focuses on N=256, the hardest case run to date.

At baseline (`out = q·cur·g`, the standard ClockMem read), N=256 is unsolved: 8.0039 bpb, indistinguishable from chance (chance floor ≈ 8.0 bpb over 256 byte classes).

### 4.2 The chain of interventions

| condition | readout | mask_eval bpb (t≥N) | top-1 acc |
|---|---|---|---|
| base | `out = q·cur·g` | 8.0039 | ~chance |
| B (learnable window, delta-init) | `out = q·read·g` | 8.0093 | — (w stayed ≈ delta, i.e. did not move) |
| B2.1 (structured finite-diff init + 10x lr on the window) | `out = q·read·g` | 8.0164 | — (still did not move) |
| **oracle** (exact recovery taps, recomputed from live alpha/beta each forward — no learned parameter) | `out = q·read·g` | 8.0021 | ~chance |
| **inject** (exact recovered write added directly to the residual, bypassing `q·g` entirely) | `out = q·cur·g + write_recovered` | **0.0353** | **99.6%** |
| inject, replication (independent run, same recipe/seed) | same | 0.0339 | 99.637% |
| R1a (learnable K×d window, seeded at the exact finite-difference solution) | `out = q·cur·g + Σ_j w_j·state_{t-j}` | 1.7254 | 43.4%→ actually measured 73.284% (see note) |
| R1b (same learnable window, zero-init) | same | 3.5431 | 44.050% |
| R1c-seed (structured family, 2 params/channel, seeded at the oracle solution) | `out = q·cur·g + inv_beta·(state_{t-N+1} - alpha_read·state_{t-N})` | 0.0355 | 99.641% |
| R1c-neutral (same structured family, zero-init — no seeding) | same | 0.0348 | 99.649% |
| R2-block (gate reintroduced with a floor, one scalar `s` per block: `s = floor + (1-floor)·sigmoid(z)`, `out = q·cur·g + s·read`) | as shown | 0.0359 | 99.609% |
| R2-block, cross-machine replication (independent run on a second machine, different hardware/OS, no GPU) | same | 0.0344 (train) | 99.639% (train) |
| R2-channel (same floored-gate construction, one independent `s` per channel — 512 gates instead of 1 per block) | `out = q·cur·g + s⊙read` | 0.0356 | 99.609% |

Note on R1a: the top-1 figure quoted verbally during the investigation (43.4%) and the figure in the corresponding eval log (73.284%) do not match; the table reports the logged figure. This discrepancy has not been root-caused and should be resolved before this table is finalized for external use — flagged here rather than silently reconciled.

### 4.3 What each step rules out

- **The state can store the information.** An independent finite-difference probe (`state_probe.rs`) recovers the write `(cur_t - alpha·cur_{t-1})/beta` from the raw recurrent state at 54-85% accuracy — even in models that never learned the task (loss ≈ chance). The value is present; it is not being written into the state incorrectly.
- **An exact readout exists and is representable.** The oracle condition recomputes the exact linear combination of state values needed to recover the write, every forward pass, from no learned parameter. It still fails end-to-end (8.0021 bpb) — so the failure is not "the right linear combination doesn't exist."
- **The multiplicative gate `q·read·g` is where the signal dies**, confirmed by layer-by-layer ridge probing (`activation_probe.rs`): on the exact-oracle model, `read` itself decodes the token at 38.6% (val), but `q·read·g` immediately after drops to 0.84%.
- **A free learnable route recovers *some* capability but not the full solution** (R1a/R1b, 1.7-3.5 bpb) — better than chance, far from the oracle's 0.035.
- **Restricting the parametrization to the true structural family (2 params/channel instead of K) closes the gap completely** (R1c, both seeded and from-scratch, ≈0.035 either way) — confirming the free window's failure was excess degrees of freedom, not insufficient gradient.
- **The gate can be safely reintroduced** if architecturally prevented from fully closing (R2's floor: `s` never reaches 0, so `memory_path`'s own parameters always receive real gradient regardless of gate state). R2-block matches the reference solutions (0.0359 vs 0.0353-0.0355) while keeping a genuine, live gate: traced gradient on the gate's own pre-activation `z` is nonzero throughout training (5e-3 → ~1e-4 in magnitude, never exactly zero) across all 1800 steps and all 5 blocks. This was independently replicated on a second machine (different hardware, no GPU, different OS), matching to within run-to-run noise.
- **The floored-gate mechanism holds under much greater spatial freedom, and the model does not use that freedom.** R2-channel (512 independent gates, one per channel, instead of 1 per block) matches R2-block (0.0356 vs 0.0359 bpb) — the "gate with a floor" property is not an artifact of having only 5 scalar gates to fit. Notably, given 512 independent degrees of freedom, the model did not differentiate them: final per-channel `s` clustered tightly (mean 0.534-0.536, min 0.521, max 0.555 across all 5 blocks; 0 of 512 channels closed near the floor, 0 fully open) — functionally equivalent to the block-level scalar solution. The gradient reaching each individual channel's `z[c]` is roughly 1000x smaller than the gradient reaching the block-level scalar `z` (each channel accumulates over one column instead of the whole block), which is the likely mechanical reason the extra freedom went unused rather than exploited.

### 4.4 An explicit distinction worth keeping separate

Two different guarantees are easy to conflate when talking about "keeping the gate alive":

1. Can the gate's own parameter learn to open further? (gradient toward `z`)
2. Does the memory route keep learning even while the gate stays mostly closed? (gradient toward the memory path's own internal parameters, independent of the gate)

R2's floor construction guarantees (2) unconditionally by design (the memory path is never given a literal zero weight). It does **not** guarantee (1) — any bounded sigmoid-shaped gate saturates at its extremes by construction, with vanishing self-gradient there, regardless of any floor applied to its output. In the R2 run measured, the gate did not saturate (settled at `s≈0.56`, an interior point) so this distinction did not bind in practice — but it should not be assumed to hold in general, and should be traced explicitly (`z`, `s`, and `s·read` logged separately) in any follow-up.

**Verification:** all bpb/accuracy figures in the table were independently re-run (not merely re-read from a report) by Claude against the actual checkpoints and/or eval logs in this repository during drafting, 2026-08-14/15: `inject`/`inject_r2` cross-checked against `logs/eval_N256_win257_inject*.log`; `r1a`/`r1b`/`r1c`/`r1c_neutral` against their respective `logs/eval_N256_win257_*.log`; R2 re-run live against `16m5b_N256_T1_win257_r2.weights` (`mask_eval` re-executed, not just read) and cross-checked against the raw `logs/train_N256_win257_r2.log` trace line-by-line for `z`/`s`/`mean|s*read|`.

### 4.5 Does the mechanism generalize across N?

R2-channel (the structurally more permissive variant, per 4.3) was re-run at N ∈ {32, 128, 256}, same recipe/corpus family/budget, only N (and the corresponding dataset seed) changed:

| N | window K=N+1 | R2-channel bpb (t≥N) | top-1 acc |
|---|---|---|---|
| 32 | 33 | 0.0244 | 99.787% |
| 128 | 129 | 0.0299 | 99.737% |
| 256 | 257 | 0.0356 | 99.609% |

All three re-run and independently confirmed by Claude against the checkpoints (`mask_eval` re-executed, not read from a report), 2026-08-15. The mechanism resolves at all three distances — there is no N in the tested range where it fails, unlike the baseline (`out=q·cur·g`), which sits at ≈8.0 bpb regardless of N. bpb rises mildly with N (0.024 → 0.030 → 0.036), consistent with the task getting harder, not with the mechanism degrading.

**Notable negative-ish finding: the gate does not adapt to N.** Final gate value `s` clusters at ≈0.53 in all three N conditions (block 0: 0.532/0.533/0.536 at N=32/128/256; block 4: 0.532/0.532/0.534) — essentially the same fixed compromise regardless of how much memory the task needs. What *does* adapt is the memory route itself (`alpha_read`/`inv_beta`), which sustains the same effective contribution `s·read` per block across all three N despite the fixed attenuation and the differing distance. Generalization across N is carried by the floor-plus-learnable-route design, not by the gate learning to open more for harder tasks.

### 4.6 Is the ~0.535 compromise a real optimum, or gradient starvation?

Tested directly, without retraining: take the trained N=256 R2-channel checkpoint (trained `z≈0.04`, `s≈0.535`) and hand-set every gate pre-activation `z` to a fixed value (no gradient, same hand-perturbation method used for `alpha` in Section 3), then re-measure masked bpb. This distinguishes two explanations for why training stopped there — a genuine optimum (moving away should hurt) versus a gradient-magnitude artifact (moving away should be neutral or better, but the gradient was too small to get there in the training budget). With `s = floor + (1-floor)·sigmoid(z)`, `floor=0.05`:

| forced `z` (all channels/blocks) | resulting `s` (exact) | bpb | top-1 acc |
|---|---|---|---|
| -3.0 (near floor) | 0.0973 | 7.1751 | 40.340% |
| -1.0 | 0.3116 | 0.4306 | 99.609% |
| 0.0 (near, not identical to, the trained point) | 0.5250 | 0.0359 | 99.609% |
| +1.0 | 0.7445 | **0.0347** | 99.609% |
| +3.0 | 0.9573 | 0.0355 | 99.609% |
| +6.0 (≈ fully open) | 0.9976 | 0.0357 | 99.609% |

(For reference, the actual trained point is `z≈0.04`, `s≈0.535` — close to, but not identical to, the `z=0.0` row above; the small `bpb` gap between the trained checkpoint's own reported figure, 0.0356, and this table's `z=0.0` row, 0.0359, is run-to-run noise from re-measuring rather than a real difference in `s`.)

The landscape is **asymmetric, and not flat on the open side** — the best point in this sweep is `z=+1` (`s≈0.74`, bpb 0.0347), not full-open (`z=+6` gives 0.0357, slightly worse than `z=+1`). Toward the floor, there is a real, sharp cliff — forcing the gate down to `s≈0.10` collapses accuracy to 40% and bpb to 7.18, close to the unsolved-baseline regime. Toward open, bpb stays flat-to-better across a broad plateau (`s` from ≈0.53 to ≈1.0, bpb between 0.0347 and 0.0359) with a shallow minimum around `s≈0.74` — opening the gate further than training ever reached does not hurt, and mildly helps. This favors the gradient-magnitude explanation over the genuine-optimum explanation: nothing in the loss landscape penalizes a more open gate in this range; training simply never got there, consistent with the ~1000x smaller per-channel gradient noted above. The one load-bearing property is staying clear of the floor-side cliff — which is exactly the property the floor was designed to guarantee, now confirmed from the opposite direction (perturbing a trained model) rather than only from the training trace.

**Verification:** new diagnostic (`examples/gate_perturb.rs`, read-only, no `src/` changes, no retraining), written and run by Claude, 2026-08-15, against `16m5b_N256_T1_win257_r2c.weights`. The baseline row (`z=0.0`, 0.0359) reproduces the trained-model figure from Section 4.3/4.5 (0.0356) to within run rounding, confirming the perturbation harness itself is measuring the same thing as `mask_eval.rs`.

**Independent confirmation by retraining from a different starting point.** The perturbation result predicts that training initialized closer to the plateau's shallow minimum should converge there and stay, rather than drifting back to ≈0.535. Tested directly: retrained N=256 R2-channel from `z0=1.0` (`s≈0.744` at init) instead of the default `z0=0.0`, same recipe/seed/floor otherwise. Result: final `z` settled at 1.02–1.04 across all 5 blocks (`s≈0.75`, 0/512 channels closed, 0/512 fully open) — no drift back toward the `z0=0` run's 0.535 — with bpb 0.0354 / 99.609% top-1, matching the reference solutions and marginally better than the `z0=0` run's 0.0356. This independently confirms the perturbation-based reading via a completely different method (retraining from a different init, not hand-editing a trained checkpoint): the gate's resting value is determined by where training starts within the plateau, not by a single attracting optimum. Verified by Claude against the checkpoint and raw training log (`logs/train_N256_win257_r2c_z0p1.log`), 2026-08-15.

### 4.7 First real-text test: does the temperature fix (Section 3) help on an actual book?

Section 3's fix is the one piece of this work that requires no fixed `N` and applies to any corpus unchanged (unlike Section 4's R2/structured-readout machinery, which is built around knowing the target lag `N` in advance — see the scoping note in Section 8). It was tested directly on a full novel: complete public-domain text of *Don Quijote* (2.2MB), `dim=512, ffn=1024, 5 blocks, seq=512, seed=7`, 1 epoch (3875 steps), `T=1` vs `T=1.3`, run in parallel on two different machines:

| condition | machine | final validation bpb (431 windows) |
|---|---|---|
| T=1 (baseline) | local (GTX 1650) | 1.716 |
| T=1.3 (fix) | second machine (AMD RX 580, first-ever training measurement on that GPU) | 1.720 |

**No benefit, and if anything marginally worse (noise, not signal).** Same qualitative outcome as the smaller technical-prose corpora in Section 3 — free, not (yet) useful.

**A new, unexpected observation worth flagging alongside the null result:** the *untreated* (`T=1`) spectrum on this corpus does not collapse as hard as it did everywhere else tested. At steady state (step 3500):

| | fast (<10 tok) | medium (10-100) | slow (>100) |
|---|---|---|---|
| T=1 on Quijote | ≈34% | ≈39% | ≈26% |
| T=1.3 on Quijote | ≈16% | ≈54% | ≈30% |
| T=1 on the earlier 250-411KB technical-prose corpora (Section 3.1) | 98% | 2% | ~0% |

A real, large (2.2MB), narratively structured corpus induces a partially-distributed spectrum on its own, without the temperature fix — something never observed on the smaller technical-document corpora used to establish and rule out the persistence/corpus-size/context-length explanations in Section 3.1. And yet neither this naturally-partial spectrum (`T=1`) nor the further-distributed one (`T=1.3`) translates into better bpb here. Two explanations are live and undistinguished: (a) `seq=512` may still be short relative to a novel's actual dependency distances (a name or plot thread returning hundreds of pages later is far more than 512 bytes away), or (b) one epoch on a 13M-parameter model may be too little training for any available long-range memory to get exploited, independent of whether the spectrum has room for it.

**Decision (not autopiloted):** this result does not call for immediately spending more compute in any particular direction — it calls for the team to look at it together before choosing between a longer `seq`, more epochs, or a more targeted real-text evaluation (e.g., measuring recall specifically at positions where a name/entity reappears after a long gap, rather than averaging bpb over the whole book) before running anything further. Checkpoints (`16m5b_quijote_T1.weights`, `16m5b_quijote_T13.weights`) and full logs are kept for that discussion.

**Verification:** both runs launched, monitored, and their final figures independently re-read from the raw training logs by Claude, 2026-08-15 (`logs/train_quijote_T1.log`; `train_quijote_T13.log` on the second machine, copied locally).

### 4.8 A directed benchmark on real text: entity retrieval, and the first non-null result for the temperature fix

Global bpb (Section 4.7) conflates two different questions — "does long-range memory help?" and "does a 13M-parameter model learn language at all?" — and cannot separate them. Following a design proposal, a directed alternative was built: `examples/entity_retrieval.rs` automatically detects proper names that recur through *Don Quijote* (43 pass a minimum-count filter; the most frequent: Quijote 2230, Sancho 2171, Dios 511, Panza 348, Dulcinea 283, Rocinante 204) and measures, at each occurrence, whether the model assigns higher probability to the entity's first byte given that it has seen that entity before, as a function of the distance back to the prior occurrence.

**Mode 1 — independent windows (as trained), distances capped at `seq=512`.** No retrieval signal appears beyond very short range: `Δlnp` (log-prob relative to a same-entity reference where the prior occurrence was outside the window and thus invisible to the model) is positive only for `d≤32` (T=1: +0.66; T=1.3: +0.82) and negative or flat everywhere from 32 to 512 bytes for both temperatures — seeing an entity earlier in the same window does not make the model more confident about it later in that window, once past very short range. What *does* help strongly is having seen the entity anywhere during training at all (`cold`, first-ever occurrence: mean_lnp ≈ 4.1–4.5) versus a window where it recurs but out of reach (`nowin` reference: mean_lnp ≈ 3.0–3.1) — that gap is lexical knowledge of the book, not in-context retrieval. T=1 and T=1.3 are statistically indistinguishable on this metric — a third consecutive null result for the temperature fix, now on a targeted, per-entity metric rather than global bpb.

**Mode 2 — state carried across windows (`forward_carrying`), reaching distances past 512 up to 1024+.** The model was trained with independent windows, so carried state is explicitly out-of-distribution; this mode is exploratory. Here a real, large, and — critically — *reproducible* asymmetry appeared: under `T=1`, carried state collapses the model's entity predictions (rank of the correct byte among 256 rises from ≈9–13 in-distribution to ≈25–62; the entity's own bpb rises from ≈0.7–2 to ≈17–21). Under `T=1.3`, the same carry does **not** collapse predictions (rank stays ≈8–12, matching in-distribution behavior).

Because the original `T=1` and `T=1.3` checkpoints had been trained on two different machines (Section 4.7), this asymmetry was initially confounded with hardware. It was resolved with a full 2×2 control — both temperatures retrained and evaluated on *both* machines independently (Claude ran the Martha side, Dante the Cristina side, without coordinating on intermediate numbers):

| | Cristina (GTX 1650) | Martha (RX 580) |
|---|---|---|
| **T=1**, carry, val, rank of correct byte (mean across distance buckets) | 25.4–61.8 (collapses) | 25.7–62.4 (collapses) |
| **T=1.3**, carry, val, rank of correct byte (mean across distance buckets) | 8.1–11.9 (stable) | 8.1–11.8 (stable) |

All four cells were trained and evaluated independently (not copied between machines), and the same-temperature pairs match closely across hardware while the cross-temperature difference is large and consistent in both. This rules out the hardware confound cleanly: **the collapse is a real, reproducible property of `T=1`, and its absence a real, reproducible property of `T=1.3`** — independent of which machine trained or evaluated the model.

**This is the first non-null result for the temperature fix on anything resembling real language**, after three null results (Section 3's own corpus, Section 4.7's global bpb, and this section's own in-distribution retrieval metric). The honest caveat is sitting right next to it: it only appears in an out-of-distribution regime (carried state) that the model was never trained for, so it is evidence that `T=1.3` makes the memory spectrum more *robust to distribution shift in how state is used*, not yet evidence that it improves in-distribution long-range language modeling. Whether that robustness itself is useful — e.g., whether it would matter if the model were trained on longer sequences or with persistent state to begin with — is untested.

**Verification:** entity counts, bucket sample sizes, and headline numbers independently re-derived by Claude by re-running `entity_retrieval.rs` from scratch (not reading Dante's reported figures) for the T=1/Cristina baseline and for both Martha-side conditions; the Cristina-side `T=1.3` retraining and its carry-mode numbers were produced independently by Dante and cross-checked by Claude against the raw log. All artifacts: `examples/entity_retrieval.rs`; checkpoints `16m5b_quijote_{T1,T13,T13_Cristina,T1_martha}.weights`; logs `logs/entity_retrieval_quijote_{T1,T13}_{indep,carry}.log`, `logs/entity_retrieval_quijote_T13_cristina_carry.log`, `logs/entity_retrieval_quijote_T1_martha_carry.log`.

### 4.9 Is carry-stability special to the temperature fix, or free with any slow recurrence?

Section 4.8's positive result raises an immediate question: is `T=1.3` doing something specific, or would *any* slow-decay recurrence be stable under carried state, making the finding generic rather than a property of this fix? Tested directly with a baseline built inside the same codebase (proposed as a cheaper alternative to porting an external SSM implementation, Section 6): `log_clock` replaced by a fixed, non-learned `alpha=0.999` on every channel of every block (`EVA_SSM_ALPHA`), removing `2560` decay parameters (`13,405,701 → 13,403,141`) and all gradient to them (confirmed in the training trace: `slow=512/512, mean_alpha=0.9990, mean_|grad|=0.000000` from step 500 onward) — the simplest possible instance of a slow-decay SSM, deliberately not tuned or learned at all.

| condition | global val bpb | independent-window retrieval | carry, rank at `d>1024` (val) |
|---|---|---|---|
| T=1 | 1.716 | null (Section 4.8) | 25.4–62.4 (collapses) |
| T=1.3 | 1.720 | null (Section 4.8) | 8.1–11.9 (stable, ~flat with distance) |
| SSM (`alpha=0.999` fixed) | 1.988 (worse — no fast channels left for local structure) | null, same pattern | 11.3–18.2 (does not collapse, but degrades with distance more than T=1.3) |

Two things resolve at once:

1. **Carry-stability is not exclusive to `T=1.3`.** The fixed-slow SSM baseline — which never learns anything about its decay rate — is also stable under carried state where `T=1` collapses. Slow decay itself, however it is arrived at (learned-and-compressed via temperature, or simply fixed), avoids the out-of-distribution collapse. `T=1`'s collapse was specific pathology of that particular parametrization (Section 3's saturation trap pushing nearly all channels to fast decay), not something `T=1.3` uniquely fixes — a fixed-slow baseline sidesteps the same trap trivially, by never being subject to it.
2. **Stability is still not usefulness, confirmed a second way.** In no condition — `T=1`, `T=1.3`, or the SSM baseline — does carried state at long distance ever beat that same model's own no-memory reference (the `d>1024` bucket never approaches, let alone exceeds, the `nowin`/`cold` baseline in any of the three). The SSM result rules out one route to usefulness (simply being more stable) without finding another.

This closes the most direct version of the SSM-comparison question raised in Section 6 (does a modern-style slow-decay SSM share ClockMem's failure to show long-range benefit on real text?) with a "yes" — on the cheap in-codebase baseline, not yet on an external state-of-the-art implementation. The open question this leaves, largely unaddressed by this draft's approach so far, is architectural rather than parametrization-level: what would make retained state actually informative for prediction, as opposed to merely available and stable.

**Verification:** checkpoint (`16m5b_quijote_SSM.weights`), global bpb, and carry-mode bucket table independently re-read by Claude against the raw logs (`logs/train_quijote_SSM.log`, `logs/entity_retrieval_quijote_SSM_{indep,carry}.log`) — all figures (1.988 bpb; carry val top1 14.60% and rank at `d>1024`) match Dante's reported numbers exactly.

### 4.10 Reframing the question: content-addressed recall, not positional recall

> **CORRECTION (2026-08-17).** The headline numbers in this section were produced at a training budget that later turned out to be *data-starved*, and its central speculation (a late "induction-head" phase transition) turned out to be wrong. Both mechanisms improve enormously with more unique data, and their ordering does not change but their magnitudes do: ClockMem goes 6.29% → **99.85%**, attention goes exact-chance → **~47%**. The reframing below (positional vs content-addressed recall) stands and is the reason the rest of the work happened; the measurements do not. **Section 4.11 supersedes this section's numbers and conclusions.** Kept as written because the sequence of what was believed, and on what evidence, is part of the record.

Section 4.9 leaves an architectural question open. Pursuing it exposed a flaw in the benchmark used through the rest of Section 4: `y[t]=F(x[t-N])` tests purely *positional* recall — the model always knows exactly how far back to look (a fixed, globally-shared `N`), and `F` is a single fixed permutation learnable once as a static weight, memorized across the whole training set rather than derived fresh per sequence. Real entity recall (Section 4.8) is not that: the same name can recur at any distance, and what it's "bound to" in context has to be extracted from that specific context, not looked up in a global table. This is the standard distinction in the literature between positional and *content-addressed* (associative) recall — the latter is the specific capability tested by the induction-head / MQAR family of benchmarks (Olsson et al. 2022; Fu et al. 2023 H3/Hyena; Arora et al. 2023 Based/MQAR), and linear-attention-family mechanisms (which ClockMem structurally belongs to: `q_t · state_t` with a decayed running sum) are documented in that literature to lag full attention specifically on this capability.

A new benchmark (`examples/generate_associative.rs`, `eval_associative.rs`) was built to test it directly: byte sequences containing (KEY, VALUE) pairs from disjoint byte ranges, with bindings established *fresh within each window* (not a global fixed table). A key's first occurrence is an unpredictable binding; every later occurrence of that same key is a query, and the byte immediately after it (the ordinary next-byte target) is only predictable if the model actually recalls that specific key's binding — at a gap that varies at random, never known in advance.

**First attempt (8 active keys, mean gap ≈119 bytes) was null for both mechanisms tested.** Both ClockMem and the attention baseline (Section 2's architecture, full causal, same parameter budget) scored at exact chance (≈3.1%, 32-symbol value alphabet) in every distance bucket — uninformative: the task was too hard for the compute budget of a single overnight run to distinguish anything.

**Simplifying to 2 active keys, small corpus (300 windows), initially produced a false positive.** Training-set filler bpb collapsed to 0.0136 (near-zero, versus a ≈7.585-bit chance floor) — the model had memorized the entire corpus verbatim rather than learning anything general; the mild elevation in key-value accuracy was an overfitting artifact, not signal.

**Same easy task (2 keys), corpus enlarged to 1500 windows to make outright memorization costly, produced the first genuine content-addressed recall signal of this draft:**

| bucket (gap since the key's previous occurrence) | ClockMem, val acc | ClockMem, val bpb | Attention, val acc | Attention, val bpb |
|---|---|---|---|---|
| ≤32 | **6.29%** | 4.45 | 4.55 | 5.21 |
| 32–128 | **4.76%** | 4.75 | 3.24 | 5.25 |
| 128–256 | 2.11% | 5.07 | 7.37 (n=95, noisy) | 5.13 |
| binding (cold, no-memory reference) | 3.33% | 5.31 | 3.33 | 5.25 |
| filler (structural sanity check) | bpb 7.55 (≈ true chance 7.585 — no memorization artifact this run) | | bpb 7.55 | |
| **TRAIN, all key/query buckets** | 24–30% (elevated, some memorization present, expected with more exposure) | | **3.11–3.43% (exact chance, including on TRAIN)** | |

ClockMem shows a real, distance-decaying signal on held-out data (6.29% → 4.76% → chance by 128–256 bytes), with the filler-bpb sanity check confirming it is not another memorization artifact. **Attention, on the identical task, scored at exact chance everywhere — including on the training set**, meaning it had not learned the key→value association at all, not even by memorizing specific training examples. This is counter to the naive expectation that full attention should trivially dominate a linear-recurrent mechanism at content-addressed recall (Section 4.10's own framing, and the explicit design rationale in `src/model/attn.rs`). The attention implementation was checked and is not structurally limited (full causal softmax over the whole sequence, not a restricted window) — so this is not an implementation artifact, at least not one identified so far.

**The most likely honest explanation, not yet confirmed:** induction-head-style circuits are documented in the literature as sometimes forming through a comparatively abrupt phase transition during training, after a threshold amount of exposure, rather than gradually (Olsson et al. 2022). At only ≈2700 training steps, attention may simply not have crossed that threshold yet, while ClockMem's continuously-adjustable state may not depend on the same discrete transition. This has not been tested (e.g. by training attention substantially longer on the same task) and should not be read as "ClockMem beats attention at associative recall" in general — it is one data point, one scale, one training budget, run once each.

**Verification:** both `generate_associative.rs` and `eval_associative.rs` are new, `src/`-untouched examples; entity self-check (0 failures recomputing bindings from raw bytes alone) passed on every corpus generated. All bpb/accuracy figures in the table read directly from the eval tool's output by Claude, who also wrote and ran both the generator and evaluator and launched all four training runs (ClockMem and attention, at both difficulty settings) across both machines. Checkpoints: `16m5b_assoc_{clock,attn}.weights` (8-key, null), `16m5b_assoc_k2_clock.weights` (2-key, small, overfit), `16m5b_assoc_k2_big_{clock,attn}.weights` (2-key, large, the reported result). Logs: `logs/train_assoc_*.log`, eval outputs in the scratch directory referenced in `TRABAJO-2026-08-15.md`'s Claude section for this date.

### 4.11 With enough unique data, ClockMem solves content-addressed recall — and then collapses as the number of keys grows

Section 4.10's runs used 1500 windows (2698 optimizer steps). Repeating the identical setup on a corpus 15x larger (22,499 windows, 20,249 steps, still only **one** epoch, so each window is seen once rather than fifteen times) changes the picture completely. Nothing else differs: same architecture, same parameter count, same seed, same recipe, same gap statistics.

| bucket (gap since key's previous occurrence) | ClockMem, val acc | ClockMem, val bpb | Attention, val acc | Attention, val bpb |
|---|---|---|---|---|
| ≤32 | **99.85%** | 0.026 | 46.05% | 3.767 |
| 32–128 | **99.69%** | 0.037 | 47.99% | 3.674 |
| 128–256 | **97.71%** | 0.126 | 50.29% | 3.682 |
| binding (first occurrence; no-memory reference) | 2.83% | 5.02 | 3.67% | 5.66 |
| filler (memorization sanity check) | bpb 7.53 | | bpb 7.55 | (true chance 7.585 — neither run memorized) |

Three things resolve at once:

1. **The ceiling in Section 4.10 was data, not architecture.** ClockMem's 6.29% → 99.85% comes purely from more unique training data at the same model size. Both no-leakage controls hold: first occurrences stay at chance (the model is not seeing the answer some other way) and filler bpb stays at chance (no verbatim memorization).
2. **Section 4.10's induction-head speculation was wrong.** Attention did not need a late phase transition; it needed more distinct data. It went from *exact chance including on the training set* to ~47%. This also retroactively explains the catastrophic divergence of the 15-epoch attention run (train loss falling while val loss climbed to 17.8 bpb): that was overfitting to a small corpus, not instability — a single epoch over varied data trains cleanly.
3. **A prior design claim in this codebase is experimentally refuted.** The header comment of `src/model/attn.rs` asserts that attention "can do associative retrieval — key A looks up value B" and that ClockMem "can't". At 2 keys, ClockMem does, at 99.85%, while attention reaches only ~47%.

**On attention's ~47%: retrieval without binding, now measured directly.** With exactly 2 active keys per window there are exactly 2 candidate values in scope, so a model that has learned *which values are in play* but not *which key binds to which* scores 50% by construction. Attention's numbers (46.05 / 47.99 / 50.29) sit on that value and are flat across distance. To distinguish this from generic error, the evaluator was extended to classify what the top-1 prediction actually *is* at every query position: this key's bound value, **another key's** bound value, a value byte bound to nothing in that window, or not a value byte at all. Run on all three large-corpus checkpoints:

| top-1 prediction is… | ClockMem, 2 keys | ClockMem, 8 keys | Attention, 2 keys |
|---|---|---|---|
| this key's bound value (correct) | **99.65%** | 23.42% | 47.23% |
| **another key's bound value** | **0.00%** | **76.49%** | **42.45%** |
| a value byte bound to nothing here | 0.35% | 0.08% | 10.17% |
| not a value byte at all | 0.00% | 0.00% | 0.15% |
| *(within-candidate-set total)* | *99.65%* | *99.91%* | *89.68%* |
| n (query positions scored) | 5,967 | 25,610 | 5,967 |

This is a clean dissociation between two distinct failure modes, and it settles the interpretation:

- **ClockMem at 2 keys binds.** It never once confused the two keys in 5,967 queries — a literal 0.00%.
- **Attention retrieves but does not bind.** 89.68% of its predictions are one of the two values actually bound in that window (so it locates the candidate set nearly perfectly, from anywhere in the window), but it splits them near-evenly (47.23 vs 42.45) — a coin flip between correct candidates.
- **ClockMem at 8 keys degrades into attention's failure mode.** It knows the candidate set even better than attention does (99.91% of predictions are one of the 8 bound values; it has not forgotten anything, nor lost the structural rule that a value follows a key) yet picks the wrong key's value 76.49% of the time. Its residual binding is real but weak: 23.42% correct against 12.5% for guessing uniformly among the 8 candidates it has correctly identified.

The last row is what makes the channel-partitioning account hard to escape. The 8-key failure is not degraded memory, not undertrained structure, and not confusion about what the candidates are — **it is specifically a binding failure**, appearing exactly when the number of keys grows past what a partition of 512 channels can hold apart, and it is the same failure mode that attention exhibits from the start.

**Scaling with the number of keys: a sharp collapse.** The 2-key result admits a degenerate solution that would not scale — with `dim=512` and 2 keys, a model could dedicate ~256 channels to each key's value and never bind anything, which ClockMem's element-wise write (`cur[c] += β·k[t,c]·v[t,c]`, no cross-channel term) permits. Tested at 8 keys, same large-data regime, gap held constant by shortening the filler proportionally (mean gap 40.4 vs 45.2; queries/bindings ratio 10.7 vs 9.9):

| | 2 keys | 8 keys |
|---|---|---|
| ≤32 | 99.85% | **31.80%** |
| 32–128 | 99.69% | 14.03% |
| 128–256 | 97.71% | 5.65% (chance = 3.125%) |
| binding (control) | 2.83% | 3.29% |
| **retention (32–128 ÷ ≤32)** | **99.8%** | **44.1%** |

**The shape changes, not just the level.** At 2 keys, memory is nearly immune to distance — it retains 99.8% of its accuracy one bucket out and is still at 97.7% at 256 bytes. At 8 keys it retains 44% and is at chance by 128 bytes. That is the signature predicted by the channel-partitioning account (fewer channels per key → a more fragile trace), not simply a harder task uniformly depressing performance.

**Caveat, stated plainly: the 8-key run never plateaued.** Its validation loss was still falling at the final step (6.699 → 6.614 → 6.563 → 6.525 → 6.493 → 6.476 → 6.461 → 6.446), whereas the 2-key run reached its final value by ~35% of training (7.236 vs 7.226 final). So 31.80% is a **floor**, not a ceiling. Against that: the same checkpoint evaluated at 25% of training scored 25.25%, so quadrupling the training budget bought +6.5 points — a rate at which reaching 99% would require an implausible budget. That is a two-point extrapolation, not a demonstration, which is why a convergence run (4 epochs, 80,996 steps, trained until validation flattens) and a matched 8-key attention run are in progress as of this writing.

The prediction-breakdown above sharpens what that convergence run can and cannot change. Whatever accuracy more training buys, the 8-key model already demonstrates that it has *not* failed to learn the task's structure or to retain the candidates: it identifies the correct candidate set 99.91% of the time. The open quantity is narrower than "does it learn the task" — it is whether the 76.49% cross-key confusion shrinks with training, or whether that is where a channel-partitioned representation runs out of room.

**Convergence result (partial, run stopped deliberately at 2x).** The convergence run was halted at step 41,500 of 80,996 — a little over double the original training budget — because its second data point already determines the answer within the precision the question requires:

| training budget | correct | cross-key confusion |
|---|---|---|
| 1x (20,249 steps) | 23.42% | 76.49% |
| 2x (41,500 steps) | 28.71% | **71.24%** |
| 4x (80,996 steps, not run) | — | ~66% (extrapolated) |

Doubling the entire training budget bought 5.25 points of confusion reduction. At that rate, reaching the 2-key model's 0.35% confusion requires roughly 13 further doublings — on the order of 10⁴ times the compute — and even reaching 50% requires four more doublings. The cross-key confusion is therefore not on a trajectory toward resolution; it is decaying logarithmically in compute, which for any feasible budget is indistinguishable from a ceiling. The run was stopped rather than completed because the remaining 39,500 steps would have produced a number already predictable to within a point or two, on a thermally-limited machine, without changing the design conclusion. This is recorded as a deliberate methodological choice, not an interrupted experiment: the third point is extrapolated and labelled as such rather than measured.

**Attention at 8 keys collapses differently, and worse.** Completing the 2×2 (matched data, matched distance, one epoch each):

| | ClockMem, 2 keys | ClockMem, 8 keys | Attention, 2 keys | Attention, 8 keys |
|---|---|---|---|---|
| correct | 99.65% | 23.42% | 47.23% | **2.89%** (chance = 3.125%) |
| another key's bound value | 0.00% | 76.49% | 42.45% | 16.74% |
| value bound to nothing here | 0.35% | 0.08% | 10.17% | **80.23%** |
| **within candidate set** | 99.65% | 99.91% | 89.68% | **19.63%** |

Attention at 8 keys is at chance and — unlike ClockMem — has also lost the ability to identify the candidate set at all (19.63%, down from 89.68% at 2 keys); 80% of its predictions are value bytes drawn from the global alphabet rather than from the window's bindings. This is not undertraining: its validation loss moved only 0.028 bpb across the entire run (6.781 → 6.753), i.e. it converged almost immediately to a solution that ignores the bindings, whereas ClockMem's 8-key loss moved 0.253 bpb and was still descending. **Caveat, and it is a large one:** this attention baseline is deliberately minimal — single-head, learned absolute positions, described in its own source header as existing "to have something to measure against". The literature reports that transformers do learn associative recall (this is the induction-head setting). These results therefore support "ClockMem substantially outperforms this reference baseline at matched budget", not "attention cannot do this".

A secondary observation, also from a data-starved regime and therefore weak: a gap-controlled 2/3/4-key sweep on the small corpus shows near-range accuracy *rising* with key count (6.29% → 13.22% → 27.92%), which tracks query density (29.8k → 47.5k → 61.5k training queries) rather than capacity, and is exactly why that sweep cannot answer the capacity question. Its one non-density-explained pattern is retention, which falls monotonically as keys increase (75.7% → 53.2% → 30.3%), matching the 8-key large-data result (44.1%).

**Verification:** all figures re-read by Claude directly from the eval tool's output on the actual checkpoints; every corpus passed the generator's self-check (bindings recomputed from raw bytes alone, 0 failures). The evaluation tool gained an optional third argument capping windows per section — scoring 22.5k windows in full takes over an hour, while 300 val windows already yield 13,986 query positions in the shortest bucket alone. Checkpoints: `16m5b_assoc_k2_huge_{clock,attn}.weights`, `16m5b_assoc_k8ctl_huge_clock.weights`, `16m5b_assoc_k{3,4}ctl_clock.weights`. Logs: `logs/train_assoc_*.log`.

## 5. Cross-cutting observation: solutions as families, not points **[OPEN — flagged, not yet measured as a phenomenon]**

Twice in Section 4, the model converged to something other than the "obvious" canonical solution while matching its performance:

- **R1c-neutral** settles with `alpha_read ≈ 0` (not matching the true physical `alpha`) and `inv_beta → 1.043` — i.e., a full-strength tap at exactly the right lag, rather than an inversion of the leaky filter — and still reaches 99.6% accuracy. The downstream head absorbs what the leaky read leaves behind.
- **R2** settles at gate value `s ≈ 0.56`, an interior compromise, rather than opening fully (`s → 1`) even though nothing in the loss should penalize a fully open gate once the memory path is useful.

Both observations are consistent with a single reading: when a family of parametrizations achieves equivalent loss, gradient descent finds *some* member of that family, not a distinguished or "intended" one. This has not yet been measured as a general phenomenon — it has been observed twice, in the same model, on the same benchmark. Before treating it as a property of the architecture (or of gradient descent generally), it should be measured directly: how large is the family of near-equal-loss solutions, and which internal parameters vary freely within it versus which are pinned. Not scoped or scheduled as of this draft.

## 6. Related work (honest positioning)

None of the individual mechanisms here are unprecedented in isolation, and this section says so directly rather than implying novelty where there isn't any:

- The read/write/retrieve/hold/forget vocabulary and the general idea of a differentiable external memory closely parallel Neural Turing Machines (Graves et al., 2014) and Differentiable Neural Computers (Graves et al., 2016).
- A learned scalar gate that can attenuate but never fully close a residual path is the same idea as Highway Networks (Srivastava et al., 2015).
- Per-channel learned decay in a linear recurrence is a known family (gated linear SSMs / gated linear attention variants); the specific finding here is not that such decay exists, but that a specific, common squashing choice (plain sigmoid) creates a self-reinforcing saturation trap in the decay parameter's own gradient, and that this generalizes (in kind, not necessarily magnitude) to the separate multiplicative output gate `q·g` found in Section 4.
- **Close precedent, found after most of Section 4 was already run (2026-08-15):** "Retention Is Not Enough: Resolution and Optimization in State Space Long-Range Memory" (Research Square preprint, posted 2026-06-09, not peer-reviewed at time of writing). Studying Mamba on delayed-copy, it reports that fixing the decay rate to a slow constant produces near-perfect long-range retrieval, and frames the mechanism as one of *resolution* (an inverted-U over the decay rate — both too fast and too slow fail) rather than raw retention capacity, with a further distinction between the decay parameter itself and the effective timestep trajectory. This is a genuinely close precedent for the *spirit* of Section 3 (decay-rate parametrization determines whether long-range recall is learnable, independent of how much the state could in principle retain). We have not been able to obtain the full text (access blocked as of this writing; summary here is reconstructed from indexed abstract text only, not a full read, and should be re-verified before being cited more strongly than this). Based on what is available, it does not appear to address the mechanism in Section 4 of this draft: a *downstream* multiplicative output gate destroying an already-correct, already-recoverable read, independent of the decay/resolution question, nor the structured-vs-free readout distinction (R1a/R1b vs R1c) or the floored-gate construction (R2). This should be treated as provisional until the full paper can be read.

What we believe is not simply restated from prior work: the two-mechanism causal chain (parametrization trap in the decay, independently a second parametrization trap in the output gate, in the *same* trained model, each diagnosed by direct intervention rather than inferred), and the empirical demonstration that a structurally-restricted hypothesis class (R1c) succeeds where an unrestricted one with strictly more expressive power (R1a/R1b) fails on the identical task. Given the precedent above, the honest framing of the open question is narrower than "does long-range recurrent memory work" — it is closer to: *can a recurrent memory achieve retention, positional resolution, and learnable gating simultaneously, without the gating route reintroducing the same class of trap that afflicts the decay parametrization on its own?* A direct head-to-head against a modern SSM baseline (same delayed-copy task, matched N sweep, comparing bpb, FLOPs, state size, and positional-recovery accuracy) is the natural next step before any claim of contribution is finalized — not yet run.

## 7. Limitations

- Single model scale (13M parameters), single architecture family (ClockMem specifically — the saturation-trap mechanism is plausible for other sigmoid-gated recurrences but untested on any of them).
- The Section 4 long-range results (storage, oracle, gate-destroys-signal, structured readout, floored gate) are synthetic-only. R2's current implementation requires knowing the target lag `N` in advance, which real text does not provide — it does not transfer to real corpora without new design work (Section 8), and none of Section 4's fixes have been tested on real text as of this draft.
- Several individual results in Section 4 (R1c, R1a/R1b) rest on a single training run each at the time of writing. `inject` has been replicated once (0.0353 → 0.0339) and R2-block has been replicated cross-machine (0.0359 local vs 0.0344 on a second, GPU-less machine) — both same order of magnitude, not yet more than one replication each.
- The Section 3 temperature fix shows no measured *in-distribution benefit* on any corpus tested so far, technical-prose or real literary text (Section 4.7), nor on a targeted real-text retrieval metric in-distribution (Section 4.8, mode 1) — same performance, not better, in all three. Its one measured benefit (Section 4.8, mode 2) is specifically in an out-of-distribution regime (carried state) the model was never trained for; whether that generalizes to anything the model is actually used for is open.
- The entity-retrieval benchmark (Section 4.8) covers one book, one language, one set of 43 auto-detected proper nouns, and one model scale — not yet a general instrument.

## 8. Next steps **[OPEN, in progress as of this draft]**

- ~~Structural generality check: does the R2 floored-gate mechanism hold with a per-channel gate (`R2-channel`) as well as the current per-block scalar gate (`R2-block`)?~~ **Done, holds** (Section 4.3).
- ~~Does the same mechanism generalize across N (32/128/256)?~~ **Done, holds at all three** (Section 4.5) — but the gate itself does not adapt to N; only the memory route does.
- ~~Is the gate's resting value a real optimum or gradient starvation?~~ **Done, gradient starvation** — confirmed two independent ways: hand-perturbation of a trained checkpoint (Section 4.6) and retraining from a different gate init (`z0=1.0`, converges to and stays at a different, marginally better point) — both point to the same asymmetric plateau-plus-cliff landscape.
- ~~Does the Section 3 temperature fix help on real text?~~ **Tested three ways: global bpb (no), in-distribution targeted retrieval (no), out-of-distribution carried state (yes, and reproduced in a full 2×2 cross-machine control)** (Sections 4.7–4.8). The one positive result found so far is specifically about robustness under distribution shift, not in-distribution long-range modeling.
- ~~Is carry-stability specific to the temperature fix, or generic to any slow-decay recurrence?~~ **Done: generic** (Section 4.9) — a fixed, non-learned slow-decay baseline shares the same stability, and shares the same "stable but not useful" ceiling (never beats the no-memory reference at long range, in any of the three conditions tested).
- The open question is no longer about this fix specifically. It is now: **what would make retained state actually informative for prediction**, as opposed to merely present and stable? Not scoped — this is an architectural question, not a parametrization tweak, and per Section 4.9's closing note is the natural next thing to pursue if this line continues.
- Whether the gate's floor/temperature/init (Section 4) should themselves be learned or tuned — still lower priority; Section 4's mechanism remains synthetic-only (see Limitations) and untested for real-text transfer.
- Whether the Section 5 "solution family, not point" observation is a general property worth measuring directly.
- **In flight (Section 4.11):** an 8-key convergence run (4 epochs, 80,996 steps, trained until validation flattens) to rule out undertraining as the explanation for the 8-key collapse; and a matched 8-key attention run, to complete the 2×2 of {ClockMem, attention} × {2 keys, 8 keys}. If ClockMem plateaus near 30% while attention holds near its own 2-key level (~47%), the two mechanisms are failing for different reasons — attention by not binding at all, ClockMem by running out of channels to partition.
- **Conditional on that outcome:** if the channel-partitioning account survives, the indicated change is to the *write* rather than the read — a low-rank outer-product write (project `k` and `v` to a small rank `r`, keep an `r×r` state per block, write `k v^T` instead of `k ⊙ v`), which is the minimal modification that gives the state a place to store bindings at all. Every intervention in Sections 4.2–4.6 targeted the read path; none could have fixed a binding that was destroyed at write time.
- **Generalizing Section 4's structured/floored-gate mechanism to real text**, where the target lag `N` is not known in advance — candidate directions (untried): a bank of parallel routes at several candidate `N` values, or making `N` (or an analogous per-channel notion of "how far back to look") itself learnable rather than a fixed hyperparameter. Not started.
- ~~A direct head-to-head against an SSM baseline~~ **Done, cheap in-codebase version** (Section 4.9: fixed slow `alpha`, not an external state-of-the-art implementation like Mamba). Whether a real external baseline would tell a different story (e.g. via selective/input-dependent gating, which this fixed-alpha baseline deliberately lacks) is untested.

## Acknowledgments

This work is the product of a four-way collaboration with no single author: Valentín (project owner, decisions, hardware), Dante (OpenCode, implementation lead for most experiments in Section 4), GPT (hypothesis generation and experimental design review, no direct repository access), and Claude (implementation, cross-verification of all reported figures against raw logs, this draft). See `OVERVIEW.md` for the informal description of how this process works.
