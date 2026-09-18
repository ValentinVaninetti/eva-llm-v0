use crate::nn::{param, Linear, Module};
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// Is the flattening of `alpha` sigmoid
/// saturation (measured: |grad| 20-100x smaller where alpha≈0), not a
/// preference of the loss? With this on, ClockMem uses
/// `ops::algebraic_sigmoid` (polynomial tail) instead of `ops::sigmoid`
/// (exponential tail), with everything else identical -- same
/// architecture, same `alpha` range, same `EVA_ALPHA_MAX`, the only thing
/// that changes is how much gradient signal survives near the extremes.
fn antisat() -> bool {
    std::env::var("EVA_ALPHA_ANTISAT").is_ok()
}

/// Second intervention, after finding that `algebraic_sigmoid` was not
/// touching the right region: temperature on
/// THE SAME sigmoid, `alpha=sigmoid(z/T)` with T>1. Unlike
/// `algebraic_sigmoid`, this does NOT change the shape of the curve -- it
/// stretches the same sigmoid, so a given `z` (the same one any channel
/// already had) ends up with MORE derivative, verified with a script before
/// touching the code: at z=-3..-5 (where the fast band lives today), T=1.3
/// gives ~1.4x-2.4x more derivative than T=1. I chose T=1.3 explicitly for
/// that reason, not as an arbitrary round number.
///
/// Explicit trade-off, not hidden: for "only T changes" to be literally true
/// in the code (same `log_clock` values at startup, zero change to
/// initialization), the INITIAL `alpha` shifts a bit (less extreme -- with
/// T=1.3, alpha=0.018 at init becomes ≈0.044). There's no way to have
/// EXACTLY the same initial alpha AND more derivative there at the same time
/// with a single reparametrization family (it's the same reason
/// `algebraic_sigmoid` with an adjusted inverse ended up giving LESS signal
/// to the fast band, not more). The choice here is to preserve `z` (what the
/// optimizer actually sees), not `alpha`.
fn temperature() -> f32 {
    std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0)
}

/// SSM BASELINE (2026-08-15): `EVA_SSM_ALPHA=a` replaces
/// the LEARNED per-channel clock with a FIXED slow alpha for every channel.
/// Everything else is identical to base ClockMem -- same q/k/v/g, same beta,
/// same recurrence, same readout `q*cur*g`, same windowed training, same
/// carry eval. This is the proposed baseline ("a fixed slow alpha inside
/// our own codebase"), to answer the control: is the carry
/// stability of T=1.3 a property of ANY slow-decay recurrence, or of the
/// LEARNED alpha distribution specifically? Default 0.999: the code's own
/// documented "slow but safe" value (~1000-token memory, state magnitude
/// stays within ~1.25x of what training windows see, vs ~3x for 0.9999).
fn ssm_alpha() -> Option<f32> {
    std::env::var("EVA_SSM_ALPHA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|&a| a > 0.0 && a < 1.0)
}

/// ERROR-GATED WRITE (probe): `EVA_WRITE_ERROR=1` makes the write magnitude
/// position-dependent inside the SAME forward (see `ops::clockmem_gated_from`):
/// each position writes `beta * r_t` instead of `beta`, where `r_t` shrinks
/// the write when the content being written is already in the state. No new
/// parameters; `EVA_WRITE_ERROR_P` sets the gate exponent (default 1.0, 0
/// disables the gate) and `EVA_WRITE_ERROR_TRACE=1` records `r_t` for the
/// measurement. Returning `None` keeps the plain write.
fn write_error_gate() -> Option<(f32, bool)> {
    if !std::env::var("EVA_WRITE_ERROR").is_ok() {
        return None;
    }
    let p = std::env::var("EVA_WRITE_ERROR_P")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    let trace = std::env::var("EVA_WRITE_ERROR_TRACE").is_ok();
    Some((p, trace))
}

/// LOW-RANK OUTER-PRODUCT WRITE (task #58): `EVA_WRITE_LOWRANK=r` adds the
/// rank-`r` matrix-state write of `ops::clockmem_lowrank_from` alongside the
/// element-wise one. Unset (or 0) keeps the plain write, so every existing
/// checkpoint and every run without the flag behaves exactly as before.
///
/// This is a WRITE experiment and is not meant to combine with the read
/// experiments (R1/R1c/R2/oracle/inject), which answer a different question;
/// if a read flag is also set, the read flag wins, matching how the other
/// variants already dispatch.
fn lowrank_rank() -> Option<usize> {
    std::env::var("EVA_WRITE_LOWRANK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&r| r > 0)
}

/// GATE ABLATION (2026-08-19). `EVA_NO_GATE=1` replaces the
/// multiplicative output gate with a constant 1, so the readout becomes
/// `out = q*cur` instead of `out = q*cur*g`.
///
/// WHY. The `query_swap` diagnostic showed several trained models answer the
/// same byte no matter WHICH key they are asked about (directional shift
/// +0.0001 against +0.9941 for a model that binds). Section 4 of the draft
/// already documented this exact gate destroying a read that was verifiably
/// correct upstream of it, and the fix that worked there (R1/inject) was to
/// route around it. So it is the suspect with a record, and it had never
/// been ablated outright.
///
/// The `wg` projection stays in the parameter list and keeps its weights;
/// it simply stops receiving gradient. That is deliberate: it keeps the
/// parameter count IDENTICAL to the baseline, so an A/B differs in the
/// mechanism and not in model size.
fn no_gate() -> bool {
    std::env::var("EVA_NO_GATE").is_ok()
}

/// PER-CHANNEL READ ABLATION (2026-08-20).
/// `EVA_READ_MASK=c1,c2,...` cancels the READ of those channels by setting
/// `g[c] = 0`, without touching the write or the state dynamics.
///
/// WHY LIKE THIS. `out[t,c] = q[t,c]*cur[c]*g[t,c]`, and `g` is computed out
/// here rather than inside the op, so zeroing it is an exact ablation of that
/// channel's read with no new op and no change to the backward pass. The state
/// keeps evolving the same way: the only thing cut is what that channel
/// contributes to the output.
///
/// WHAT FOR. A model solves associative recall at 99.77% and we do not know
/// how. Two hypotheses with different signatures when channels are turned off:
///   - partitioned by key  -> turning off a small, specific group sinks ONE
///     key and leaves the others almost intact
///   - distributed / hashed representation -> the drop is gradual and even,
///     with no magic group
/// Inference only: it does not affect training, and without the variable the
/// behaviour is identical to always.
fn read_mask() -> Option<Vec<usize>> {
    let v = std::env::var("EVA_READ_MASK").ok()?;
    let idx: Vec<usize> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    if idx.is_empty() { None } else { Some(idx) }
}

/// Applies the mask to `g` (which already comes in shape [S, D]).
fn apply_read_mask(g: Tensor) -> Tensor {
    match read_mask() {
        None => g,
        Some(idx) => {
            let (s, d) = (g.shape[0], g.shape[1]);
            let mut data = g.data.to_vec();
            for t in 0..s {
                for &c in &idx {
                    if c < d { data[t * d + c] = 0.0; }
                }
            }
            // Constant on purpose: this is an analysis intervention, not a
            // gradient route. Never used during training.
            Tensor::new(data, g.shape.clone())
        }
    }
}

fn alpha_squash(z: &Tensor) -> Tensor {
    if antisat() {
        ops::algebraic_sigmoid(z)
    } else {
        let t = temperature();
        if t != 1.0 { ops::sigmoid(&ops::scale(z, 1.0 / t)) } else { ops::sigmoid(z) }
    }
}

/// Inverse of `alpha_squash`, only to initialize `log_clock` pointing at the
/// same target `alpha` regardless of which squashing is active -- otherwise
/// changing the squashing would also change the initial range and it
/// wouldn't be a single variable between the two conditions anymore.
fn alpha_squash_inv(a: f32) -> f32 {
    if antisat() {
        // s(z)=0.5(1+z/sqrt(1+z^2))  =>  z = (2a-1) / (2*sqrt(a(1-a)))
        (2.0 * a - 1.0) / (2.0 * (a * (1.0 - a)).sqrt())
    } else {
        (a / (1.0 - a)).ln()
    }
}

pub struct ClockMem {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wg: Linear,
    pub log_clock: Tensor,
    pub beta: Tensor,
    pub wread: Option<Tensor>,
    pub alpha_read: Option<Tensor>,
    pub inv_beta: Option<Tensor>,
    pub gate_z: Option<Tensor>,
    /// Low-rank outer-product write: the four rank-r projections
    /// and the per-slot clock of the matrix state. All Some together or all
    /// None; created only when EVA_WRITE_LOWRANK=r.
    pub pk: Option<Tensor>,
    pub pv: Option<Tensor>,
    pub pq: Option<Tensor>,
    pub po: Option<Tensor>,
    pub log_clock_m: Option<Tensor>,
    /// SSM baseline: when Some(a), the per-channel clock is REPLACED by this
    /// fixed alpha (log_clock is then excluded from parameters). See
    /// `ssm_alpha()`.
    pub ssm_alpha: Option<f32>,
}

impl ClockMem {
    pub fn new(d: usize, rng: &mut Rng) -> Self {
        let log_clock = {
                // CLOCK CEILING. With 0.9999 a channel forgets so slowly
                // that it accumulates ~10,000 terms: harmless within a
                // 64-token window, but with persistent state across the
                // whole corpus it SATURATES -- measured, the state's
                // magnitude reached 881. With 0.999 the effective memory is
                // ~1000 tokens, fifteen times the window, which is exactly
                // what we want without blowing up.
                let a_max = std::env::var("EVA_ALPHA_MAX")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.9999);
                let a_min = 0.01f32;
                let logits: Vec<f32> = (0..d)
                    .map(|i| {
                        let t = if d == 1 { 1.0 } else { i as f32 / (d - 1) as f32 };
                        let a = a_max * (a_min / a_max).powf(t);
                        alpha_squash_inv(a)
                    })
                    .collect();
                param(logits, vec![d])
            };
        let beta = param(vec![1.0], vec![1]);
        // Windowed clock read.
        // EVA_READ_WIN=K turns the read into a learned window sum over the
        let df_n = std::env::var("EVA_READ_DF_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        // ORACLE READOUT (after B2.1): EVA_READ_ORACLE=1 with
        // EVA_READ_WIN=K and EVA_READ_DF_N=N replaces the learned wread with
        // the MATHEMATICALLY EXACT inverse, recomputed from the current
        // alpha/beta on every forward (so it stays exact even as alpha is
        // learned). No wread parameter exists, so nothing is trained; the
        // question is whether the recoverable write is enough to solve the
        // masked task end-to-end. With no oracle, the B2.1 learned init applies.
        let oracle_on = std::env::var("EVA_READ_ORACLE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            != 0;
        // CLEAN INJECTION (EVA_READ_INJECT=1): the recovered write goes to
        // the residual stream directly, without q*g (see ops::clockmem_inject).
        let inject_on = std::env::var("EVA_READ_INJECT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            != 0;
        // R1 (2026-08-15): the windowed read WITHOUT the q*g
        // gate, with a LEARNED wread (ops::clockmem_readwin_inj). Two inits:
        //   - R1a: EVA_READ_DF_N=N seeds the finite-difference taps
        //     (w[N-1]=1/beta, w[N]=-alpha/beta) -- the read starts functional
        //     and the question is whether the gradient now moves it.
        //   - R1b: no EVA_READ_DF_N -> all-zero init -- the model must
        //     discover the long read by itself through the additive path.
        let r1_on = std::env::var("EVA_READ_R1")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            != 0;
        // R1c (2026-08-15): the STRUCTURED learnable
        // inverse. Replaces the free Kxd wread of R1a/R1b with ~2
        let r1c_on = std::env::var("EVA_READ_R1C")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            != 0;
        let r1c_neutral = std::env::var("EVA_READ_R1C_NEUTRAL").is_ok();
        // R2 (2026-08-15): the structured memory path
        // (same family as R1c) behind a gated scalar that CANNOT annul it.
        //   out = q*cur*g + s*read,  s = floor + (1-floor)*sigmoid(z)
        // The floor (EVA_READ_R2_FLOOR, default 0.05) guarantees s >= floor > 0
        // even if the gate saturates closed, so memory_path keeps contributing
        // and alpha_read/inv_beta keep receiving REAL gradient (property (2)).
        // z (EVA_READ_R2_Z0, default 0.0) is a scalar pre-activation per block;
        // its own gradient is NOT required to work at saturation (property (1)).
        let r2_on = std::env::var("EVA_READ_R2")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            != 0;
        let r2_z0 = std::env::var("EVA_READ_R2_Z0")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0);
        // R2-channel (2026-08-15): same gate with one z PER CHANNEL
        // (shape [D]) instead of one scalar per block. the control for "is
        // the block scalar imposing the inductive bias". The scalar family is
        // a subset (z constant == block), so the floor guarantee holds per
        // channel. The distribution of s[c] is traced (EVA_WREAD_TRACE).
        let r2_channel = std::env::var("EVA_READ_R2_CHANNEL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            != 0;
        let (alpha_read, inv_beta) = if r1c_on || r2_on {
            let alpha_v = alpha_squash(&Tensor::new(log_clock.data.to_vec(), vec![d]));
            let ar_data: Vec<f32> = if r1c_neutral {
                vec![0.0; d]
            } else {
                alpha_v.data.to_vec()
            };
            let ib_data = vec![1.0 / beta.data[0]; d];
            (Some(param(ar_data, vec![d])), Some(param(ib_data, vec![d])))
        } else {
            (None, None)
        };
        let gate_z = if r2_on {
            if r2_channel {
                Some(param(vec![r2_z0; d], vec![d]))
            } else {
                Some(param(vec![r2_z0], vec![1]))
            }
        } else {
            None
        };
        let wread = if r1c_on || r2_on || oracle_on || (inject_on && !r1_on) {
            None
        } else {
            std::env::var("EVA_READ_WIN")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&k| k > 0)
                .map(|k| {
                    let mut data = vec![0.0f32; k * d];
                    if !r1_on {
                        for c in 0..d {
                            data[c] = 1.0;
                        }
                    }
                    if let Some(n) = df_n {
                        let beta_v = beta.data[0];
                        // Detached copy: with a requires_grad=false input,
                        // finalize returns a plain tensor, no graph node leaks.
                        let alpha = alpha_squash(&Tensor::new(log_clock.data.to_vec(), vec![d]));
                        if n >= 1 && n - 1 < k {
                            for c in 0..d {
                                data[(n - 1) * d + c] = 1.0 / beta_v;
                            }
                        }
                        if n < k {
                            for c in 0..d {
                                data[n * d + c] = -alpha.data[c] / beta_v;
                            }
                        }
                    }
                    param(data, vec![k, d])
                })
        };
        // The low-rank write's parameters.
        //
        let (pk, pv, pq, po, log_clock_m) = match lowrank_rank() {
            None => (None, None, None, None, None),
            Some(r) => {
                let bound = 1.0 / (d as f32).sqrt();
                let mk = |rng: &mut Rng| {
                    param((0..d * r).map(|_| rng.uniform(-bound, bound)).collect(), vec![d, r])
                };
                let pk = mk(rng);
                let pv = mk(rng);
                let pq = mk(rng);
                let po_data = if std::env::var("EVA_WRITE_LOWRANK_PO_RAND").is_ok() {
                    let b = 1.0 / (r as f32).sqrt();
                    (0..r * d).map(|_| rng.uniform(-b, b)).collect()
                } else {
                    vec![0.0f32; r * d]
                };
                // The matrix state exists to hold BINDINGS across the gaps of
                // the associative benchmark (mean ~40 bytes, max ~400 in a
                // 512-byte window). The main clock's init spreads alpha
                // geometrically down to 0.01, where a channel forgets in a
                // single step -- fine for D=512 channels, but with only r
                // slots it would spend half of them on decay rates that can
                // hold no binding at all. So the same geometric spread is
                // used over a deliberately slower range. This is a design
                // choice, not a measured optimum: both ends are flags.
                let am_max = std::env::var("EVA_LOWRANK_AM_MAX")
                    .ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.9999);
                let am_min = std::env::var("EVA_LOWRANK_AM_MIN")
                    .ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.9);
                let logits: Vec<f32> = (0..r)
                    .map(|i| {
                        let t = if r == 1 { 0.0 } else { i as f32 / (r - 1) as f32 };
                        alpha_squash_inv(am_max * (am_min / am_max).powf(t))
                    })
                    .collect();
                (Some(pk), Some(pv), Some(pq), Some(param(po_data, vec![r, d])),
                 Some(param(logits, vec![r])))
            }
        };
        ClockMem {
            wq: Linear::new(d, d, rng),
            wk: Linear::new(d, d, rng),
            wv: Linear::new(d, d, rng),
            wg: Linear::new(d, d, rng),
            log_clock,
            beta,
            wread,
            alpha_read,
            inv_beta,
            gate_z,
            pk,
            pv,
            pq,
            po,
            log_clock_m,
            ssm_alpha: ssm_alpha(),
        }
    }
}

impl ClockMem {
    /// Physical clock alpha (alpha_squash of log_clock) as a PLAIN tensor
    /// (detached copy, no graph node): the reference point for comparing
    /// the learned alpha_read against, and for the df taps.
    pub(crate) fn alpha_physical(&self) -> Tensor {
        if let Some(a) = self.ssm_alpha {
            Tensor::new(vec![a; self.log_clock.shape[0]], vec![self.log_clock.shape[0]])
        } else {
            alpha_squash(&Tensor::new(self.log_clock.data.to_vec(), vec![self.log_clock.shape[0]]))
        }
    }

    /// ORACLE READOUT: the exact inverse, recomputed from the CURRENT
    /// alpha/beta every forward so the recovered write always matches the
    /// real state dynamics even as alpha is learned. Returns None unless
    /// EVA_READ_ORACLE=1 with EVA_READ_WIN=K and EVA_READ_DF_N=N set.
    /// w[0,c]=1 (delta, keeps the seed region behaving like base),
    /// w[N-1,c]=1/beta, w[N,c]=-alpha_c/beta -> read[t]=write[t-N+1].
    pub(crate) fn oracle_read(&self) -> Option<Tensor> {
        if std::env::var("EVA_READ_ORACLE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            == 0
        {
            return None;
        }
        let k = std::env::var("EVA_READ_WIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())?;
        let n = std::env::var("EVA_READ_DF_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())?;
        if k == 0 || n == 0 {
            return None;
        }
        let d = self.log_clock.shape[0];
        let beta_v = self.beta.data[0];
        let alpha = alpha_squash(&Tensor::new(self.log_clock.data.to_vec(), vec![d]));
        let mut data = vec![0.0f32; k * d];
        for c in 0..d {
            data[c] = 1.0;
        }
        if n >= 1 && n - 1 < k {
            for c in 0..d {
                data[(n - 1) * d + c] = 1.0 / beta_v;
            }
        }
        if n < k {
            for c in 0..d {
                data[n * d + c] = -alpha.data[c] / beta_v;
            }
        }
        Some(Tensor::new(data, vec![k, d]))
    }

    /// Clean-injection mode: returns N when EVA_READ_INJECT=1, else None.
    fn inject_n(&self) -> Option<usize> {
        if std::env::var("EVA_READ_INJECT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            == 0
        {
            return None;
        }
        std::env::var("EVA_READ_DF_N").ok().and_then(|v| v.parse::<usize>().ok())
    }

    /// R1 mode: the learned windowed read without the q*g gate. Returns the
    /// wread parameter when EVA_READ_R1=1 (requires EVA_READ_WIN=K).
    fn r1_read(&self) -> Option<&Tensor> {
        if std::env::var("EVA_READ_R1")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            == 0
        {
            return None;
        }
        self.wread.as_ref()
    }

    /// R1c mode: the structured learnable inverse. Returns (alpha_read,
    /// inv_beta, N) when EVA_READ_R1C=1 with EVA_READ_DF_N=N set. Requires
    /// the two per-channel parameters created in `new`.
    fn r1c_read(&self) -> Option<(&Tensor, &Tensor, usize)> {
        if std::env::var("EVA_READ_R1C")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            == 0
        {
            return None;
        }
        let n = std::env::var("EVA_READ_DF_N").ok().and_then(|v| v.parse::<usize>().ok())?;
        Some((self.alpha_read.as_ref()?, self.inv_beta.as_ref()?, n))
    }

    /// R2 mode: the gated structured read. Returns (alpha_read, inv_beta, z,
    /// floor, N) when EVA_READ_R2=1 with EVA_READ_DF_N=N set.
    fn r2_read(&self) -> Option<(&Tensor, &Tensor, &Tensor, f32, usize)> {
        if std::env::var("EVA_READ_R2")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            == 0
        {
            return None;
        }
        let n = std::env::var("EVA_READ_DF_N").ok().and_then(|v| v.parse::<usize>().ok())?;
        let floor = std::env::var("EVA_READ_R2_FLOOR")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.05);
        Some((self.alpha_read.as_ref()?, self.inv_beta.as_ref()?, self.gate_z.as_ref()?, floor, n))
    }

    /// The low-rank write's five tensors, all-or-nothing.
    fn lowrank_parts(&self) -> Option<(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor)> {
        Some((
            self.pk.as_ref()?,
            self.pv.as_ref()?,
            self.pq.as_ref()?,
            self.po.as_ref()?,
            self.log_clock_m.as_ref()?,
        ))
    }

    /// R2 dispatch: z [1] -> scalar block gate, z [D] -> per-channel gate.
    /// The channel family contains the scalar one (z constant == block).
    fn r2_call(&self, q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, alpha: &Tensor, s0: &[f32]) -> (Tensor, Vec<f32>) {
        let (ar, ib, z, floor, nn) = self.r2_read().expect("r2_call without R2");
        if z.shape == vec![q.shape[1]] {
            ops::clockmem_inject_learn_gc(q, k, v, g, alpha, &self.beta, ar, ib, z, floor, nn, s0)
        } else {
            ops::clockmem_inject_learn_g(q, k, v, g, alpha, &self.beta, ar, ib, z, floor, nn, s0)
        }
    }

    /// Same as `forward`, starting from the state the previous window left
    /// behind and returning the one left for the next.
    pub fn forward_from(&self, x: &Tensor, s0: &[f32]) -> (Tensor, Vec<f32>) {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let g = if no_gate() {
            Tensor::new(vec![1.0; x.shape[0] * x.shape[1]], x.shape.clone())
        } else {
            ops::sigmoid(&self.wg.forward(x))
        };
        let g = apply_read_mask(g);
        let alpha = match self.ssm_alpha {
            Some(a) => Tensor::new(vec![a; q.shape[1]], vec![q.shape[1]]),
            None => alpha_squash(&self.log_clock),
        };
        match self.r2_read() {
            Some(_) => self.r2_call(&q, &k, &v, &g, &alpha, s0),
            None => match self.r1c_read() {
                Some((ar, ib, nn)) => {
                    ops::clockmem_inject_learn(&q, &k, &v, &g, &alpha, &self.beta, ar, ib, nn, s0)
                }
                None => match self.r1_read() {
                    Some(w) => ops::clockmem_readwin_inj(&q, &k, &v, &g, &alpha, &self.beta, w, s0),
                    None => match self.inject_n() {
                        Some(n) => ops::clockmem_inject(&q, &k, &v, &g, &alpha, &self.beta, n, s0),
                        None => match self.oracle_read() {
                            Some(w) => ops::clockmem_readwin(&q, &k, &v, &g, &alpha, &self.beta, &w, s0),
                            None => match &self.wread {
                                Some(w) => ops::clockmem_readwin(&q, &k, &v, &g, &alpha, &self.beta, w, s0),
                                None => match self.lowrank_parts() {
                                    Some((pk, pv, pq, po, lcm)) => {
                                        let am = alpha_squash(lcm);
                                        ops::clockmem_lowrank_from(
                                            &q, &k, &v, &g, &alpha, &self.beta, pk, pv, pq, po, &am, s0,
                                        )
                                    }
                                    None => match write_error_gate() {
                                        Some((p, trace)) => {
                                            ops::clockmem_gated_from(&q, &k, &v, &g, &alpha, &self.beta, p, s0, trace)
                                        }
                                        None => ops::clockmem_from(&q, &k, &v, &g, &alpha, &self.beta, s0),
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
    }
}

impl Module for ClockMem {
    fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let g = if no_gate() {
            Tensor::new(vec![1.0; x.shape[0] * x.shape[1]], x.shape.clone())
        } else {
            ops::sigmoid(&self.wg.forward(x))
        };
        let g = apply_read_mask(g);
        let alpha = match self.ssm_alpha {
            Some(a) => Tensor::new(vec![a; q.shape[1]], vec![q.shape[1]]),
            None => alpha_squash(&self.log_clock),
        };
        let s0 = vec![0.0; q.shape[1]];
        match self.r2_read() {
            Some(_) => self.r2_call(&q, &k, &v, &g, &alpha, &s0).0,
            None => match self.r1c_read() {
                Some((ar, ib, nn)) => ops::clockmem_inject_learn(&q, &k, &v, &g, &alpha, &self.beta, ar, ib, nn, &s0).0,
                None => match self.r1_read() {
                    Some(w) => ops::clockmem_readwin_inj(&q, &k, &v, &g, &alpha, &self.beta, w, &s0).0,
                    None => match self.inject_n() {
                        Some(n) => ops::clockmem_inject(&q, &k, &v, &g, &alpha, &self.beta, n, &s0).0,
                        None => match self.oracle_read() {
                            Some(w) => ops::clockmem_readwin(&q, &k, &v, &g, &alpha, &self.beta, &w, &s0).0,
                            None => match &self.wread {
                                Some(w) => ops::clockmem_readwin(&q, &k, &v, &g, &alpha, &self.beta, w, &s0).0,
                                None => match self.lowrank_parts() {
                                    Some((pk, pv, pq, po, lcm)) => {
                                        let am = alpha_squash(lcm);
                                        ops::clockmem_lowrank_from(
                                            &q, &k, &v, &g, &alpha, &self.beta, pk, pv, pq, po, &am, &s0,
                                        )
                                        .0
                                    }
                                    None => match write_error_gate() {
                                        Some((p, trace)) => {
                                            ops::clockmem_gated_from(&q, &k, &v, &g, &alpha, &self.beta, p, &s0, trace).0
                                        }
                                        None => ops::clockmem(&q, &k, &v, &g, &alpha, &self.beta),
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut out = Vec::new();
        out.extend(self.wq.parameters());
        out.extend(self.wk.parameters());
        out.extend(self.wv.parameters());
        out.extend(self.wg.parameters());
        if self.ssm_alpha.is_none() { out.push(&self.log_clock); }
        out.push(&self.beta);
        if let Some(w) = &self.wread { out.push(w); }
        if let Some(a) = &self.alpha_read { out.push(a); }
        if let Some(b) = &self.inv_beta { out.push(b); }
        if let Some(z) = &self.gate_z { out.push(z); }
        if let Some(t) = &self.pk { out.push(t); }
        if let Some(t) = &self.pv { out.push(t); }
        if let Some(t) = &self.pq { out.push(t); }
        if let Some(t) = &self.po { out.push(t); }
        if let Some(t) = &self.log_clock_m { out.push(t); }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out = Vec::new();
        out.extend(self.wq.parameters_mut());
        out.extend(self.wk.parameters_mut());
        out.extend(self.wv.parameters_mut());
        out.extend(self.wg.parameters_mut());
        if self.ssm_alpha.is_none() { out.push(&mut self.log_clock); }
        out.push(&mut self.beta);
        if let Some(w) = &mut self.wread { out.push(w); }
        if let Some(a) = &mut self.alpha_read { out.push(a); }
        if let Some(b) = &mut self.inv_beta { out.push(b); }
        if let Some(z) = &mut self.gate_z { out.push(z); }
        if let Some(t) = &mut self.pk { out.push(t); }
        if let Some(t) = &mut self.pv { out.push(t); }
        if let Some(t) = &mut self.pq { out.push(t); }
        if let Some(t) = &mut self.po { out.push(t); }
        if let Some(t) = &mut self.log_clock_m { out.push(t); }
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut out = Vec::new();
        out.extend(self.wq.named_parameters(&format!("{}.wq", prefix)));
        out.extend(self.wk.named_parameters(&format!("{}.wk", prefix)));
        out.extend(self.wv.named_parameters(&format!("{}.wv", prefix)));
        out.extend(self.wg.named_parameters(&format!("{}.wg", prefix)));
        if self.ssm_alpha.is_none() { out.push((format!("{}.log_clock", prefix), &self.log_clock)); }
        out.push((format!("{}.beta", prefix), &self.beta));
        if let Some(w) = &self.wread { out.push((format!("{}.wread", prefix), w)); }
        if let Some(a) = &self.alpha_read { out.push((format!("{}.alpha_read", prefix), a)); }
        if let Some(b) = &self.inv_beta { out.push((format!("{}.inv_beta", prefix), b)); }
        if let Some(z) = &self.gate_z { out.push((format!("{}.gate_z", prefix), z)); }
        if let Some(t) = &self.pk { out.push((format!("{}.pk", prefix), t)); }
        if let Some(t) = &self.pv { out.push((format!("{}.pv", prefix), t)); }
        if let Some(t) = &self.pq { out.push((format!("{}.pq", prefix), t)); }
        if let Some(t) = &self.po { out.push((format!("{}.po", prefix), t)); }
        if let Some(t) = &self.log_clock_m { out.push((format!("{}.log_clock_m", prefix), t)); }
        out
    }

    fn named_parameters_mut(&mut self, prefix: &str) -> Vec<(String, &mut Tensor)> {
        let mut out = Vec::new();
        out.extend(self.wq.named_parameters_mut(&format!("{}.wq", prefix)));
        out.extend(self.wk.named_parameters_mut(&format!("{}.wk", prefix)));
        out.extend(self.wv.named_parameters_mut(&format!("{}.wv", prefix)));
        out.extend(self.wg.named_parameters_mut(&format!("{}.wg", prefix)));
        if self.ssm_alpha.is_none() { out.push((format!("{}.log_clock", prefix), &mut self.log_clock)); }
        out.push((format!("{}.beta", prefix), &mut self.beta));
        if let Some(w) = &mut self.wread { out.push((format!("{}.wread", prefix), w)); }
        if let Some(a) = &mut self.alpha_read { out.push((format!("{}.alpha_read", prefix), a)); }
        if let Some(b) = &mut self.inv_beta { out.push((format!("{}.inv_beta", prefix), b)); }
        if let Some(z) = &mut self.gate_z { out.push((format!("{}.gate_z", prefix), z)); }
        if let Some(t) = &mut self.pk { out.push((format!("{}.pk", prefix), t)); }
        if let Some(t) = &mut self.pv { out.push((format!("{}.pv", prefix), t)); }
        if let Some(t) = &mut self.pq { out.push((format!("{}.pq", prefix), t)); }
        if let Some(t) = &mut self.po { out.push((format!("{}.po", prefix), t)); }
        if let Some(t) = &mut self.log_clock_m { out.push((format!("{}.log_clock_m", prefix), t)); }
        out
    }
}
