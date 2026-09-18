use crate::tensor::autograd::make_node;
use crate::tensor::{next_id, Tensor};

fn finalize(
    inputs: &[&Tensor],
    op: &'static str,
    data: Vec<f32>,
    shape: Vec<usize>,
    saved_v: Vec<std::sync::Arc<Vec<f32>>>,
    saved_f: Vec<f32>,
    saved_u: Vec<usize>,
) -> Tensor {
    let needs = inputs.iter().any(|t| t.requires_grad);
    let id = next_id();
    if needs {
        let grad_flags: Vec<bool> = inputs.iter().map(|t| t.requires_grad).collect();
        let node = make_node(op, inputs, grad_flags, saved_v, saved_f, saved_u, id, data.len());
        Tensor { data: std::sync::Arc::new(data), shape, id, requires_grad: true, node: Some(node) }
    } else {
        Tensor { data: std::sync::Arc::new(data), shape, id, requires_grad: false, node: None }
    }
}

// Per-position (e_t, r_t) pairs recorded by `clockmem_gated_from` (when
// traced), in call order. Thread-local because eval and the measurement
// probe run on a single thread and the record has to match the call order of
// the blocks; a global would interleave blocks.
thread_local! {
    static WRITE_TRACE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub fn write_trace_reset() {
    WRITE_TRACE.with(|t| t.borrow_mut().clear());
}

pub fn write_trace_take() -> Vec<f32> {
    WRITE_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let nd = a.len().max(b.len());
    let mut out = vec![1; nd];
    for i in 0..nd {
        let da = if i < nd - a.len() { 1 } else { a[i - (nd - a.len())] };
        let db = if i < nd - b.len() { 1 } else { b[i - (nd - b.len())] };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            panic!("shapes not broadcastable: {:?} vs {:?}", a, b);
        }
    }
    out
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let out_shape = broadcast_shape(&a.shape, &b.shape);
    let out = add_arrays(&a.data, &a.shape, &b.data, &b.shape, &out_shape);
    let mut su = vec![a.shape.len()];
    su.extend_from_slice(&a.shape);
    su.push(b.shape.len());
    su.extend_from_slice(&b.shape);
    finalize(&[a, b], "add", out, out_shape, vec![a.data.clone(), b.data.clone()], vec![], su)
}

fn add_arrays(a: &[f32], ash: &[usize], b: &[f32], bsh: &[usize], osh: &[usize]) -> Vec<f32> {
    if ash == osh && bsh == osh {
        return a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    }
    let n = osh.iter().product::<usize>();
    let mut out = vec![0.0; n];
    for oidx in 0..n {
        let ia = br_index(oidx, ash, osh);
        let ib = br_index(oidx, bsh, osh);
        out[oidx] = a[ia] + b[ib];
    }
    out
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let out_shape = broadcast_shape(&a.shape, &b.shape);
    let out = mul_arrays(&a.data, &a.shape, &b.data, &b.shape, &out_shape);
    let mut su = vec![a.shape.len()];
    su.extend_from_slice(&a.shape);
    su.push(b.shape.len());
    su.extend_from_slice(&b.shape);
    finalize(&[a, b], "mul", out, out_shape, vec![a.data.clone(), b.data.clone()], vec![], su)
}

fn mul_arrays(a: &[f32], ash: &[usize], b: &[f32], bsh: &[usize], osh: &[usize]) -> Vec<f32> {
    if ash == osh && bsh == osh {
        return a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    }
    let n = osh.iter().product::<usize>();
    let mut out = vec![0.0; n];
    for oidx in 0..n {
        let ia = br_index(oidx, ash, osh);
        let ib = br_index(oidx, bsh, osh);
        out[oidx] = a[ia] * b[ib];
    }
    out
}

fn br_index(oidx: usize, xshape: &[usize], oshape: &[usize]) -> usize {
    if xshape == oshape {
        return oidx;
    }
    let off = oshape.len() - xshape.len();
    let mut idx = 0;
    let mut rem = oidx;
    let mut pow = 1;
    for od in (0..oshape.len()).rev() {
        let oc = rem % oshape[od];
        rem /= oshape[od];
        if od >= off {
            let xd = od - off;
            let xc = if xshape[xd] == 1 { 0 } else { oc };
            idx += xc * pow;
            pow *= xshape[xd];
        }
    }
    idx
}

pub fn scale(a: &Tensor, s: f32) -> Tensor {
    let out = a.data.iter().map(|&x| x * s).collect();
    finalize(&[a], "scale", out, a.shape.clone(), vec![], vec![s], vec![])
}

pub fn sum_all(x: &Tensor) -> Tensor {
    let sum = x.data.iter().sum();
    finalize(&[x], "sum_all", vec![sum], vec![], vec![], vec![], vec![x.numel()])
}

pub fn neg(a: &Tensor) -> Tensor {
    scale(a, -1.0)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    add(a, &scale(b, -1.0))
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 2, "matmul expects 2D tensors");
    assert_eq!(b.shape.len(), 2, "matmul expects 2D tensors");
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul inner dims mismatch: {:?} x {:?}", a.shape, b.shape);
    let mut out = vec![0.0; m * n];
    crate::math::matmul(&a.data, &b.data, m, k, n, &mut out);
    finalize(&[a, b], "matmul", out, vec![m, n], vec![a.data.clone(), b.data.clone()], vec![], vec![m, k, n])
}

pub fn transpose(x: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 2, "transpose expects a 2D tensor");
    let (r, c) = (x.shape[0], x.shape[1]);
    let out = crate::math::transpose(&x.data, r, c);
    finalize(&[x], "transpose", out, vec![c, r], vec![], vec![], vec![r, c])
}

/// Row-wise softmax over the causal prefix: row `i` only sees columns
/// `0..=i`, and the rest stays at zero.
///
/// The masking happens IN HERE and not as a `-inf` added beforehand, for two
/// reasons. Numerical: `exp(-inf)` at the edge gives NaN as soon as anyone
/// subtracts the max. And cost: masking outside would force materializing an
/// SxS matrix of `-inf` per layer and per step, which is memory and traffic
/// just to represent "don't look here".
pub fn softmax_causal(x: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 2, "softmax_causal expects (S,S)");
    let (s, n) = (x.shape[0], x.shape[1]);
    assert_eq!(s, n, "softmax_causal expects a square score matrix");
    let mut out = vec![0.0; s * s];
    for i in 0..s {
        let row = &x.data[i * s..i * s + i + 1];
        // Subtract the max before exponentiating: without this, large
        // logits overflow to inf and the whole row comes out NaN.
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for j in 0..=i {
            let e = (x.data[i * s + j] - mx).exp();
            out[i * s + j] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..=i {
            out[i * s + j] *= inv;
        }
    }
    finalize(&[x], "softmax_causal", out.clone(), x.shape.clone(), vec![std::sync::Arc::new(out)], vec![], vec![s])
}

/// The first `n` rows of a (S,D) tensor.
///
/// Needed because during generation the window grows one token at a time and
/// the position table has a fixed length. Slicing it with `Tensor::new`
/// looked equivalent and isn't: a hand-built tensor has no node, so the
/// gradient never comes back and the table learns nothing -- it fails
/// silently, training a dead parameter.
pub fn slice_rows(x: &Tensor, n: usize) -> Tensor {
    assert_eq!(x.shape.len(), 2, "slice_rows expects (S,D)");
    let (s, d) = (x.shape[0], x.shape[1]);
    assert!(n <= s, "slice_rows: {n} rows out of a tensor of {s}");
    finalize(&[x], "slice_rows", x.data[..n * d].to_vec(), vec![n, d], vec![], vec![], vec![s, d, n])
}

pub fn silu(x: &Tensor) -> Tensor {
    let out: Vec<f32> = x.data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    finalize(&[x], "silu", out, x.shape.clone(), vec![x.data.clone()], vec![], vec![])
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    let out: Vec<f32> = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    finalize(&[x], "sigmoid", out.clone(), x.shape.clone(), vec![std::sync::Arc::new(out)], vec![], vec![])
}

/// Squashing R->(0,1) with a POLYNOMIAL tail instead of an exponential one:
/// the derivative decays as 1/|z|^3 (`sigmoid` decays as exp(-|z|)) -- much
/// more signal survives far from the center. Exists to test whether the
/// flattening of `alpha` in ClockMem is sigmoid
/// saturation (measured: |grad| 20-100x smaller in the fast band) and not a
/// preference of the loss, this op should let gradient keep arriving even
/// with alpha near 0.
///
/// s(z) = 0.5*(1 + z/sqrt(1+z^2))  -- same domain/range as sigmoid.
pub fn algebraic_sigmoid(x: &Tensor) -> Tensor {
    let out: Vec<f32> = x.data.iter().map(|&v| 0.5 * (1.0 + v / (1.0 + v * v).sqrt())).collect();
    finalize(&[x], "algebraic_sigmoid", out, x.shape.clone(), vec![x.data.clone()], vec![], vec![])
}

pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.shape.len(), 2, "rms_norm expects 2D input (S,D)");
    let (s, d) = (x.shape[0], x.shape[1]);
    let mut out = vec![0.0; s * d];
    for i in 0..s {
        let mut sq = 0.0;
        for j in 0..d {
            let v = x.data[i * d + j];
            sq += v * v;
        }
        let inv = 1.0 / (sq / d as f32 + eps).sqrt();
        for j in 0..d {
            out[i * d + j] = x.data[i * d + j] * inv * w.data[j];
        }
    }
    finalize(&[x, w], "rms_norm", out, x.shape.clone(), vec![x.data.clone(), w.data.clone()], vec![eps], vec![s, d])
}

pub fn gather(emb: &Tensor, idx: &[usize]) -> Tensor {
    let (v, d) = (emb.shape[0], emb.shape[1]);
    let mut out = vec![0.0; idx.len() * d];
    for (t, &ix) in idx.iter().enumerate() {
        out[t * d..(t + 1) * d].copy_from_slice(&emb.data[ix * d..(ix + 1) * d]);
    }
    let mut su = vec![v, d];
    su.extend_from_slice(idx);
    finalize(&[emb], "gather", out, vec![idx.len(), d], vec![], vec![], su)
}

pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> Tensor {
    let (s, v) = (logits.shape[0], logits.shape[1]);
    let mut loss = 0.0;
    for i in 0..s {
        let row = &logits.data[i * v..(i + 1) * v];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for &x in row {
            sum += (x - m).exp();
        }
        let lsm = (row[targets[i]] - m) - sum.ln();
        loss += -lsm;
    }
    loss /= s as f32;
    finalize(&[logits], "ce", vec![loss], vec![], vec![logits.data.clone()], vec![], targets.to_vec())
}

/// 8. THAT IT BETS -- the per-span stake head.
///
/// A loss that reads the hidden state AT THE START of each `span_len`-long
/// span, emits a stake `s = sigmoid(w*h + b)`, and scores it against how well
/// the span actually went (`good` = fraction of correct positions):
/// `loss = mean (s - good)^2`.
///
/// WHY IT'S BUILT THIS WAY:
/// - `hidden` comes in detached from training: the head is a probe on the
///   model's state, it doesn't send it gradient. Letting gradient cross that
///   boundary is decision 4, which is more expensive, and only gets measured
///   if this probe wins.
/// - `good` isn't differentiable (it's argmax correctness), so it comes in
///   as data computed outside and lives in `saved_f`, same as the CE targets.
/// - `saved_v` holds `hidden` and `w` by reference (Arc): under the new
///   contract, saving doesn't copy the buffer.
pub fn stake_loss(
    hidden: &Tensor,
    w: &Tensor,
    b: &Tensor,
    good: &[f32],
    span_len: usize,
) -> Tensor {
    assert_eq!(hidden.shape.len(), 2, "stake_loss expects hidden (S,D)");
    assert_eq!(w.shape.len(), 1, "stake_loss expects w (D,)");
    let (s, d) = (hidden.shape[0], hidden.shape[1]);
    assert_eq!(w.shape[0], d, "w doesn't have D elements");
    assert_eq!(b.shape.len(), 1, "stake_loss expects b (1,)");
    assert_eq!(b.shape[0], 1, "b has to be a scalar");
    let k = good.len();
    let mut loss = 0.0f32;
    let mut zs = vec![0.0f32; k];
    for (kk, &good_k) in good.iter().enumerate() {
        let start = kk * span_len;
        assert!(start + span_len <= s, "span {kk} goes past the end of the window");
        let mut z = b.data[0];
        for j in 0..d {
            z += w.data[j] * hidden.data[start * d + j];
        }
        zs[kk] = z;
        let sval = 1.0 / (1.0 + (-z).exp());
        let e = sval - good_k;
        loss += e * e;
    }
    loss /= k.max(1) as f32;
    let mut sf = good.to_vec();
    sf.extend_from_slice(&zs);
    finalize(
        &[hidden, w, b],
        "stake_loss",
        vec![loss],
        vec![],
        vec![hidden.data.clone(), w.data.clone()],
        sf,
        vec![span_len, k, s],
    )
}

pub fn depthwise_conv1d(x: &Tensor, w: &Tensor, b: &Tensor, k: usize) -> Tensor {
    let (s, d) = (x.shape[0], x.shape[1]);
    let mut out = vec![0.0; s * d];
    for t in 0..s {
        for c in 0..d {
            let mut acc = b.data[c];
            for u in 0..k {
                let src = t as isize - (k as isize - 1) + u as isize;
                if src >= 0 {
                    acc += w.data[c * k + u] * x.data[src as usize * d + c];
                }
            }
            out[t * d + c] = acc;
        }
    }
    finalize(&[x, w, b], "conv1d", out, x.shape.clone(), vec![x.data.clone(), w.data.clone()], vec![], vec![s, d, k])
}

pub fn clockmem(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
) -> Tensor {
    let d = q.shape[1];
    clockmem_from(q, k, v, g, alpha, beta, &vec![0.0; d]).0
}

/// Same thing, but starting from a given state and returning the final
/// state.
///
/// WHY THIS EXISTS: ClockMem's state has a fixed size and its own per-channel
/// forgetting, so **there's no technical reason to reset it on every
/// window**. Resetting is a holdover from the transformer, where the context
/// IS the window and there's no other option. Here, letting it keep running
/// gives memory beyond the window **at no extra byte of cost**.
///
/// `s0` comes in as a CONSTANT: gradient doesn't flow back through it.
/// That's truncated BPTT, and it's intentional -- backpropagating through the
/// entire corpus would mean holding the graph of the entire corpus, which is
/// exactly what we don't want.
pub fn clockmem_from(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    let beta_v = beta.data[0];
    let mut state = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
            for c in 0..d {
                out[t * d + c] = q.data[t * d + c] * cur[c] * g.data[t * d + c];
            }
        }
    });
    let final_state = cur;
    let t = finalize(
        &[q, k, v, g, alpha, beta],
        "clockmem",
        out,
        q.shape.clone(),
        vec![
            q.data.clone(),
            k.data.clone(),
            v.data.clone(),
            g.data.clone(),
            std::sync::Arc::new(state),
            alpha.data.clone(),
            std::sync::Arc::new(s0.to_vec()),
        ],
        vec![beta_v],
        vec![s, d],
    );
    (t, final_state)
}

/// Error-gated write: probe variant of `clockmem_from` where the write
/// magnitude is position-dependent.
///
/// The recurrence is still `cur = alpha*cur + beta_t * k*v`, but
/// `beta_t = beta * r_t` with
///
/// ```text
/// e_t   = 1 - cos(k_t*v_t, state_{t-1})      // how new the write is
/// e_ref = (1-eta)*e_ref + eta*e_t            // running reference, detached
/// r_t   = clamp((e_t / e_ref)^p, 0, 1)
/// ```
///
/// A write that repeats what the state already knows (small e_t) is scaled
/// down. `e_ref` is a per-call (per-window) running average starting at 1.0,
/// so the gate is self-normalizing and the magnitude of the writes doesn't
/// have to be tuned by hand.
///
/// THE GATE IS DETACHED: r_t is a function of k/v/state, but no gradient
/// flows through it -- the backward treats each beta_t as a constant. That
/// isolates "what changes when the write magnitude depends on novelty" from
/// "what changes when gradients also chase the gate". It adds no parameters;
/// the only knob is the exponent `p` (p=0 disables the gate: r_t=1).
///
/// `trace` records each (e_t, r_t) pair into the write trace for the
/// measurement probe.
pub fn clockmem_gated_from(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    p: f32,
    s0: &[f32],
    trace: bool,
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    let beta_v = beta.data[0];
    let eta = 0.1f32;
    let eps = 1e-6f32;
    let mut state = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    let mut rs = Vec::with_capacity(s);
    let mut e_ref = 1.0f32;
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            let base = t * d;
            let prev = if t == 0 { s0 } else { &state[(t - 1) * d..base] };
            let mut dot = 0.0;
            let mut nw2 = 0.0;
            let mut np2 = 0.0;
            for c in 0..d {
                let w = k.data[base + c] * v.data[base + c];
                dot += w * prev[c];
                nw2 += w * w;
                np2 += prev[c] * prev[c];
            }
            let e = 1.0 - dot / (nw2.sqrt() * np2.sqrt() + eps);
            e_ref = (1.0 - eta) * e_ref + eta * e;
            let r = (e / e_ref.max(1e-9)).powf(p).clamp(0.0, 1.0);
            rs.push(r);
            if trace {
                WRITE_TRACE.with(|t| t.borrow_mut().extend_from_slice(&[e, r]));
            }
            let beta_t = beta_v * r;
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_t * k.data[base + c] * v.data[base + c];
                state[base + c] = cur[c];
            }
            for c in 0..d {
                out[base + c] = q.data[base + c] * cur[c] * g.data[base + c];
            }
        }
    });
    let final_state = cur;
    let mut saved_f = Vec::with_capacity(s + 1);
    saved_f.push(beta_v);
    saved_f.extend_from_slice(&rs);
    let t = finalize(
        &[q, k, v, g, alpha, beta],
        "clockmem_gated",
        out,
        q.shape.clone(),
        vec![
            q.data.clone(),
            k.data.clone(),
            v.data.clone(),
            g.data.clone(),
            std::sync::Arc::new(state),
            alpha.data.clone(),
            std::sync::Arc::new(s0.to_vec()),
        ],
        saved_f,
        vec![s, d],
    );
    (t, final_state)
}

/// Same state as `clockmem_from`, but the READ is a learned window over the
/// last K states instead of the single current state:
///   read[t,c] = sum_{j=0..K-1} wread[j,c] * state[t-j,c]   (zeros for t-j<0)
///   out[t,c]  = q[t,c] * read[t,c] * g[t,c]
/// EXPERIMENT (2026-08-14): test whether ClockMem's long
/// range fails because out=q*cur*g can only see a blurred leaky average, not
/// a specific past position. The window is the minimal fix: with K>=N+1 the
/// readout CAN express the finite difference that recovers the write at
/// position t-N+1. State update and alpha are untouched.
pub fn clockmem_readwin(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    wread: &Tensor,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    let kwin = wread.shape[0];
    assert_eq!(wread.shape[1], d, "wread must be [K, D]");
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    let beta_v = beta.data[0];
    let mut state = vec![0.0; s * d];
    let mut read = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
        }
        // windowed read: read[t,c] = sum_j w[j,c]*state[t-j,c] (t-j>=0),
        // parallel over t-bands (c inner, sequential/SIMD friendly).
        for t in 0..s {
            for c in 0..d {
                read[t * d + c] = 0.0;
            }
        }
        let bands = crate::pool::global().workers() + 1;
        let (s_f, d_f, kwin_f) = (s, d, kwin);
        let state_ptr = crate::pool::Ptr(state.as_ptr());
        let w_ptr = crate::pool::Ptr(wread.data.as_ptr());
        let read_ptr = crate::pool::Ptr(read.as_mut_ptr());
        crate::pool::global().run(bands, move |b| {
            let t0 = b * s_f / bands;
            let t1 = ((b + 1) * s_f / bands).min(s_f);
            let state = state_ptr.as_ref(s_f * d_f);
            let w = w_ptr.as_ref(kwin_f * d_f);
            let read = read_ptr.as_mut(s_f * d_f);
            for t in t0..t1 {
                let jmax = kwin_f.min(t + 1);
                let toff = t * d_f;
                for j in 0..jmax {
                    let woff = j * d_f;
                    let soff = (t - j) * d_f;
                    for c in 0..d_f {
                        read[toff + c] += w[woff + c] * state[soff + c];
                    }
                }
            }
        });
        for t in 0..s {
            for c in 0..d {
                out[t * d + c] = q.data[t * d + c] * read[t * d + c] * g.data[t * d + c];
            }
        }
    });
    let saved_v = vec![
        q.data.clone(),
        k.data.clone(),
        v.data.clone(),
        g.data.clone(),
        std::sync::Arc::new(state),
        std::sync::Arc::new(read),
        alpha.data.clone(),
        std::sync::Arc::new(s0.to_vec()),
        wread.data.clone(),
    ];
    let t = finalize(
        &[q, k, v, g, alpha, beta, wread],
        "clockmem_readwin",
        out,
        q.shape.clone(),
        saved_v,
        vec![beta_v],
        vec![s, d, kwin],
    );
    (t, cur)
}

/// CLEAN INJECTION (2026-08-14, after the activation probe
/// showed q*g kills the recovered write: b0.read 38.6% -> b0.m 0.84%).
///
/// Same state as `clockmem_from`, but the OUTPUT is the gated current-state
/// read PLUS the exactly recovered write, injected into the residual stream
/// WITHOUT passing through q*g:
///   out[t,c] = q[t,c]*cur[t,c]*g[t,c] + write_rec[t,c]
///   write_rec[t,c] = (cur[t-N+1,c] - alpha_c*cur[t-N,c]) / beta   (t>=N)
/// This is the oracle readout minus the gate on the recovered write. The
/// write is a fixed function of the state (no learned taps), so whatever the
/// head can decode from it tests whether the q*g gate is what destroys the
/// long-range signal.
pub fn clockmem_inject(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    n: usize,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    assert!(n >= 1 && n < s, "N has to be in [1, s)");
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    let beta_v = beta.data[0];
    let mut state = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
            for c in 0..d {
                out[t * d + c] = q.data[t * d + c] * cur[c] * g.data[t * d + c];
                if t >= n {
                    out[t * d + c] +=
                        (state[(t - n + 1) * d + c] - alpha.data[c] * state[(t - n) * d + c]) / beta_v;
                }
            }
        }
    });
    let t = finalize(
        &[q, k, v, g, alpha, beta],
        "clockmem_inject",
        out,
        q.shape.clone(),
        vec![
            q.data.clone(),
            k.data.clone(),
            v.data.clone(),
            g.data.clone(),
            std::sync::Arc::new(state),
            alpha.data.clone(),
            std::sync::Arc::new(s0.to_vec()),
        ],
        vec![beta_v],
        vec![s, d, n],
    );
    (t, cur)
}

/// R1c (2026-08-15): the structured learnable inverse.
/// Same state and additive residual injection as `clockmem_inject`, but the
/// recovered write uses PER-CHANNEL LEARNABLE parameters instead of the
/// physical alpha/beta:
///   read[t,c] = inv_beta[c] * (state[t-N+1,c] - alpha_read[c]*state[t-N,c])
///   out[t,c]  = q[t,c]*cur[t,c]*g[t,c] + read[t,c]            (t>=N)
/// Two safety choices, both after R1a/R1b showed the free
/// Kxd matrix drifting into noisy taps:
///   * NO free matrix: the read is the inverse-leaky-filter family, ~2
///     params/channel (alpha_read, inv_beta) -- the noisy-tap escape R1a
///     found is structurally impossible here.
///   * NO division by a learned parameter: we learn inv_beta = 1/beta_read
///     directly and it MULTIPLIES, so a beta_read->0 singularity (a new
///     pathological gradient like alpha/q*g) cannot appear.
///
/// By construction the family contains the verified oracle: alpha_read=alpha_c
/// and inv_beta=1/beta (=1 here) makes read[t] = write[t-N+1] EXACTLY, i.e.
/// the op reduces to `clockmem_inject` (tested in r1c_reproduces_inject).
pub fn clockmem_inject_learn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    alpha_read: &Tensor,
    inv_beta: &Tensor,
    n: usize,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    assert!(n >= 1 && n < s, "N has to be in [1, s)");
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    assert_eq!(alpha_read.shape, vec![d], "alpha_read must be [D]");
    assert_eq!(inv_beta.shape, vec![d], "inv_beta must be [D]");
    let beta_v = beta.data[0];
    let mut state = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
            for c in 0..d {
                out[t * d + c] = q.data[t * d + c] * cur[c] * g.data[t * d + c];
                if t >= n {
                    out[t * d + c] += inv_beta.data[c]
                        * (state[(t - n + 1) * d + c] - alpha_read.data[c] * state[(t - n) * d + c]);
                }
            }
        }
    });
    let t = finalize(
        &[q, k, v, g, alpha, beta, alpha_read, inv_beta],
        "clockmem_inject_learn",
        out,
        q.shape.clone(),
        vec![
            q.data.clone(),
            k.data.clone(),
            v.data.clone(),
            g.data.clone(),
            std::sync::Arc::new(state),
            alpha.data.clone(),
            std::sync::Arc::new(s0.to_vec()),
            alpha_read.data.clone(),
            inv_beta.data.clone(),
        ],
        vec![beta_v],
        vec![s, d, n],
    );
    (t, cur)
}

// R2 trace support: per-clock-block accumulator for the EFFECTIVE memory
// contribution |s*read| (the term R2 actually adds to the residual). Keyed by
// the block's alpha_read tensor id. Written during the op forward, read and
// reset by the training trace (WreadTrace) -- so `s * memory_path` is logged
// separately from `z` and `s`, as we asked.
use std::sync::Mutex;
static R2_READ: Mutex<Vec<(usize, f64, u64)>> = Mutex::new(Vec::new());

fn r2_accum(key: usize, v: f64) {
    if let Ok(mut m) = R2_READ.lock() {
        for e in m.iter_mut() {
            if e.0 == key {
                e.1 += v;
                e.2 += 1;
                return;
            }
        }
        m.push((key, v, 1));
    }
}

/// Mean |s*read| accumulated since the last call (read+reset). None if the
/// block has never run in R2 mode.
pub fn r2_read_stats(key: usize) -> Option<(f32, u64)> {
    let mut m = R2_READ.lock().unwrap();
    for e in m.iter_mut() {
        if e.0 == key {
            let n = e.2;
            let mean = if n > 0 { (e.1 / n as f64) as f32 } else { 0.0 };
            e.1 = 0.0;
            e.2 = 0;
            return Some((mean, n));
        }
    }
    None
}

/// R2 (2026-08-15): the structured memory path behind a
/// gated scalar that CANNOT annul it.
///   out[t,c] = q[t,c]*cur[t,c]*g[t,c] + s*read[t,c]       (t>=N)
///   s        = floor + (1-floor)*sigmoid(z)   (z learnable, floor>0 fixed)
///   read[t,c] = inv_beta[c]*(state[t-N+1,c] - alpha_read[c]*state[t-N,c])
/// The floor guarantees s >= floor > 0: even if the gate saturates closed,
/// memory_path keeps contributing and its own params (alpha_read, inv_beta)
/// keep receiving REAL gradient -- the property (2) this experiment tests.
/// We learn inv_beta directly (never divide by a learned param, the /// prevention). z receives gradient only while sigmoid(z) is unsaturated
/// (property (1), explicitly NOT required by the design). At z such that
/// sigmoid(z)=1 the op reduces to `clockmem_inject_learn`.
pub fn clockmem_inject_learn_g(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    alpha_read: &Tensor,
    inv_beta: &Tensor,
    z: &Tensor,
    floor: f32,
    n: usize,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    assert!(n >= 1 && n < s, "N has to be in [1, s)");
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    assert_eq!(alpha_read.shape, vec![d], "alpha_read must be [D]");
    assert_eq!(inv_beta.shape, vec![d], "inv_beta must be [D]");
    assert_eq!(z.shape, vec![1], "z must be [1]");
    assert!(floor > 0.0 && floor < 1.0, "floor must be in (0,1)");
    let beta_v = beta.data[0];
    let z_v = z.data[0];
    let sg = 1.0 / (1.0 + (-z_v).exp());
    let s_gate = floor + (1.0 - floor) * sg;
    let mut state = vec![0.0; s * d];
    let mut eff = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    let mut eff_acc = 0.0f64;
    let mut eff_n = 0u64;
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
            for c in 0..d {
                out[t * d + c] = q.data[t * d + c] * cur[c] * g.data[t * d + c];
                if t >= n {
                    let rd = inv_beta.data[c]
                        * (state[(t - n + 1) * d + c] - alpha_read.data[c] * state[(t - n) * d + c]);
                    let e = s_gate * rd;
                    eff[t * d + c] = e;
                    out[t * d + c] += e;
                    eff_acc += e.abs() as f64;
                    eff_n += 1;
                }
            }
        }
        r2_accum(alpha_read.id, eff_acc / eff_n.max(1) as f64);
    });
    let sp = sg * (1.0 - sg); // sigmoid'(z), for the gate gradient in backward
    let t = finalize(
        &[q, k, v, g, alpha, beta, alpha_read, inv_beta, z],
        "clockmem_inject_learn_g",
        out,
        q.shape.clone(),
        vec![
            q.data.clone(),
            k.data.clone(),
            v.data.clone(),
            g.data.clone(),
            std::sync::Arc::new(state),
            alpha.data.clone(),
            std::sync::Arc::new(s0.to_vec()),
            alpha_read.data.clone(),
            inv_beta.data.clone(),
            std::sync::Arc::new(eff),
        ],
        vec![beta_v, s_gate, sp, floor],
        vec![s, d, n],
    );
    (t, cur)
}

/// R2-channel (2026-08-15): the block gate vs channel gate control.
/// Same as `clockmem_inject_learn_g` but z is [D]: one gate PER CHANNEL,
///   s[c] = floor + (1-floor)*sigmoid(z[c])   (z[c] learned, floor>0 fixed)
///   out[t,c] = q*cur*g + s[c]*read[t,c]
/// With z constant across channels this reduces EXACTLY to the scalar version
/// (the scalar family is a subset of the channel family), so R2-block vs
/// R2-channel isolates the "spatial freedom" of the gate: same floor, same
/// memory path, same budget. The floor guarantee (s[c] >= floor > 0, memory
/// path can never be annulled) is per channel here.
pub fn clockmem_inject_learn_gc(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    alpha_read: &Tensor,
    inv_beta: &Tensor,
    z: &Tensor,
    floor: f32,
    n: usize,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    assert!(n >= 1 && n < s, "N has to be in [1, s)");
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    assert_eq!(alpha_read.shape, vec![d], "alpha_read must be [D]");
    assert_eq!(inv_beta.shape, vec![d], "inv_beta must be [D]");
    assert_eq!(z.shape, vec![d], "z must be [D] for the channel gate");
    assert!(floor > 0.0 && floor < 1.0, "floor must be in (0,1)");
    let beta_v = beta.data[0];
    let mut sg = vec![0.0; d];
    let mut sp = vec![0.0; d];
    for c in 0..d {
        let zv = z.data[c];
        let s2 = 1.0 / (1.0 + (-zv).exp());
        sg[c] = floor + (1.0 - floor) * s2;
        sp[c] = s2 * (1.0 - s2);
    }
    let mut state = vec![0.0; s * d];
    let mut eff = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    let mut eff_acc = 0.0f64;
    let mut eff_n = 0u64;
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
            for c in 0..d {
                out[t * d + c] = q.data[t * d + c] * cur[c] * g.data[t * d + c];
                if t >= n {
                    let rd = inv_beta.data[c]
                        * (state[(t - n + 1) * d + c] - alpha_read.data[c] * state[(t - n) * d + c]);
                    let e = sg[c] * rd;
                    eff[t * d + c] = e;
                    out[t * d + c] += e;
                    eff_acc += e.abs() as f64;
                    eff_n += 1;
                }
            }
        }
        r2_accum(alpha_read.id, eff_acc / eff_n.max(1) as f64);
    });
    let t = finalize(
        &[q, k, v, g, alpha, beta, alpha_read, inv_beta, z],
        "clockmem_inject_learn_gc",
        out,
        q.shape.clone(),
        vec![
            q.data.clone(),
            k.data.clone(),
            v.data.clone(),
            g.data.clone(),
            std::sync::Arc::new(state),
            alpha.data.clone(),
            std::sync::Arc::new(s0.to_vec()),
            alpha_read.data.clone(),
            inv_beta.data.clone(),
            std::sync::Arc::new(eff),
            std::sync::Arc::new(sg),
            std::sync::Arc::new(sp),
        ],
        vec![beta_v, floor],
        vec![s, d, n],
    );
    (t, cur)
}

/// R1 (2026-08-15): the windowed read WITHOUT the q*g gate.
///   state: same leaky recurrence cur = alpha*cur + beta*k*v.
///   read[t,c] = sum_{j<K} wread[j,c] * state[t-j,c]   (zeros for t-j<0)
///   out[t,c]  = q[t,c]*cur[t,c]*g[t,c] + read[t,c]
/// Same state as `clockmem_readwin`, but the read goes to the residual
/// ADDITIVELY instead of being multiplied by q*g. With wread initialized to
/// the finite-difference taps (R1a, EVA_READ_DF_N) this equals
/// `clockmem_inject` at init; with a zero init (R1b) the model must discover
/// the long read on its own. The gradient into wread flows through the
/// additive path -- there is no gate that can annul it, which is the whole
/// hypothesis (the q*g gate is the long-range bottleneck).
pub fn clockmem_readwin_inj(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    wread: &Tensor,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    let kwin = wread.shape[0];
    assert_eq!(wread.shape[1], d, "wread must be [K, D]");
    assert_eq!(s0.len(), d, "el estado inicial no tiene D elementos");
    let beta_v = beta.data[0];
    let mut state = vec![0.0; s * d];
    let mut read = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut cur = s0.to_vec();
    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
        }
        // windowed read: read[t,c] = sum_j w[j,c]*state[t-j,c] (t-j>=0),
        // parallel over t-bands (c inner, sequential/SIMD friendly).
        let bands = crate::pool::global().workers() + 1;
        let (s_f, d_f, kwin_f) = (s, d, kwin);
        let state_ptr = crate::pool::Ptr(state.as_ptr());
        let w_ptr = crate::pool::Ptr(wread.data.as_ptr());
        let read_ptr = crate::pool::Ptr(read.as_mut_ptr());
        crate::pool::global().run(bands, move |b| {
            let t0 = b * s_f / bands;
            let t1 = ((b + 1) * s_f / bands).min(s_f);
            let state = state_ptr.as_ref(s_f * d_f);
            let w = w_ptr.as_ref(kwin_f * d_f);
            let read = read_ptr.as_mut(s_f * d_f);
            for t in t0..t1 {
                let jmax = kwin_f.min(t + 1);
                let toff = t * d_f;
                for j in 0..jmax {
                    let woff = j * d_f;
                    let soff = (t - j) * d_f;
                    for c in 0..d_f {
                        read[toff + c] += w[woff + c] * state[soff + c];
                    }
                }
            }
        });
        for t in 0..s {
            for c in 0..d {
                out[t * d + c] =
                    q.data[t * d + c] * state[t * d + c] * g.data[t * d + c] + read[t * d + c];
            }
        }
    });
    let saved_v = vec![
        q.data.clone(),
        k.data.clone(),
        v.data.clone(),
        g.data.clone(),
        std::sync::Arc::new(state),
        std::sync::Arc::new(read),
        alpha.data.clone(),
        std::sync::Arc::new(s0.to_vec()),
        wread.data.clone(),
    ];
    let t = finalize(
        &[q, k, v, g, alpha, beta, wread],
        "clockmem_readwin_inj",
        out,
        q.shape.clone(),
        saved_v,
        vec![beta_v],
        vec![s, d, kwin],
    );
    (t, cur)
}

/// LOW-RANK OUTER-PRODUCT WRITE (task #58,
/// 2026-08-18). The one intervention Section 4 never tried: it changes the
/// WRITE, not the read.
///
/// WHY. Every mechanism in Sections 4.2-4.6 (learnable window, structured
/// init, oracle, injection, R1c, R2-with-floor, error-gated write) targets
/// how the state is READ. Section 4.11 measured that the 8-key failure is
/// not a memory failure at all: the model identifies the correct candidate
/// set 99.91% of the time and still picks the wrong key's value 76.49% of
/// the time. That is a BINDING failure, and it is structural. The base
/// write is element-wise,
///
/// ```text
/// cur[c] += beta * k[t,c] * v[t,c]
/// ```
///
/// with no term coupling channel `c` to channel `c'`. Such a state can
/// DEDICATE channels to a key; it has nowhere to put "this key goes with
/// that value". No amount of reading fixes something destroyed at write.
///
/// WHAT THIS DOES. `k` and `v` are projected to a small rank `r`, and the
/// state gains an `r x r` matrix written with an OUTER PRODUCT, which is
/// exactly the missing cross term:
///
/// ```text
/// kr[t,i] = sum_c k[t,c]*pk[c,i]      vr[t,j] = sum_c v[t,c]*pv[c,j]
/// qr[t,i] = sum_c q[t,c]*pq[c,i]
/// M[i,j]  = am[i]*M[i,j] + beta*kr[t,i]*vr[t,j]      <- the binding lives here
/// rd[t,j] = sum_i qr[t,i]*M[i,j]
/// mem[t,c]= sum_j rd[t,j]*po[j,c]
/// out[t,c]= q[t,c]*cur[c]*g[t,c] + mem[t,c]
/// ```
///
/// TWO DESIGN CHOICES, both taken from results already in this codebase,
/// not from taste:
///
/// 1. **The element-wise path stays.** It reaches 99.65% at 2 keys; removing
///    it would change two things at once. The low-rank path is ADDITIVE.
/// 2. **The new path bypasses `g`.** Section 4 spent most of its length on a
///    multiplicative output gate destroying an already-correct read, and the
///    fix that worked (R1/inject) was to send the recovered value to the
///    residual stream WITHOUT `q*g`. Repeating `*g` here would rebuild the
///    exact failure this op exists to escape. There is deliberately no gate
///    on `mem`, floored or otherwise.
///
/// COST. `4*d*r` new parameters (65,536 at d=512, r=32 -- about 0.5% of the
/// 13.4M model) and an `r*r` state per block (1024 numbers against 512).
/// Arithmetic grows by roughly 6% of a block at those sizes.
///
/// LIMITATION, DECLARED: `M` starts at zero every window and is not carried
/// across windows, unlike the element-wise `cur` (whose final value is
/// returned as before). For the associative benchmark this is correct rather
/// than merely tolerable -- bindings are generated fresh per 512-byte window
/// and `--seq 512` matches that -- but it means this op is NOT yet a drop-in
/// for the persistent-state runs on real text.
///
/// `alpha_m` arrives already in (0,1), same convention as `alpha`: the
/// squashing lives in the model, not in the op.
pub fn clockmem_lowrank_from(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    pk: &Tensor,
    pv: &Tensor,
    pq: &Tensor,
    po: &Tensor,
    alpha_m: &Tensor,
    s0: &[f32],
) -> (Tensor, Vec<f32>) {
    let (s, d) = (q.shape[0], q.shape[1]);
    let r = alpha_m.shape[0];
    assert_eq!(s0.len(), d, "the initial state does not have D elements");
    assert_eq!(pk.shape, vec![d, r], "pk must be [D,r]");
    assert_eq!(pv.shape, vec![d, r], "pv must be [D,r]");
    assert_eq!(pq.shape, vec![d, r], "pq must be [D,r]");
    assert_eq!(po.shape, vec![r, d], "po must be [r,D]");
    let beta_v = beta.data[0];

    let mut state = vec![0.0; s * d];
    let mut out = vec![0.0; s * d];
    let mut kr = vec![0.0; s * r];
    let mut vr = vec![0.0; s * r];
    let mut qr = vec![0.0; s * r];
    // Full history of M: the backward needs M[t-1] to get alpha_m's
    // gradient, exactly like the element-wise path needs the previous state.
    let mut mhist = vec![0.0; s * r * r];
    let mut cur = s0.to_vec();
    let mut m = vec![0.0f32; r * r];

    crate::prof::time(crate::prof::P::ClockFwd, || {
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha.data[c] * cur[c] + beta_v * k.data[t * d + c] * v.data[t * d + c];
                state[t * d + c] = cur[c];
            }
            // Project down to rank r.
            for c in 0..d {
                let (kc, vc, qc) = (k.data[t * d + c], v.data[t * d + c], q.data[t * d + c]);
                for i in 0..r {
                    kr[t * r + i] += kc * pk.data[c * r + i];
                    vr[t * r + i] += vc * pv.data[c * r + i];
                    qr[t * r + i] += qc * pq.data[c * r + i];
                }
            }
            // Outer-product write: the cross term the element-wise write lacks.
            for i in 0..r {
                let ai = alpha_m.data[i];
                let bki = beta_v * kr[t * r + i];
                for j in 0..r {
                    m[i * r + j] = ai * m[i * r + j] + bki * vr[t * r + j];
                }
            }
            mhist[t * r * r..(t + 1) * r * r].copy_from_slice(&m);
            // Read: query the matrix, project back up to D.
            let mut rd = vec![0.0f32; r];
            for i in 0..r {
                let qi = qr[t * r + i];
                if qi == 0.0 { continue; }
                for j in 0..r {
                    rd[j] += qi * m[i * r + j];
                }
            }
            for c in 0..d {
                out[t * d + c] = q.data[t * d + c] * cur[c] * g.data[t * d + c];
            }
            for j in 0..r {
                let rj = rd[j];
                if rj == 0.0 { continue; }
                for c in 0..d {
                    out[t * d + c] += rj * po.data[j * d + c];
                }
            }
        }
    });

    let final_state = cur;
    let t = finalize(
        &[q, k, v, g, alpha, beta, pk, pv, pq, po, alpha_m],
        "clockmem_lowrank",
        out,
        q.shape.clone(),
        vec![
            q.data.clone(),
            k.data.clone(),
            v.data.clone(),
            g.data.clone(),
            std::sync::Arc::new(state),
            alpha.data.clone(),
            std::sync::Arc::new(s0.to_vec()),
            std::sync::Arc::new(kr),
            std::sync::Arc::new(vr),
            std::sync::Arc::new(qr),
            std::sync::Arc::new(mhist),
            pk.data.clone(),
            pv.data.clone(),
            pq.data.clone(),
            po.data.clone(),
            alpha_m.data.clone(),
        ],
        vec![beta_v],
        vec![s, d, r],
    );
    (t, final_state)
}
