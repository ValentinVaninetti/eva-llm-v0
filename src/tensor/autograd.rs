use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::tensor::Tensor;

pub struct Input {
    pub id: usize,
    pub len: usize,
    pub node: Option<Arc<Node>>,
}

impl Node {
    /// Bytes this node **actually** holds.
    ///
    /// A shared buffer doesn't count: if someone else keeps it alive --and
    /// for the weights that someone is the model-- holding it here doesn't
    /// reserve a single byte. Counting by length, like the first version of
    /// this meter did, gave the exact same number before and after we
    /// stopped cloning. The instrument couldn't see the change it was built
    /// to measure.
    fn own_weight(&self) -> usize {
        self.saved_v
            .iter()
            .filter(|v| std::sync::Arc::strong_count(v) == 1)
            .map(|v| v.len() * 4)
            .sum::<usize>()
            + self.saved_f.len() * 4
            + self.saved_u.len() * std::mem::size_of::<usize>()
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // The weight was computed at creation and travels with the node:
        // recomputing it here would give a different result if someone
        // cloned the Arc in the meantime, and the counter would end up
        // unbalanced.
        crate::tensor::held::leave(self.held_weight);
    }
}

pub struct Node {
    pub op: &'static str,
    pub inputs: Vec<Input>,
    pub grad_flags: Vec<bool>,
    /// What backward needs from the forward pass.
    ///
    /// `Arc` and not `Vec`: saving the input of an operation used to mean
    /// **copying the buffer**. For `x @ W` that meant cloning the weight
    /// matrix, which is already alive in the model. Measured: 83% of what
    /// `matmul` was saving was duplicated weights, and `matmul` is 65% of
    /// the graph. Now saving just means bumping a counter.
    pub saved_v: Vec<std::sync::Arc<Vec<f32>>>,
    pub saved_f: Vec<f32>,
    pub saved_u: Vec<usize>,
    pub out_id: usize,
    pub out_len: usize,
    /// What this node actually reserves, fixed at creation time.
    held_weight: usize,
}

pub fn make_node(
    op: &'static str,
    inputs: &[&Tensor],
    grad_flags: Vec<bool>,
    saved_v: Vec<std::sync::Arc<Vec<f32>>>,
    saved_f: Vec<f32>,
    saved_u: Vec<usize>,
    out_id: usize,
    out_len: usize,
) -> Arc<Node> {
    let inputs: Vec<Input> = inputs
        .iter()
        .map(|t| Input { id: t.id, len: t.numel(), node: t.node.clone() })
        .collect();
    let mut n = Node {
        op, inputs, grad_flags, saved_v, saved_f, saved_u, out_id, out_len,
        held_weight: 0,
    };
    n.held_weight = n.own_weight();
    crate::tensor::held::enter(op, n.held_weight);
    Arc::new(n)
}

pub fn backward(out: &Tensor) -> HashMap<usize, Vec<f32>> {
    let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
    if out.node.is_none() {
        return grads;
    }
    grads.insert(out.id, vec![1.0; out.numel()]);

    let mut topo: Vec<Arc<Node>> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    visit(out.node.clone().unwrap(), &mut topo, &mut visited);

    for node in topo.iter().rev() {
        let g_out = grads.remove(&node.out_id).expect("missing upstream grad");
        let contrib = backward_op(node, &g_out);
        for (idx, g) in contrib {
            if node.grad_flags[idx] {
                let inp = &node.inputs[idx];
                let entry = grads.entry(inp.id).or_insert_with(|| vec![0.0; inp.len]);
                debug_assert_eq!(entry.len(), g.len());
                for i in 0..g.len() {
                    entry[i] += g[i];
                }
            }
        }
    }

    let bytes: usize = grads.values().map(|v| v.len() * 4).sum();
    crate::tensor::held::grads(bytes);
    grads
}

fn visit(node: Arc<Node>, topo: &mut Vec<Arc<Node>>, visited: &mut HashSet<usize>) {
    if !visited.insert(node.out_id) {
        return;
    }
    for inp in &node.inputs {
        if let Some(n) = &inp.node {
            visit(n.clone(), topo, visited);
        }
    }
    topo.push(node);
}

/// Dimensions handled without touching the heap. The tensors in this model
/// have between 1 and 3; eight leaves plenty of headroom.
const MAX_DIMS: usize = 8;

/// Walks the output, handing back `(output index, source index)`.
///
/// WITHOUT ALLOCATING ANYTHING, which is the whole point. The previous
/// version called `index_to_coords` for every element, which returned a
/// `Vec`, and then built another coordinate `Vec` alongside it: **two
/// allocations per number**. For a 64x1024 tensor that's 131 thousand
/// allocations in ONE call, and this gets called in the backward of `add`
/// and `mul`, which is where almost half the time was going.
///
/// The source index is tracked with an odometer: the dimension being
/// broadcast has stride 0, which is exactly what repeating a value means.
///
/// Returns `false` if the shapes aren't compatible, and **that's decided
/// only once**: the old check was inside the loop but only depended on the
/// shapes, so it gave the same answer for millions of elements.
fn walk(out_shape: &[usize], src_shape: &[usize], mut visit: impl FnMut(usize, usize)) -> bool {
    let nd = out_shape.len();
    if nd > MAX_DIMS || src_shape.len() > nd {
        return false;
    }
    let off = nd - src_shape.len();
    let mut step = [0usize; MAX_DIMS];
    let mut stride = 1usize;
    for d in (0..src_shape.len()).rev() {
        if src_shape[d] == out_shape[off + d] {
            step[off + d] = stride;
        } else if src_shape[d] == 1 {
            step[off + d] = 0;
        } else {
            return false;
        }
        stride *= src_shape[d];
    }

    let total: usize = out_shape.iter().product();
    let mut coord = [0usize; MAX_DIMS];
    let mut src = 0usize;
    for o in 0..total {
        visit(o, src);
        for d in (0..nd).rev() {
            coord[d] += 1;
            src += step[d];
            if coord[d] < out_shape[d] {
                break;
            }
            // Wrapped around: subtract what the full cycle added.
            src -= step[d] * out_shape[d];
            coord[d] = 0;
        }
    }
    true
}

fn broadcast_to(x: &[f32], xshape: &[usize], oshape: &[usize]) -> Vec<f32> {
    if xshape == oshape {
        return x.to_vec();
    }
    let mut out = vec![0.0; oshape.iter().product()];
    // Incompatible shapes leave everything at zero, same as before.
    walk(oshape, xshape, |o, s| out[o] = x[s]);
    out
}

fn reduce(g: &[f32], out_shape: &[usize], to_shape: &[usize]) -> Vec<f32> {
    if out_shape == to_shape {
        return g.to_vec();
    }
    let mut out = vec![0.0; to_shape.iter().product()];
    walk(out_shape, to_shape, |o, s| out[s] += g[o]);
    out
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
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

fn n_su(n: &Node, i: usize) -> usize {
    n.saved_u[i]
}

fn backward_op(n: &Node, g: &[f32]) -> Vec<(usize, Vec<f32>)> {
    let mut out: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut push = |idx: usize, gv: Vec<f32>| {
        if n.grad_flags[idx] {
            out.push((idx, gv));
        }
    };

    match n.op {
        "add" => {
            let a_nd = n.saved_u[0];
            let a_shape = n.saved_u[1..1 + a_nd].to_vec();
            let b_nd = n.saved_u[1 + a_nd];
            let b_shape = n.saved_u[2 + a_nd..2 + a_nd + b_nd].to_vec();
            let out_shape = broadcast_shape(&a_shape, &b_shape);
            push(0, reduce(g, &out_shape, &a_shape));
            push(1, reduce(g, &out_shape, &b_shape));
        }
        "mul" => {
            let a = &n.saved_v[0];
            let b = &n.saved_v[1];
            let a_nd = n.saved_u[0];
            let a_shape = n.saved_u[1..1 + a_nd].to_vec();
            let b_nd = n.saved_u[1 + a_nd];
            let b_shape = n.saved_u[2 + a_nd..2 + a_nd + b_nd].to_vec();
            let out_shape = broadcast_shape(&a_shape, &b_shape);
            let b_br = broadcast_to(&b, &b_shape, &out_shape);
            let a_br = broadcast_to(&a, &a_shape, &out_shape);
            let mut ga = vec![0.0; g.len()];
            let mut gb = vec![0.0; g.len()];
            for i in 0..g.len() {
                ga[i] = g[i] * b_br[i];
                gb[i] = g[i] * a_br[i];
            }
            push(0, reduce(&ga, &out_shape, &a_shape));
            push(1, reduce(&gb, &out_shape, &b_shape));
        }
        "scale" => {
            let s = n.saved_f[0];
            let gs: Vec<f32> = g.iter().map(|&x| x * s).collect();
            push(0, gs);
        }
        "sum_all" => {
            let n = n.saved_u[0];
            push(0, vec![g[0]; n]);
        }
        "matmul" => {
            let a = &n.saved_v[0];
            let b = &n.saved_v[1];
            let (m, k, nn) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let mut ga = vec![0.0; m * k];
            let mut gb = vec![0.0; k * nn];
            let bt = crate::math::transpose(b, k, nn);
            crate::math::matmul(g, &bt, m, nn, k, &mut ga);
            let at = crate::math::transpose(a, m, k);
            crate::math::matmul(&at, g, k, m, nn, &mut gb);
            push(0, ga);
            push(1, gb);
        }
        "transpose" => {
            let (r, c) = (n.saved_u[0], n.saved_u[1]);
            // Transpose is its own inverse: the gradient just comes back
            // flipped and nothing else.
            push(0, crate::math::transpose(g, c, r));
        }
        "softmax_causal" => {
            let s = n.saved_u[0];
            let y = &n.saved_v[0];
            let mut gx = vec![0.0; s * s];
            for i in 0..s {
                // dL/dx_j = y_j * (g_j - Sum_k g_k y_k), with the sum only
                // over what the row actually looks at.
                let mut dot = 0.0;
                for j in 0..=i {
                    dot += g[i * s + j] * y[i * s + j];
                }
                for j in 0..=i {
                    gx[i * s + j] = y[i * s + j] * (g[i * s + j] - dot);
                }
            }
            push(0, gx);
        }
        "slice_rows" => {
            let (s, d, n) = (n_su(n, 0), n_su(n, 1), n_su(n, 2));
            // Whatever wasn't sliced got nothing: zero.
            let mut gx = vec![0.0; s * d];
            gx[..n * d].copy_from_slice(&g[..n * d]);
            push(0, gx);
        }
        "silu" => {
            let x = &n.saved_v[0];
            let mut gx = vec![0.0; x.len()];
            for i in 0..x.len() {
                let sig = 1.0 / (1.0 + (-x[i]).exp());
                gx[i] = g[i] * (sig + x[i] * sig * (1.0 - sig));
            }
            push(0, gx);
        }
        "sigmoid" => {
            let s = &n.saved_v[0];
            let mut gx = vec![0.0; s.len()];
            for i in 0..s.len() {
                gx[i] = g[i] * s[i] * (1.0 - s[i]);
            }
            push(0, gx);
        }
        "algebraic_sigmoid" => {
            // s(z) = 0.5*(1+z/sqrt(1+z^2)); s'(z) = 0.5/(1+z^2)^1.5.
            let x = &n.saved_v[0];
            let mut gx = vec![0.0; x.len()];
            for i in 0..x.len() {
                let denom = (1.0 + x[i] * x[i]).powf(1.5);
                gx[i] = g[i] * 0.5 / denom;
            }
            push(0, gx);
        }
        "rms_norm" => {
            let x = &n.saved_v[0];
            let eps = n.saved_f[0];
            let s = n.saved_u[0];
            let d = n.saved_u[1];
            let w = &n.saved_v[1];
            let mut gx = vec![0.0; s * d];
            let mut gw = vec![0.0; d];
            for i in 0..s {
                let mut sq = 0.0;
                for j in 0..d {
                    sq += x[i * d + j] * x[i * d + j];
                }
                let inv = 1.0 / (sq / d as f32 + eps).sqrt();
                let inv3 = inv * inv * inv;
                let mut c = 0.0;
                for j in 0..d {
                    c += g[i * d + j] * w[j] * x[i * d + j];
                }
                c /= d as f32;
                for j in 0..d {
                    let xv = x[i * d + j];
                    let mut gr = g[i * d + j] * w[j] * inv;
                    gr -= c * inv3 * xv;
                    gx[i * d + j] = gr;
                    gw[j] += g[i * d + j] * xv * inv;
                }
            }
            push(0, gx);
            push(1, gw);
        }
        "gather" => {
            let (v, d) = (n.saved_u[0], n.saved_u[1]);
            let idx = &n.saved_u[2..];
            let mut ge = vec![0.0; v * d];
            for (t, &ix) in idx.iter().enumerate() {
                for j in 0..d {
                    ge[ix * d + j] += g[t * d + j];
                }
            }
            push(0, ge);
        }
        "ce" => {
            let logits = &n.saved_v[0];
            let targets = &n.saved_u;
            let s = n.saved_u.len();
            let v = logits.len() / s;
            let mut gl = vec![0.0; s * v];
            let scale = g[0] / s as f32;
            for i in 0..s {
                let row = &logits[i * v..(i + 1) * v];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for &x in row {
                    sum += (x - m).exp();
                }
                for j in 0..v {
                    let p = (row[j] - m).exp() / sum;
                    gl[i * v + j] = p * scale;
                }
                gl[i * v + targets[i]] -= scale;
            }
            push(0, gl);
        }
        "conv1d" => {
            let x = &n.saved_v[0];
            let w = &n.saved_v[1];
            let (s, d, k) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let mut gx = vec![0.0; s * d];
            let mut gw = vec![0.0; d * k];
            let mut gb = vec![0.0; d];
            for t in 0..s {
                for c in 0..d {
                    let go = g[t * d + c];
                    for u in 0..k {
                        let src = t as isize - (k as isize - 1) + u as isize;
                        if src >= 0 {
                            let si = src as usize;
                            gx[si * d + c] += go * w[c * k + u];
                            gw[c * k + u] += go * x[si * d + c];
                        }
                    }
                    gb[c] += go;
                }
            }
            push(0, gx);
            push(1, gw);
            push(2, gb);
        }
        "clockmem" => {
            let (s, d) = (n.saved_u[0], n.saved_u[1]);
            let beta = n.saved_f[0];
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[5];
            let s0 = &n.saved_v[6];
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0;
            let mut gs = vec![0.0; d];
            crate::prof::time(crate::prof::P::ClockBwd, || {
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                }
                for c in 0..d {
                    // At t=0 the previous state is whatever the previous
                    // window left behind, not zero. If this stayed at 0.0
                    // with persistent state, alpha's gradient would be
                    // wrong right at the window boundary -- and nothing
                    // would fail loudly.
                    let prev_state = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev_state;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            });
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
        }
        "clockmem_lowrank" => {
            // Backward of the low-rank outer-product write (see
            // ops::clockmem_lowrank_from). The element-wise half is
            // identical to "clockmem"; everything after `gm` is the new
            // path. The r x r carry `gm` plays exactly the role `gs` plays
            // for the element-wise state: it walks backwards in t picking up
            // each position's contribution and decaying by alpha_m.
            let (s, d, r) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let beta = n.saved_f[0];
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[5];
            let s0 = &n.saved_v[6];
            let kr = &n.saved_v[7];
            let vr = &n.saved_v[8];
            let qr = &n.saved_v[9];
            let mhist = &n.saved_v[10];
            let pk = &n.saved_v[11];
            let pv = &n.saved_v[12];
            let pq = &n.saved_v[13];
            let po = &n.saved_v[14];
            let am = &n.saved_v[15];

            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0f32;
            let mut gs = vec![0.0; d];
            let mut gpk = vec![0.0; d * r];
            let mut gpv = vec![0.0; d * r];
            let mut gpq = vec![0.0; d * r];
            let mut gpo = vec![0.0; r * d];
            let mut gam = vec![0.0; r];
            let mut gm = vec![0.0f32; r * r];

            crate::prof::time(crate::prof::P::ClockBwd, || {
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];

                // ---- element-wise path (unchanged from "clockmem") ----
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                }

                // ---- low-rank path ----
                // out[t,c] += sum_j rd[t,j]*po[j,c], and rd is not stored:
                // recompute it from qr and M[t] (r*r work, cheaper than
                // having saved s*r more numbers).
                let mt = &mhist[t * r * r..(t + 1) * r * r];
                let mut grd = vec![0.0f32; r];
                for j in 0..r {
                    let mut acc = 0.0f32;
                    for c in 0..d {
                        acc += gt[c] * po[j * d + c];
                    }
                    grd[j] = acc;
                }
                for j in 0..r {
                    let mut rdj = 0.0f32;
                    for i in 0..r {
                        rdj += qr[t * r + i] * mt[i * r + j];
                    }
                    if rdj != 0.0 {
                        for c in 0..d {
                            gpo[j * d + c] += rdj * gt[c];
                        }
                    }
                }
                // rd[t,j] = sum_i qr[t,i]*M[t][i,j]
                let mut gqr = vec![0.0f32; r];
                for i in 0..r {
                    let mut acc = 0.0f32;
                    for j in 0..r {
                        acc += grd[j] * mt[i * r + j];
                        gm[i * r + j] += qr[t * r + i] * grd[j];
                    }
                    gqr[i] = acc;
                }
                // M[t][i,j] = am[i]*M[t-1][i,j] + beta*kr[t,i]*vr[t,j]
                let mut gkr = vec![0.0f32; r];
                let mut gvr = vec![0.0f32; r];
                for i in 0..r {
                    let mut acc_kr = 0.0f32;
                    for j in 0..r {
                        let gmij = gm[i * r + j];
                        // At t=0 the previous M is zero (M is not carried
                        // across windows -- declared in the op's doc), so
                        // alpha_m gets no gradient there, which is correct.
                        let prev = if t == 0 { 0.0 } else { mhist[(t - 1) * r * r + i * r + j] };
                        gam[i] += gmij * prev;
                        acc_kr += gmij * beta * vr[t * r + j];
                        gvr[j] += gmij * beta * kr[t * r + i];
                        gb += gmij * kr[t * r + i] * vr[t * r + j];
                    }
                    gkr[i] = acc_kr;
                }
                // Projections down: kr/vr/qr = k/v/q times pk/pv/pq.
                for c in 0..d {
                    let (kc, vc, qc) = (k[t * d + c], v[t * d + c], q[t * d + c]);
                    let (mut ak, mut av, mut aq) = (0.0f32, 0.0f32, 0.0f32);
                    for i in 0..r {
                        gpk[c * r + i] += kc * gkr[i];
                        gpv[c * r + i] += vc * gvr[i];
                        gpq[c * r + i] += qc * gqr[i];
                        ak += gkr[i] * pk[c * r + i];
                        av += gvr[i] * pv[c * r + i];
                        aq += gqr[i] * pq[c * r + i];
                    }
                    gk[t * d + c] += ak;
                    gv[t * d + c] += av;
                    gq[t * d + c] += aq;
                }
                // Decay the r x r carry, same as gs[c] *= alpha[c] below.
                for i in 0..r {
                    let ai = am[i];
                    for j in 0..r {
                        gm[i * r + j] *= ai;
                    }
                }

                // ---- element-wise recurrence (unchanged) ----
                for c in 0..d {
                    let prev_state = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev_state;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            });
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
            push(6, gpk);
            push(7, gpv);
            push(8, gpq);
            push(9, gpo);
            push(10, gam);
        }
        "clockmem_gated" => {
            // Same graph as "clockmem", but the write magnitude is the
            // position-dependent beta_t = beta * r_t the forward saved (the
            // gate is detached, so r_t are constants here). Gradients only
            // differ in the extra factor r_t on the write path.
            let (s, d) = (n.saved_u[0], n.saved_u[1]);
            let beta = n.saved_f[0];
            let rs = &n.saved_f[1..];
            assert_eq!(rs.len(), s, "clockmem_gated saved {} r_t for {} positions", rs.len(), s);
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[5];
            let s0 = &n.saved_v[6];
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0;
            let mut gs = vec![0.0; d];
            crate::prof::time(crate::prof::P::ClockBwd, || {
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                }
                let beta_t = beta * rs[t];
                for c in 0..d {
                    let prev_state = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev_state;
                    gb += gs[c] * k[t * d + c] * v[t * d + c] * rs[t];
                    gk[t * d + c] += gs[c] * beta_t * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta_t * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            });
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
        }
        "clockmem_readwin" => {            let (s, d, kwin) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let beta = n.saved_f[0];
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let read = &n.saved_v[5];
            let alpha = &n.saved_v[6];
            let s0 = &n.saved_v[7];
            let w = &n.saved_v[8];
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0f32;
            let mut gw = vec![0.0; kwin * d];
            let mut gs = vec![0.0; d];
            // vals[t,c] = g[t,c]*q[t,c]*gg[t,c] -- reused by every j in the
            // window for both `direct` and `gw`.
            let mut vals = vec![0.0; s * d];
            for i in 0..s * d {
                vals[i] = g[i] * q[i] * gg[i];
            }
            let bands = crate::pool::global().workers() + 1;
            // direct[t,c] = sum_j vals[t+j,c]*w[j,c] (t+j < s), parallel over
            // t-bands with c inner for sequential/SIMD friendly access.
            let mut direct = vec![0.0; s * d];
            let (s_f, d_f, kwin_f) = (s, d, kwin);
            let vals_ptr = crate::pool::Ptr(vals.as_ptr());
            let w_ptr = crate::pool::Ptr(w.as_ptr());
            let direct_ptr = crate::pool::Ptr(direct.as_mut_ptr());
            crate::pool::global().run(bands, move |b| {
                let t0 = b * s_f / bands;
                let t1 = ((b + 1) * s_f / bands).min(s_f);
                let vals = vals_ptr.as_ref(s_f * d_f);
                let w = w_ptr.as_ref(kwin_f * d_f);
                let direct = direct_ptr.as_mut(s_f * d_f);
                for t in t0..t1 {
                    let jmax = kwin_f.min(s_f - t);
                    let toff = t * d_f;
                    for j in 0..jmax {
                        let woff = j * d_f;
                        let moff = (t + j) * d_f;
                        for c in 0..d_f {
                            direct[toff + c] += vals[moff + c] * w[woff + c];
                        }
                    }
                }
            });
            // gw[j,c] = sum_{t>=j} vals[t,c]*state[t-j,c], parallel over j-bands.
            let (s_f2, d_f2, kwin_f2) = (s, d, kwin);
            let vals_ptr2 = crate::pool::Ptr(vals.as_ptr());
            let state_ptr = crate::pool::Ptr(state.as_ptr());
            let gw_ptr = crate::pool::Ptr(gw.as_mut_ptr());
            crate::pool::global().run(bands, move |b| {
                let j0 = b * kwin_f2 / bands;
                let j1 = ((b + 1) * kwin_f2 / bands).min(kwin_f2);
                let vals = vals_ptr2.as_ref(s_f2 * d_f2);
                let state = state_ptr.as_ref(s_f2 * d_f2);
                let gw = gw_ptr.as_mut(kwin_f2 * d_f2);
                for j in j0..j1 {
                    let woff = j * d_f2;
                    for t in j..s_f2 {
                        let toff = t * d_f2;
                        let soff = (t - j) * d_f2;
                        for c in 0..d_f2 {
                            gw[woff + c] += vals[toff + c] * state[soff + c];
                        }
                    }
                }
            });
            for t in (0..s).rev() {
                for c in 0..d {
                    let go = g[t * d + c];
                    gq[t * d + c] += go * gg[t * d + c] * read[t * d + c];
                    ggg[t * d + c] += go * q[t * d + c] * read[t * d + c];
                    gs[c] += direct[t * d + c];
                }
                for c in 0..d {
                    let prev = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
            push(6, gw);
        }
        "concat" => {
            let (s, d) = (n.saved_u[0], n.saved_u[1]);
            for i in 0..s {
                let part = g[i * d..(i + 1) * d].to_vec();
                push(i, part);
            }
        }
        "stake_loss" => {
            // saved_v = [hidden, w], saved_f = [good (K), z (K)], saved_u =
            // [span_len, K, S]. The loss is mean_k (s_k - good_k)^2 with
            // s_k = sigmoid(z_k), so each span contributes
            //   d/dz_k = g[0] * 2/K * (s_k - good_k) * s_k * (1 - s_k)
            // and from there it's distributed to w (Sum_k d_k*h_k), b
            // (Sum_k d_k), and the hidden-state row that starts the span
            // (d_k*w).
            let hidden = &n.saved_v[0];
            let w = &n.saved_v[1];
            let span_len = n.saved_u[0];
            let k = n.saved_u[1];
            let s = n.saved_u[2];
            let d = hidden.len() / s;
            let good = &n.saved_f[..k];
            let zs = &n.saved_f[k..];
            let scale = g[0] * 2.0 / k.max(1) as f32;
            let mut gh = vec![0.0; hidden.len()];
            let mut gw = vec![0.0; d];
            let mut gb = 0.0;
            for (kk, &zk) in zs.iter().enumerate() {
                let sig = 1.0 / (1.0 + (-zk).exp());
                let der = scale * (sig - good[kk]) * sig * (1.0 - sig);
                let start = kk * span_len;
                for j in 0..d {
                    gh[start * d + j] += der * w[j];
                    gw[j] += der * hidden[start * d + j];
                }
                gb += der;
            }
            push(0, gh);
            push(1, gw);
            push(2, vec![gb]);
        }
        "clockmem_inject" => {
            let (s, d, nn) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let beta = n.saved_f[0];
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[5];
            let s0 = &n.saved_v[6];
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0f32;
            let mut gs = vec![0.0; d];
            // State-gradient contributions from the injected write_rec term:
            //   write_rec[t,c] = cur[t-N+1,c]/beta - alpha_c*cur[t-N,c]/beta
            //   -> state[t-N+1] gets +g[t]/beta, state[t-N] gets -g[t]*alpha/beta.
            let mut gs_extra = vec![0.0; s * d];
            for t in nn..s {
                let u = t - nn + 1;
                let w = t - nn;
                for c in 0..d {
                    let go = g[t * d + c];
                    gs_extra[u * d + c] += go / beta;
                    gs_extra[w * d + c] += -go * alpha[c] / beta;
                    // Explicit alpha dependence of the -alpha/beta tap.
                    ga[c] += go * (-state[w * d + c] / beta);
                    // Explicit beta dependence: d(write_rec)/d(beta) = -write_rec/beta.
                    let rec = (state[u * d + c] - alpha[c] * state[w * d + c]) / beta;
                    gb += go * (-rec / beta);
                }
            }
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                    gs[c] += gs_extra[t * d + c];
                }
                for c in 0..d {
                    let prev_state = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev_state;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
        }
        "clockmem_readwin_inj" => {
            let (s, d, kwin) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let beta = n.saved_f[0];
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[6];
            let s0 = &n.saved_v[7];
            let w = &n.saved_v[8];
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0f32;
            let mut gw = vec![0.0; kwin * d];
            let mut gs = vec![0.0; d];
            // direct[t,c] = go[t,c]*q[t,c]*gg[t,c]   (the q*cur*g term)
            //             + sum_j w[j,c]*go[t+j,c]   (the read term, t+j<s)
            let bands = crate::pool::global().workers() + 1;
            let mut direct = vec![0.0; s * d];
            for i in 0..s * d {
                direct[i] = g[i] * q[i] * gg[i];
            }
            let (s_f, d_f, kwin_f) = (s, d, kwin);
            let g_ptr = crate::pool::Ptr(g.as_ptr());
            let w_ptr = crate::pool::Ptr(w.as_ptr());
            let direct_ptr = crate::pool::Ptr(direct.as_mut_ptr());
            crate::pool::global().run(bands, move |b| {
                let t0 = b * s_f / bands;
                let t1 = ((b + 1) * s_f / bands).min(s_f);
                let g = g_ptr.as_ref(s_f * d_f);
                let w = w_ptr.as_ref(kwin_f * d_f);
                let direct = direct_ptr.as_mut(s_f * d_f);
                for t in t0..t1 {
                    let jmax = kwin_f.min(s_f - t);
                    let toff = t * d_f;
                    for j in 0..jmax {
                        let woff = j * d_f;
                        let goff = (t + j) * d_f;
                        for c in 0..d_f {
                            direct[toff + c] += w[woff + c] * g[goff + c];
                        }
                    }
                }
            });
            // gw[j,c] = sum_{t>=j} g[t,c]*state[t-j,c], parallel over j-bands.
            let g_ptr2 = crate::pool::Ptr(g.as_ptr());
            let state_ptr = crate::pool::Ptr(state.as_ptr());
            let gw_ptr = crate::pool::Ptr(gw.as_mut_ptr());
            crate::pool::global().run(bands, move |b| {
                let j0 = b * kwin_f / bands;
                let j1 = ((b + 1) * kwin_f / bands).min(kwin_f);
                let g = g_ptr2.as_ref(s_f * d_f);
                let state = state_ptr.as_ref(s_f * d_f);
                let gw = gw_ptr.as_mut(kwin_f * d_f);
                for j in j0..j1 {
                    let woff = j * d_f;
                    for t in j..s_f {
                        let toff = t * d_f;
                        let soff = (t - j) * d_f;
                        for c in 0..d_f {
                            gw[woff + c] += g[toff + c] * state[soff + c];
                        }
                    }
                }
            });
            for t in (0..s).rev() {
                for c in 0..d {
                    let go = g[t * d + c];
                    gq[t * d + c] += go * gg[t * d + c] * state[t * d + c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gs[c] += direct[t * d + c];
                }
                for c in 0..d {
                    let prev = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
            push(6, gw);
        }
        "clockmem_inject_learn" => {
            let (s, d, nn) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let beta = n.saved_f[0];
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[5];
            let s0 = &n.saved_v[6];
            let ar = &n.saved_v[7];
            let ib = &n.saved_v[8];
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0f32;
            let mut gar = vec![0.0; d];
            let mut gib = vec![0.0; d];
            let mut gs = vec![0.0; d];
            // read[t,c] = ib[c]*(state[t-N+1,c] - ar[c]*state[t-N,c])  (t>=N)
            //   d/state[t-N+1] = ib[c]      d/state[t-N] = -ib[c]*ar[c]
            //   d/ar[c] = -ib[c]*state[t-N,c]
            //   d/ib[c] = state[t-N+1,c] - ar[c]*state[t-N,c]
            // The physical alpha/beta do NOT appear in read (they only drive
            // the state recurrence), so their gradients come from the
            // recurrence alone -- unlike clockmem_inject, where write_rec
            // also depends on the physical alpha/beta.
            let mut gs_extra = vec![0.0; s * d];
            for t in nn..s {
                let u = t - nn + 1;
                let w = t - nn;
                for c in 0..d {
                    let go = g[t * d + c];
                    gs_extra[u * d + c] += go * ib[c];
                    gs_extra[w * d + c] += -go * ib[c] * ar[c];
                    gar[c] += go * (-ib[c] * state[w * d + c]);
                    gib[c] += go * (state[u * d + c] - ar[c] * state[w * d + c]);
                }
            }
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                    gs[c] += gs_extra[t * d + c];
                }
                for c in 0..d {
                    let prev_state = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev_state;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
            push(6, gar);
            push(7, gib);
        }
        "clockmem_inject_learn_g" => {
            let (s, d, nn) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let (beta, s_gate, sp, floor) = (n.saved_f[0], n.saved_f[1], n.saved_f[2], n.saved_f[3]);
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[5];
            let s0 = &n.saved_v[6];
            let ar = &n.saved_v[7];
            let ib = &n.saved_v[8];
            let eff = &n.saved_v[9]; // s_gate * read_uns (the effective contribution)
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0f32;
            let mut gar = vec![0.0; d];
            let mut gib = vec![0.0; d];
            let mut gz = 0.0f32;
            let mut gs = vec![0.0; d];
            // Memory-path terms, scaled by s_gate (>= floor > 0, never annulled):
            //   eff[t,c] = s_gate*ib[c]*(state[t-N+1,c] - ar[c]*state[t-N,c])
            //   d/state[t-N+1] = s_gate*ib[c]   d/state[t-N] = -s_gate*ib[c]*ar[c]
            //   d/ar[c] = -s_gate*ib[c]*state[t-N,c]
            //   d/ib[c] = s_gate*(state[t-N+1,c] - ar[c]*state[t-N,c])
            // Gate gradient: d(s)/dz = (1-floor)*sigmoid'(z) = (1-floor)*sp,
            // and d(out)/dz = read_uns*ds/dz = (eff/s_gate)*(1-floor)*sp.
            let mut gs_extra = vec![0.0; s * d];
            for t in nn..s {
                let u = t - nn + 1;
                let w = t - nn;
                for c in 0..d {
                    let go = g[t * d + c];
                    gs_extra[u * d + c] += go * s_gate * ib[c];
                    gs_extra[w * d + c] += -go * s_gate * ib[c] * ar[c];
                    gar[c] += go * s_gate * (-ib[c] * state[w * d + c]);
                    gib[c] += go * s_gate * (state[u * d + c] - ar[c] * state[w * d + c]);
                    gz += go * (eff[t * d + c] / s_gate) * (1.0 - floor) * sp;
                }
            }
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                    gs[c] += gs_extra[t * d + c];
                }
                for c in 0..d {
                    let prev_state = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev_state;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
            push(6, gar);
            push(7, gib);
            push(8, vec![gz]);
        }
        "clockmem_inject_learn_gc" => {
            let (s, d, nn) = (n.saved_u[0], n.saved_u[1], n.saved_u[2]);
            let (beta, floor) = (n.saved_f[0], n.saved_f[1]);
            let q = &n.saved_v[0];
            let k = &n.saved_v[1];
            let v = &n.saved_v[2];
            let gg = &n.saved_v[3];
            let state = &n.saved_v[4];
            let alpha = &n.saved_v[5];
            let s0 = &n.saved_v[6];
            let ar = &n.saved_v[7];
            let ib = &n.saved_v[8];
            let eff = &n.saved_v[9]; // s[c]*read_uns (the effective contribution)
            let sg = &n.saved_v[10]; // s[c] per channel
            let sp = &n.saved_v[11]; // sigmoid'(z[c]) per channel
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0f32;
            let mut gar = vec![0.0; d];
            let mut gib = vec![0.0; d];
            let mut gz = vec![0.0; d];
            let mut gs = vec![0.0; d];
            // Same as the scalar-gate backward, with s[c] in place of the
            // block scalar: the floor guarantee holds per channel, so every
            // channel's memory path keeps real gradient.
            let mut gs_extra = vec![0.0; s * d];
            for t in nn..s {
                let u = t - nn + 1;
                let w = t - nn;
                for c in 0..d {
                    let go = g[t * d + c];
                    gs_extra[u * d + c] += go * sg[c] * ib[c];
                    gs_extra[w * d + c] += -go * sg[c] * ib[c] * ar[c];
                    gar[c] += go * sg[c] * (-ib[c] * state[w * d + c]);
                    gib[c] += go * sg[c] * (state[u * d + c] - ar[c] * state[w * d + c]);
                    gz[c] += go * (eff[t * d + c] / sg[c]) * (1.0 - floor) * sp[c];
                }
            }
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                    gs[c] += gs_extra[t * d + c];
                }
                for c in 0..d {
                    let prev_state = if t == 0 { s0[c] } else { state[(t - 1) * d + c] };
                    ga[c] += gs[c] * prev_state;
                    gb += gs[c] * k[t * d + c] * v[t * d + c];
                    gk[t * d + c] += gs[c] * beta * v[t * d + c];
                    gv[t * d + c] += gs[c] * beta * k[t * d + c];
                }
                for c in 0..d {
                    gs[c] *= alpha[c];
                }
            }
            push(0, gq);
            push(1, gk);
            push(2, gv);
            push(3, ggg);
            push(4, ga);
            push(5, vec![gb]);
            push(6, gar);
            push(7, gib);
            push(8, gz);
        }
        _ => panic!("backward not implemented for op {}", n.op),
    }
    out
}
