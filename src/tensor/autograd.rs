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
        _ => panic!("backward not implemented for op {}", n.op),
    }
    out
}
