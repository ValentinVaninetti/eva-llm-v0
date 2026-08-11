use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::tensor::Tensor;

pub struct Input {
    pub id: usize,
    pub len: usize,
    pub node: Option<Arc<Node>>,
}

pub struct Node {
    pub op: &'static str,
    pub inputs: Vec<Input>,
    pub grad_flags: Vec<bool>,
    pub saved_v: Vec<Vec<f32>>,
    pub saved_f: Vec<f32>,
    pub saved_u: Vec<usize>,
    pub out_id: usize,
    pub out_len: usize,
}

pub fn make_node(
    op: &'static str,
    inputs: &[&Tensor],
    grad_flags: Vec<bool>,
    saved_v: Vec<Vec<f32>>,
    saved_f: Vec<f32>,
    saved_u: Vec<usize>,
    out_id: usize,
    out_len: usize,
) -> Arc<Node> {
    let inputs: Vec<Input> = inputs
        .iter()
        .map(|t| Input { id: t.id, len: t.numel(), node: t.node.clone() })
        .collect();
    Arc::new(Node {
        op,
        inputs,
        grad_flags,
        saved_v,
        saved_f,
        saved_u,
        out_id,
        out_len,
    })
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

fn index_to_coords(i: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut rem = i;
    for d in (0..shape.len()).rev() {
        coords[d] = rem % shape[d];
        rem /= shape[d];
    }
    coords
}

fn coords_to_index(c: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    for d in 0..shape.len() {
        idx = idx * shape[d] + c[d];
    }
    idx
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

fn broadcast_to(x: &[f32], xshape: &[usize], oshape: &[usize]) -> Vec<f32> {
    if xshape == oshape {
        return x.to_vec();
    }
    let n = oshape.iter().product::<usize>();
    let mut out = vec![0.0; n];
    let off = oshape.len() - xshape.len();
    for oidx in 0..n {
        let oc = index_to_coords(oidx, oshape);
        let mut xc = vec![0; xshape.len()];
        let mut ok = true;
        for d in 0..xshape.len() {
            let od = off + d;
            if xshape[d] == oshape[od] {
                xc[d] = oc[od];
            } else if xshape[d] == 1 {
                xc[d] = 0;
            } else {
                ok = false;
            }
        }
        if ok {
            out[oidx] = x[coords_to_index(&xc, xshape)];
        }
    }
    out
}

fn reduce(g: &[f32], out_shape: &[usize], to_shape: &[usize]) -> Vec<f32> {
    if out_shape == to_shape {
        return g.to_vec();
    }
    let mut out = vec![0.0; to_shape.iter().product()];
    let off = out_shape.len() - to_shape.len();
    for oidx in 0..g.len() {
        let oc = index_to_coords(oidx, out_shape);
        let mut tc = vec![0; to_shape.len()];
        let mut ok = true;
        for d in 0..to_shape.len() {
            let od = off + d;
            if to_shape[d] == out_shape[od] {
                tc[d] = oc[od];
            } else if to_shape[d] == 1 {
                tc[d] = 0;
            } else {
                ok = false;
            }
        }
        if ok {
            out[coords_to_index(&tc, to_shape)] += g[oidx];
        }
    }
    out
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
            let mut gq = vec![0.0; s * d];
            let mut gk = vec![0.0; s * d];
            let mut gv = vec![0.0; s * d];
            let mut ggg = vec![0.0; s * d];
            let mut ga = vec![0.0; d];
            let mut gb = 0.0;
            let mut gs = vec![0.0; d];
            for t in (0..s).rev() {
                let gt = &g[t * d..(t + 1) * d];
                for c in 0..d {
                    let go = gt[c];
                    ggg[t * d + c] += go * q[t * d + c] * state[t * d + c];
                    gq[t * d + c] += go * state[t * d + c] * gg[t * d + c];
                    gs[c] += go * q[t * d + c] * gg[t * d + c];
                }
                for c in 0..d {
                    let prev_state = if t == 0 { 0.0 } else { state[(t - 1) * d + c] };
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
        "concat" => {
            let (s, d) = (n.saved_u[0], n.saved_u[1]);
            for i in 0..s {
                let part = g[i * d..(i + 1) * d].to_vec();
                push(i, part);
            }
        }
        _ => panic!("backward not implemented for op {}", n.op),
    }
    out
}
