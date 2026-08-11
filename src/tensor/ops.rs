use crate::tensor::autograd::make_node;
use crate::tensor::{next_id, Tensor};

fn finalize(
    inputs: &[&Tensor],
    op: &'static str,
    data: Vec<f32>,
    shape: Vec<usize>,
    saved_v: Vec<Vec<f32>>,
    saved_f: Vec<f32>,
    saved_u: Vec<usize>,
) -> Tensor {
    let needs = inputs.iter().any(|t| t.requires_grad);
    let id = next_id();
    if needs {
        let grad_flags: Vec<bool> = inputs.iter().map(|t| t.requires_grad).collect();
        let node = make_node(op, inputs, grad_flags, saved_v, saved_f, saved_u, id, data.len());
        Tensor { data, shape, id, requires_grad: true, node: Some(node) }
    } else {
        Tensor { data, shape, id, requires_grad: false, node: None }
    }
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

/// Softmax por fila sobre el prefijo causal: la fila `i` sólo ve las columnas
/// `0..=i`, y el resto queda en cero.
///
/// El enmascarado va ACÁ ADENTRO y no como un `-inf` sumado antes, por dos
/// razones. Numérica: `exp(-inf)` en el borde da NaN apenas alguien resta el
/// máximo. Y de costo: enmascarar afuera obliga a materializar una matriz SxS
/// de `-inf` por capa y por paso, que es memoria y tráfico para representar
/// "acá no mires".
pub fn softmax_causal(x: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 2, "softmax_causal expects (S,S)");
    let (s, n) = (x.shape[0], x.shape[1]);
    assert_eq!(s, n, "softmax_causal expects a square score matrix");
    let mut out = vec![0.0; s * s];
    for i in 0..s {
        let row = &x.data[i * s..i * s + i + 1];
        // Restar el máximo antes de exponenciar: sin esto, logits grandes
        // desbordan a inf y la fila entera sale NaN.
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
    finalize(&[x], "softmax_causal", out.clone(), x.shape.clone(), vec![out], vec![], vec![s])
}

/// Las primeras `n` filas de un tensor (S,D).
///
/// Hace falta porque en generación la ventana crece de a un token y la tabla
/// de posiciones es de largo fijo. Cortarla con `Tensor::new` parecía
/// equivalente y no lo es: un tensor construido a mano no tiene nodo, así que
/// el gradiente nunca vuelve y la tabla no aprende nada -- falla en silencio,
/// entrenando un parámetro muerto.
pub fn slice_rows(x: &Tensor, n: usize) -> Tensor {
    assert_eq!(x.shape.len(), 2, "slice_rows expects (S,D)");
    let (s, d) = (x.shape[0], x.shape[1]);
    assert!(n <= s, "slice_rows: {n} filas de un tensor de {s}");
    finalize(&[x], "slice_rows", x.data[..n * d].to_vec(), vec![n, d], vec![], vec![], vec![s, d, n])
}

pub fn silu(x: &Tensor) -> Tensor {
    let out: Vec<f32> = x.data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    finalize(&[x], "silu", out, x.shape.clone(), vec![x.data.clone()], vec![], vec![])
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    let out: Vec<f32> = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    finalize(&[x], "sigmoid", out.clone(), x.shape.clone(), vec![out], vec![], vec![])
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

/// Igual, pero arrancando de un estado dado y devolviendo el estado final.
///
/// POR QUÉ EXISTE: el estado de ClockMem es de tamaño fijo y tiene su propio
/// olvido por canal, así que **no hay ninguna razón técnica para reiniciarlo
/// en cada ventana**. Reiniciar es una herencia del transformer, donde el
/// contexto ES la ventana y no queda otra. Acá, dejarlo correr da memoria más
/// allá de la ventana **sin un byte extra de costo**.
///
/// `s0` entra como CONSTANTE: el gradiente no vuelve por ahí. Eso es
/// truncated BPTT, y es a propósito -- propagar hacia atrás por todo el corpus
/// significaría sostener el grafo de todo el corpus, que es exactamente lo que
/// no se quiere.
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
            state,
            alpha.data.clone(),
            s0.to_vec(),
        ],
        vec![beta_v],
        vec![s, d],
    );
    (t, final_state)
}
