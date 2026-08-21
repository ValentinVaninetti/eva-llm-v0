use crate::nn::param;
use crate::tensor::autograd::backward;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

fn numeric(t: &mut Tensor, f: impl Fn(&Tensor) -> f32, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; t.data.len()];
    for i in 0..t.data.len() {
        // `make_mut` because the buffer is now shared with the graph. Here
        // it's fine if it copies the first time: this is gradcheck, not the
        // hot path.
        let orig = t.data[i];
        std::sync::Arc::make_mut(&mut t.data)[i] = orig + eps;
        let fp = f(t);
        std::sync::Arc::make_mut(&mut t.data)[i] = orig - eps;
        let fm = f(t);
        std::sync::Arc::make_mut(&mut t.data)[i] = orig;
        out[i] = (fp - fm) / (2.0 * eps);
    }
    out
}

fn check(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "len mismatch in {}", name);
    for i in 0..a.len() {
        let err = (a[i] - b[i]).abs();
        let scale = (a[i].abs() + b[i].abs()).max(1.0);
        assert!(err / scale < 1e-3, "{}[{}]: analytic={} numeric={} err={}", name, i, a[i], b[i], err);
    }
}

fn gradcheck_unary(name: &str, x: &mut Tensor, forward: impl Fn(&Tensor) -> Tensor) {
    let out = forward(x);
    let grads = backward(&ops::sum_all(&out));
    let g = grads.get(&x.id).cloned().unwrap();
    let n = numeric(x, |p| sum_of(&forward(p)), 1e-3);
    check(&g, &n, name);
}

fn gradcheck_binary(name: &str, a: &mut Tensor, b: &mut Tensor, forward: impl Fn(&Tensor, &Tensor) -> Tensor) {
    let out = forward(a, b);
    let grads = backward(&ops::sum_all(&out));
    let ga = grads.get(&a.id).cloned().unwrap();
    let gb = grads.get(&b.id).cloned().unwrap();
    let na = numeric(a, |p| sum_of(&forward(p, &b.detach())), 1e-3);
    let nb = numeric(b, |p| sum_of(&forward(&a.detach(), p)), 1e-3);
    check(&ga, &na, name);
    check(&gb, &nb, name);
}

#[test]
fn gradcheck_matmul_ce() {
    let mut a = param(vec![0.14, 0.29, 0.43, 0.57, 0.71, 0.86], vec![2, 3]);
    let mut w = param(vec![0.17, 0.28, 0.39, 0.5, 0.61, 0.72], vec![3, 2]);
    let targets = vec![1usize, 0];

    let logits = ops::matmul(&a, &w);
    let loss = ops::cross_entropy(&logits, &targets);
    let grads = backward(&loss);
    let ga = grads.get(&a.id).cloned().unwrap();
    let gw = grads.get(&w.id).cloned().unwrap();

    let na = numeric(&mut a, |p| loss_of(p, &w, &targets), 1e-3);
    let nw = numeric(&mut w, |p| loss_of(&a, p, &targets), 1e-3);
    check(&ga, &na, "a");
    check(&gw, &nw, "w");
}

#[test]
fn gradcheck_rms_norm() {
    let mut a = param(vec![0.5, -1.2, 0.3, 0.9, -0.4, 0.7, 1.1, -0.2, 0.0, 0.4, -0.8, 0.2], vec![3, 4]);
    let mut w = param(vec![1.1, 0.9, 1.0, 1.2], vec![4]);
    let eps = 1e-5;

    let out = ops::rms_norm(&a, &w, eps);
    let loss = ops::sum_all(&out);
    let grads = backward(&loss);
    let ga = grads.get(&a.id).cloned().unwrap();
    let gw = grads.get(&w.id).cloned().unwrap();

    let na = numeric(&mut a, |p| sum_of(&ops::rms_norm(p, &w.detach(), eps)), 1e-3);
    let nw = numeric(&mut w, |p| sum_of(&ops::rms_norm(&a.detach(), p, eps)), 1e-3);
    check(&ga, &na, "a");
    check(&gw, &nw, "w");
}

#[test]
fn gradcheck_conv1d() {
    let mut x = param(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], vec![6, 2]);
    let mut w = param(vec![0.5, -0.2, 0.1, 0.3, 0.4, 0.1], vec![2, 3]);
    let mut b = param(vec![0.05, -0.05], vec![2]);
    let k = 3;

    let out = ops::depthwise_conv1d(&x, &w, &b, k);
    let loss = ops::sum_all(&out);
    let grads = backward(&loss);
    let gx = grads.get(&x.id).cloned().unwrap();
    let gw = grads.get(&w.id).cloned().unwrap();
    let gb = grads.get(&b.id).cloned().unwrap();

    let nx = numeric(&mut x, |p| sum_of(&ops::depthwise_conv1d(p, &w.detach(), &b.detach(), k)), 1e-3);
    let nw = numeric(&mut w, |p| sum_of(&ops::depthwise_conv1d(&x.detach(), p, &b.detach(), k)), 1e-3);
    let nb = numeric(&mut b, |p| sum_of(&ops::depthwise_conv1d(&x.detach(), &w.detach(), p, k)), 1e-3);
    check(&gx, &nx, "x");
    check(&gw, &nw, "w");
    check(&gb, &nb, "b");
}

#[test]
fn gradcheck_clockmem() {
    let s = 5;
    let d = 3;
    let data: Vec<f32> = (0..s * d).map(|i| (i as f32 - 7.0) / 10.0).collect();
    let mut q = param(data.clone(), vec![s, d]);
    let mut k = param((0..s * d).map(|i| (i as f32 * 1.7 - 4.0) / 11.0).collect(), vec![s, d]);
    let mut v = param((0..s * d).map(|i| (i as f32 * -0.9 + 2.0) / 8.0).collect(), vec![s, d]);
    let mut g = param((0..s * d).map(|i| ((i % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.1, 0.2, 0.3], vec![d]);
    let mut beta = param(vec![0.9], vec![1]);

    let out = ops::clockmem(&q, &k, &v, &g, &alpha, &beta);
    let loss = ops::sum_all(&out);
    let grads = backward(&loss);
    let gq = grads.get(&q.id).cloned().unwrap();
    let gk = grads.get(&k.id).cloned().unwrap();
    let gv = grads.get(&v.id).cloned().unwrap();
    let gg = grads.get(&g.id).cloned().unwrap();
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();

    let nq = numeric(&mut q, |p| cm_sum(p, &k, &v, &g, &alpha, &beta), 1e-3);
    let nk = numeric(&mut k, |p| cm_sum(&q, p, &v, &g, &alpha, &beta), 1e-3);
    let nv = numeric(&mut v, |p| cm_sum(&q, &k, p, &g, &alpha, &beta), 1e-3);
    let ng = numeric(&mut g, |p| cm_sum(&q, &k, &v, p, &alpha, &beta), 1e-3);
    let na = numeric(&mut alpha, |p| cm_sum(&q, &k, &v, &g, p, &beta), 1e-3);
    let nb = numeric(&mut beta, |p| cm_sum(&q, &k, &v, &g, &alpha, p), 1e-3);
    check(&gq, &nq, "q");
    check(&gk, &nk, "k");
    check(&gv, &nv, "v");
    check(&gg, &ng, "g");
    check(&ga, &na, "alpha");
    check(&gb, &nb, "beta");
}

fn loss_of(a: &Tensor, w: &Tensor, targets: &[usize]) -> f32 {
    let l = ops::cross_entropy(&ops::matmul(&a.detach(), &w.detach()), targets);
    l.data[0]
}

#[test]
fn gradcheck_add_mul_broadcast() {
    let mut a = param(vec![0.5, -1.3, 0.7, 0.2, 0.9, -0.4], vec![2, 3]);
    let mut b = param(vec![0.3, 0.8, -0.2], vec![3]);
    gradcheck_binary("add", &mut a, &mut b, |x, y| ops::add(x, y));
    gradcheck_binary("mul", &mut a, &mut b, |x, y| ops::mul(x, y));
}

#[test]
fn gradcheck_scale_silu_sigmoid_sum() {
    let mut x = param(vec![0.5, -1.3, 0.7, 0.2, 0.9, -0.4], vec![2, 3]);
    gradcheck_unary("scale", &mut x, |t| ops::scale(t, 2.5));
    gradcheck_unary("silu", &mut x, |t| ops::silu(t));
    gradcheck_unary("sigmoid", &mut x, |t| ops::sigmoid(t));
    gradcheck_unary("algebraic_sigmoid", &mut x, |t| ops::algebraic_sigmoid(t));
    gradcheck_unary("sum_all", &mut x, |t| ops::sum_all(t));
}

#[test]
fn gradcheck_gather() {
    let mut emb = param(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        vec![4, 3],
    );
    let idx = vec![2usize, 0, 3, 1, 0];
    let out = ops::gather(&emb, &idx);
    let grads = backward(&ops::sum_all(&out));
    let g = grads.get(&emb.id).cloned().unwrap();
    let n = numeric(&mut emb, |p| sum_of(&ops::gather(p, &idx)), 1e-3);
    check(&g, &n, "emb");
}

fn sum_of(t: &Tensor) -> f32 {
    t.data.iter().sum()
}

#[test]
fn gradcheck_stake_loss() {
    // The stake head: analytic derivatives against numeric ones on all
    // three inputs. If the backward were wrong nothing would fail during
    // training, it would just learn worse -- and the number #8 promises
    // wouldn't be worth anything.
    let (s, d, span) = (8usize, 4usize, 4usize);
    let mut hidden = param((0..s * d).map(|i| 0.3 * (i % 5) as f32 - 0.4).collect(), vec![s, d]);
    let mut w = param(vec![0.2, -0.3, 0.5, 0.1], vec![d]);
    let mut b = param(vec![0.7], vec![1]);
    let good = vec![0.5, 0.25];

    let loss = ops::stake_loss(&hidden, &w, &b, &good, span);
    let grads = backward(&loss);
    let gh = grads.get(&hidden.id).cloned().unwrap();
    let gw = grads.get(&w.id).cloned().unwrap();
    let gb = grads.get(&b.id).cloned().unwrap();

    let f = |h: &Tensor, ww: &Tensor, bb: &Tensor| sum_of(&ops::stake_loss(h, ww, bb, &good, span));
    let nh = numeric(&mut hidden, |h| f(h, &w, &b), 1e-3);
    let nw = numeric(&mut w, |ww| f(&hidden, ww, &b), 1e-3);
    let nb = numeric(&mut b, |bb| f(&hidden, &w, bb), 1e-3);
    check(&gh, &nh, "hidden");
    check(&gw, &nw, "w");
    check(&gb, &nb, "b");
}

fn cm_sum(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, alpha: &Tensor, beta: &Tensor) -> f32 {
    let t = ops::clockmem(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach());
    t.data.iter().sum()
}

#[test]
fn gradcheck_transpose_and_causal_softmax() {
    // The two pieces needed to build the reference attention. Without
    // gradcheck they don't get in: an attention with a wrong backward
    // doesn't fail, it just learns worse, and then the comparison with
    // ClockMem lies in our favor.
    let mut x = param(vec![0.3, -1.2, 0.7, 2.1, -0.5, 0.9], vec![2, 3]);
    gradcheck_unary("transpose", &mut x, ops::transpose);

    let mut s = param(
        vec![0.5, -0.3, 1.1, 0.2, -1.4, 0.8, 0.05, 1.7, -0.6, 0.33, 2.0, -0.9, 1.2, 0.4, -0.2, 0.65],
        vec![4, 4],
    );
    gradcheck_unary("softmax_causal", &mut s, ops::softmax_causal);
}

#[test]
fn causal_softmax_does_not_look_ahead() {
    // The property that makes attention causal. If this fails, the model
    // reads the future and its loss is a beautiful lie.
    let x = param((0..16).map(|i| (i as f32) * 0.37 - 2.0).collect(), vec![4, 4]);
    let y = ops::softmax_causal(&x);
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_eq!(0.0, y.data[i * 4 + j], "row {i} looked at column {j}");
        }
        let row: f32 = (0..=i).map(|j| y.data[i * 4 + j]).sum();
        assert!((row - 1.0).abs() < 1e-6, "row {i} does not sum to 1: {row}");
    }
}

#[test]
fn gradcheck_slice_rows() {
    // The gradient has to reach the sliced rows and NOT the ones below.
    // Without this, the position table trains silently against nothing.
    let mut x = param((0..12).map(|i| (i as f32) * 0.3 - 1.0).collect(), vec![4, 3]);
    gradcheck_unary("slice_rows", &mut x, |t| ops::slice_rows(t, 2));

    let g = {
        let y = ops::slice_rows(&x, 2);
        backward(&ops::sum_all(&y)).get(&x.id).cloned().unwrap()
    };
    assert_eq!(vec![0.0; 6], g[6..].to_vec(), "gradient reached rows that were not used");
}

#[test]
fn gradcheck_clockmem_with_carried_state() {
    // The case the usual gradcheck does NOT cover: starting from a state
    // carried over from the previous window. At t=0 the "previous state"
    // is no longer zero, and alpha's gradient depends on that. If that were
    // wrong, alpha would learn with a bias at the boundary of each window
    // -- and nothing would fail: the model would still train, just worse,
    // without saying why.
    let (s, d) = (5, 3);
    let mut q = param((0..s * d).map(|i| (i as f32 - 7.0) / 10.0).collect(), vec![s, d]);
    let mut k = param((0..s * d).map(|i| (i as f32 * 1.7 - 4.0) / 11.0).collect(), vec![s, d]);
    let mut v = param((0..s * d).map(|i| (i as f32 * -0.9 + 2.0) / 8.0).collect(), vec![s, d]);
    let mut g = param((0..s * d).map(|i| ((i % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.35, 0.6, 0.15], vec![d]);
    let mut beta = param(vec![0.9], vec![1]);
    // An initial state clearly nonzero: if the backward ignored it, the
    // difference would show up right here.
    let s0 = vec![0.8, -1.3, 0.45];

    let run = |a: &Tensor, b: &Tensor, qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor| {
        ops::clockmem_from(qq, kk, vv, gg, a, b, &s0).0
    };

    let out = run(&alpha, &beta, &q, &k, &v, &g);
    let grads = backward(&ops::sum_all(&out));
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gq = grads.get(&q.id).cloned().unwrap();

    let na = numeric(&mut alpha, |p| {
        sum_of(&ops::clockmem_from(&q.detach(), &k.detach(), &v.detach(), &g.detach(), p, &beta.detach(), &s0).0)
    }, 1e-3);
    let nb = numeric(&mut beta, |p| {
        sum_of(&ops::clockmem_from(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), p, &s0).0)
    }, 1e-3);
    let nq = numeric(&mut q, |p| {
        sum_of(&ops::clockmem_from(p, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &s0).0)
    }, 1e-3);

    check(&ga, &na, "alpha with carried state");
    check(&gb, &nb, "beta with carried state");
    check(&gq, &nq, "q with carried state");
}

#[test]
fn gradcheck_clockmem_gated_write() {
    // The gate is DETACHED by design: r_t is a constant for the backward, so
    // the backward is exactly the standard clockmem backward with the
    // per-position write magnitudes beta_t = beta*r_t. The numeric check
    // therefore uses the SAME constant-beta_t function (r_t computed once
    // from the baseline inputs): against the full forward, alpha/k/v would
    // differ through the gate, which is the design, not a bug. p=1.0
    // exercises a non-trivial gate and the state is carried over.
    let (s, d) = (5, 3);
    let p = 1.0;
    let s0 = vec![0.8, -1.3, 0.45];
    let mut q = param((0..s * d).map(|i| (i as f32 - 7.0) / 10.0).collect(), vec![s, d]);
    let mut k = param((0..s * d).map(|i| (i as f32 * 1.7 - 4.0) / 11.0).collect(), vec![s, d]);
    let mut v = param((0..s * d).map(|i| (i as f32 * -0.9 + 2.0) / 8.0).collect(), vec![s, d]);
    let mut g = param((0..s * d).map(|i| ((i % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.35, 0.6, 0.15], vec![d]);
    let mut beta = param(vec![0.9], vec![1]);

    let beta0 = beta.data[0];
    ops::write_trace_reset();
    let _ = ops::clockmem_gated_from(&q, &k, &v, &g, &alpha, &beta, p, &s0, true);
    let tr = ops::write_trace_take();
    assert_eq!(tr.len(), 2 * s, "gate trace should be one (e_t, r_t) pair per position");
    let rt: Vec<f32> = tr.chunks(2).map(|p| p[1]).collect();

    // The reference the backward implements: the same recurrence with the
    // per-position write magnitudes held CONSTANT at beta*rt[t].
    let ref_fwd = |qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor, aa: &Tensor, bb: f32| -> f32 {
        let mut cur = s0.clone();
        let mut sum = 0.0;
        for t in 0..s {
            let bt = bb * rt[t];
            for c in 0..d {
                cur[c] = aa.data[c] * cur[c] + bt * kk.data[t * d + c] * vv.data[t * d + c];
            }
            for c in 0..d {
                sum += qq.data[t * d + c] * cur[c] * gg.data[t * d + c];
            }
        }
        sum
    };

    let out = ops::clockmem_gated_from(&q, &k, &v, &g, &alpha, &beta, p, &s0, false).0;
    let grads = backward(&ops::sum_all(&out));
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gq = grads.get(&q.id).cloned().unwrap();
    let gk = grads.get(&k.id).cloned().unwrap();
    let gv = grads.get(&v.id).cloned().unwrap();
    let gg2 = grads.get(&g.id).cloned().unwrap();

    let na = numeric(&mut alpha, |aa| ref_fwd(&q.detach(), &k.detach(), &v.detach(), &g.detach(), aa, beta0), 1e-3);
    let nb = numeric(&mut beta, |bb| ref_fwd(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), bb.data[0]), 1e-3);
    let nq = numeric(&mut q, |qq| ref_fwd(qq, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), beta0), 1e-3);
    let nk = numeric(&mut k, |kk| ref_fwd(&q.detach(), kk, &v.detach(), &g.detach(), &alpha.detach(), beta0), 1e-3);
    let nv = numeric(&mut v, |vv| ref_fwd(&q.detach(), &k.detach(), vv, &g.detach(), &alpha.detach(), beta0), 1e-3);
    let ng = numeric(&mut g, |gg| ref_fwd(&q.detach(), &k.detach(), &v.detach(), gg, &alpha.detach(), beta0), 1e-3);

    check(&ga, &na, "alpha gated");
    check(&gb, &nb, "beta gated");
    check(&gq, &nq, "q gated");
    check(&gk, &nk, "k gated");
    check(&gv, &nv, "v gated");
    check(&gg2, &ng, "g gated");
}

#[test]
fn gated_write_with_p0_is_plain_clockmem() {
    // p=0 disables the gate (r_t = clamp((e/e_ref)^0, 0, 1) = 1): the gated
    // forward must reproduce the plain recurrence exactly, or "only the write
    // magnitude changes" is false at its own neutral point.
    let (s, d) = (4, 3);
    let s0 = vec![0.2, -0.5, 0.8];
    let q = param((0..s * d).map(|i| i as f32 * 0.5 - 1.0).collect(), vec![s, d]);
    let k = param((0..s * d).map(|i| i as f32 * -0.3 + 0.7).collect(), vec![s, d]);
    let v = param((0..s * d).map(|i| i as f32 * 0.9 - 0.4).collect(), vec![s, d]);
    let g = param((0..s * d).map(|i| (i % 2) as f32 - 0.5).collect(), vec![s, d]);
    let alpha = param(vec![0.4, 0.7, 0.2], vec![d]);
    let beta = param(vec![0.6], vec![1]);

    let (gated, _) = ops::clockmem_gated_from(&q, &k, &v, &g, &alpha, &beta, 0.0, &s0, false);
    let (plain, _) = ops::clockmem_from(&q, &k, &v, &g, &alpha, &beta, &s0);
    for i in 0..s * d {
        assert!((gated.data[i] - plain.data[i]).abs() < 1e-6, "p=0 gate changed output at {i}");
    }
}

#[test]
fn write_trace_records_gate_multipliers() {
    // The trace the measurement probe reads: one r_t per position, in call
    // order, all in [0,1] by construction of the clamp.
    ops::write_trace_reset();
    let (s, d) = (4, 3);
    let s0 = vec![0.0; d];
    let q = param((0..s * d).map(|i| i as f32 * 0.3 - 1.0).collect(), vec![s, d]);
    let k = param((0..s * d).map(|i| i as f32 * 0.7 + 0.1).collect(), vec![s, d]);
    let v = param((0..s * d).map(|i| i as f32 * -0.2 + 0.5).collect(), vec![s, d]);
    let g = param(vec![0.5; s * d], vec![s, d]);
    let alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let beta = param(vec![0.7], vec![1]);

    let _ = ops::clockmem_gated_from(&q, &k, &v, &g, &alpha, &beta, 1.0, &s0, true);
    let tr = ops::write_trace_take();
    assert_eq!(tr.len(), 2 * s, "expected one (e_t, r_t) pair per position");
    for (t, pair) in tr.chunks(2).enumerate() {
        let (e, r) = (pair[0], pair[1]);
        assert!(e >= 0.0, "e_{t}={e} negative");
        assert!((0.0..=1.0).contains(&r), "r_{t}={r} out of range");
    }
}

#[test]
fn gradcheck_clockmem_readwin() {
    // The windowed read must (a) reduce to the plain clockmem when K=1 with
    // w=1 on the current state, and (b) have a correct backward in general.
    // A wrong backward here would make the read window train against
    // nothing and the experiment would silently collapse to "empty clock".
    let (s, d, kwin) = (5, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let mut q = mk(1.3, -2.0);
    let mut k = mk(0.6, 1.0);
    let mut v = mk(-0.8, 0.5);
    let mut g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let mut beta = param(vec![0.7], vec![1]);
    let mut w = param(vec![1.0, 0.2, -0.4, 1.0, 0.1, 0.3, 1.0, -0.5, 0.6], vec![kwin, d]);
    let s0 = vec![0.4, -0.9, 0.25];

    let run = |qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor, a: &Tensor, b: &Tensor, ww: &Tensor| {
        ops::clockmem_readwin(qq, kk, vv, gg, a, b, ww, &s0).0
    };

    // K=1 with delta weights must equal the plain single-state read.
    let w1 = param(vec![1.0, 1.0, 1.0], vec![1, d]);
    let out_w1 = run(&q, &k, &v, &g, &alpha, &beta, &w1);
    let out_plain = ops::clockmem_from(&q, &k, &v, &g, &alpha, &beta, &s0).0;
    for i in 0..s * d {
        assert!(
            (out_w1.data[i] - out_plain.data[i]).abs() < 1e-5,
            "K=1 delta mismatch at {i}: {} vs {}",
            out_w1.data[i], out_plain.data[i]
        );
    }

    let out = run(&q, &k, &v, &g, &alpha, &beta, &w);
    let grads = backward(&ops::sum_all(&out));
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gq = grads.get(&q.id).cloned().unwrap();
    let gw = grads.get(&w.id).cloned().unwrap();

    let na = numeric(&mut alpha, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), p, &beta.detach(), &w.detach()))
    }, 1e-3);
    let nb = numeric(&mut beta, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), p, &w.detach()))
    }, 1e-3);
    let nq = numeric(&mut q, |p| {
        sum_of(&run(p, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &w.detach()))
    }, 1e-3);
    let nw = numeric(&mut w, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), p))
    }, 1e-3);

    check(&ga, &na, "alpha (windowed)");
    check(&gb, &nb, "beta (windowed)");
    check(&gq, &nq, "q (windowed)");
    check(&gw, &nw, "w (windowed)");
}

#[test]
fn gradcheck_clockmem_inject() {
    // Clean injection: out = q*cur*g + recovered write. The recovered write
    // enters the residual without q*g, so the backward has explicit state,
    // alpha and beta contributions from the two finite-difference taps that
    // the plain clockmem backward doesn't have. A wrong backward here would
    // silently train the head against nothing.
    let (s, d, n) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let mut q = mk(1.3, -2.0);
    let mut k = mk(0.6, 1.0);
    let mut v = mk(-0.8, 0.5);
    let mut g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let mut beta = param(vec![0.7], vec![1]);
    let s0 = vec![0.4, -0.9, 0.25];

    let run = |qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor, a: &Tensor, b: &Tensor| {
        ops::clockmem_inject(qq, kk, vv, gg, a, b, n, &s0).0
    };

    let out = run(&q, &k, &v, &g, &alpha, &beta);
    let grads = backward(&ops::sum_all(&out));
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gq = grads.get(&q.id).cloned().unwrap();
    let gk = grads.get(&k.id).cloned().unwrap();
    let gv = grads.get(&v.id).cloned().unwrap();
    let gg2 = grads.get(&g.id).cloned().unwrap();

    let na = numeric(&mut alpha, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), p, &beta.detach()))
    }, 1e-3);
    let nb = numeric(&mut beta, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), p))
    }, 1e-3);
    let nq = numeric(&mut q, |p| {
        sum_of(&run(p, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach()))
    }, 1e-3);
    let nk = numeric(&mut k, |p| {
        sum_of(&run(&q.detach(), p, &v.detach(), &g.detach(), &alpha.detach(), &beta.detach()))
    }, 1e-3);
    let nv = numeric(&mut v, |p| {
        sum_of(&run(&q.detach(), &k.detach(), p, &g.detach(), &alpha.detach(), &beta.detach()))
    }, 1e-3);
    let ng = numeric(&mut g, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), p, &alpha.detach(), &beta.detach()))
    }, 1e-3);

    check(&ga, &na, "alpha (inject)");
    check(&gb, &nb, "beta (inject)");
    check(&gq, &nq, "q (inject)");
    check(&gk, &nk, "k (inject)");
    check(&gv, &nv, "v (inject)");
    check(&gg2, &ng, "g (inject)");
}

#[test]
fn gradcheck_clockmem_readwin_inj() {
    // R1: out = q*cur*g + read, read = sum_j w[j]*state[t-j]. The read goes
    // to the residual WITHOUT the q*g gate, so the backward has window state
    // contributions (direct[t] = go*q*gg + sum_j w[j]*go[t+j]) and a wread
    // gradient that a gated readwin backward would lack.
    let (s, d, kwin) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let mut q = mk(1.3, -2.0);
    let mut k = mk(0.6, 1.0);
    let mut v = mk(-0.8, 0.5);
    let mut g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let mut beta = param(vec![0.7], vec![1]);
    let mut w = param(vec![0.2, -0.1, 0.4, 0.6, 0.0, -0.3, 0.1, 0.5, -0.2], vec![kwin, d]);
    let s0 = vec![0.4, -0.9, 0.25];

    let run = |qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor, a: &Tensor, b: &Tensor, ww: &Tensor| {
        ops::clockmem_readwin_inj(qq, kk, vv, gg, a, b, ww, &s0).0
    };

    let out = run(&q, &k, &v, &g, &alpha, &beta, &w);
    let grads = backward(&ops::sum_all(&out));
    let gq = grads.get(&q.id).cloned().unwrap();
    let gk = grads.get(&k.id).cloned().unwrap();
    let gv = grads.get(&v.id).cloned().unwrap();
    let gg2 = grads.get(&g.id).cloned().unwrap();
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gw = grads.get(&w.id).cloned().unwrap();

    let nq = numeric(&mut q, |p| {
        sum_of(&run(p, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &w.detach()))
    }, 1e-3);
    let nk = numeric(&mut k, |p| {
        sum_of(&run(&q.detach(), p, &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &w.detach()))
    }, 1e-3);
    let nv = numeric(&mut v, |p| {
        sum_of(&run(&q.detach(), &k.detach(), p, &g.detach(), &alpha.detach(), &beta.detach(), &w.detach()))
    }, 1e-3);
    let ng = numeric(&mut g, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), p, &alpha.detach(), &beta.detach(), &w.detach()))
    }, 1e-3);
    let na = numeric(&mut alpha, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), p, &beta.detach(), &w.detach()))
    }, 1e-3);
    let nb = numeric(&mut beta, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), p, &w.detach()))
    }, 1e-3);
    let nw = numeric(&mut w, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), p))
    }, 1e-3);

    check(&gq, &nq, "q (readwin_inj)");
    check(&gk, &nk, "k (readwin_inj)");
    check(&gv, &nv, "v (readwin_inj)");
    check(&gg2, &ng, "g (readwin_inj)");
    check(&ga, &na, "alpha (readwin_inj)");
    check(&gb, &nb, "beta (readwin_inj)");
    check(&gw, &nw, "w (readwin_inj)");
}

#[test]
fn r1c_reproduces_inject() {
    // Representability check (GPT's precondition): the structured family
    // (alpha_read, inv_beta) must contain the verified oracle BY
    // CONSTRUCTION. With alpha_read = alpha_c and inv_beta = 1/beta the
    // learned-inverse op must produce EXACTLY clockmem_inject's output --
    // if it can't represent the oracle, the R1c experiment answers nothing.
    let (s, d, n) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let (q, k, v) = (mk(1.3, -2.0), mk(0.6, 1.0), mk(-0.8, 0.5));
    let g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let beta = param(vec![0.7], vec![1]);
    let s0 = vec![0.4, -0.9, 0.25];
    let out_inject = ops::clockmem_inject(&q, &k, &v, &g, &alpha, &beta, n, &s0).0;
    // Family at the oracle point: alpha_read=alpha, inv_beta=1/beta.
    let alpha_read = param(alpha.data.to_vec(), vec![d]);
    let inv_beta = param(vec![1.0 / beta.data[0]; d], vec![d]);
    let out_learn = ops::clockmem_inject_learn(
        &q, &k, &v, &g, &alpha, &beta, &alpha_read, &inv_beta, n, &s0,
    )
    .0;
    for i in 0..s * d {
        let err = (out_learn.data[i] - out_inject.data[i]).abs();
        assert!(err < 1e-6, "i={i}: learn={} inject={} err={err}", out_learn.data[i], out_inject.data[i]);
    }
}

#[test]
fn gradcheck_clockmem_inject_learn() {
    // R1c: out = q*cur*g + read with read[t,c] = inv_beta[c]*(state[t-N+1,c]
    // - alpha_read[c]*state[t-N,c]). Unlike clockmem_inject, the physical
    // alpha/beta only enter through the state recurrence (read uses the
    // separate learnable alpha_read/inv_beta), and there are two new
    // per-channel gradients to check. Also a wrong backward would silently
    // leave alpha_read/inv_beta at init (the "gradient absent" failure mode
    // GPT wants to be able to distinguish).
    let (s, d, n) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let mut q = mk(1.3, -2.0);
    let mut k = mk(0.6, 1.0);
    let mut v = mk(-0.8, 0.5);
    let mut g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let mut beta = param(vec![0.7], vec![1]);
    let mut ar = param(vec![0.4, 0.2, 0.6], vec![d]);
    let mut ib = param(vec![0.9, 1.1, 0.8], vec![d]);
    let s0 = vec![0.4, -0.9, 0.25];

    let run = |qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor, a: &Tensor, b: &Tensor, aa: &Tensor, ii: &Tensor| {
        ops::clockmem_inject_learn(qq, kk, vv, gg, a, b, aa, ii, n, &s0).0
    };

    let out = run(&q, &k, &v, &g, &alpha, &beta, &ar, &ib);
    let grads = backward(&ops::sum_all(&out));
    let gq = grads.get(&q.id).cloned().unwrap();
    let gk = grads.get(&k.id).cloned().unwrap();
    let gv = grads.get(&v.id).cloned().unwrap();
    let gg2 = grads.get(&g.id).cloned().unwrap();
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gar = grads.get(&ar.id).cloned().unwrap();
    let gib = grads.get(&ib.id).cloned().unwrap();

    let nq = numeric(&mut q, |p| {
        sum_of(&run(p, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach()))
    }, 1e-3);
    let nk = numeric(&mut k, |p| {
        sum_of(&run(&q.detach(), p, &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach()))
    }, 1e-3);
    let nv = numeric(&mut v, |p| {
        sum_of(&run(&q.detach(), &k.detach(), p, &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach()))
    }, 1e-3);
    let ng = numeric(&mut g, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), p, &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach()))
    }, 1e-3);
    let na = numeric(&mut alpha, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), p, &beta.detach(), &ar.detach(), &ib.detach()))
    }, 1e-3);
    let nb = numeric(&mut beta, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), p, &ar.detach(), &ib.detach()))
    }, 1e-3);
    let nar = numeric(&mut ar, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), p, &ib.detach()))
    }, 1e-3);
    let nib = numeric(&mut ib, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), p))
    }, 1e-3);

    check(&gq, &nq, "q (inject_learn)");
    check(&gk, &nk, "k (inject_learn)");
    check(&gv, &nv, "v (inject_learn)");
    check(&gg2, &ng, "g (inject_learn)");
    check(&ga, &na, "alpha (inject_learn)");
    check(&gb, &nb, "beta (inject_learn)");
    check(&gar, &nar, "alpha_read (inject_learn)");
    check(&gib, &nib, "inv_beta (inject_learn)");
}

#[test]
fn r2_floor_keeps_memory_path_alive() {
    // R2 property (the only thing the design demands): even with the gate
    // SATURATED CLOSED (z=-20, sigmoid≈0), the floor keeps s>=floor>0 and the
    // memory path's own params (alpha_read, inv_beta) still receive REAL
    // (nonzero) gradient. If this test fails, the "cannot be annulled"
    // guarantee is broken and R2 answers nothing.
    let (s, d, n) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let q = mk(1.3, -2.0);
    let k = mk(0.6, 1.0);
    let v = mk(-0.8, 0.5);
    let g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let beta = param(vec![0.7], vec![1]);
    let ar = param(vec![0.4, 0.2, 0.6], vec![d]);
    let ib = param(vec![0.9, 1.1, 0.8], vec![d]);
    let z = param(vec![-20.0], vec![1]);
    let s0 = vec![0.4, -0.9, 0.25];
    let floor = 0.05f32;

    let sg = 1.0 / (1.0 + 20.0f32.exp());
    assert!(sg < 1e-8, "z=-20 should saturate the sigmoid");
    let out = ops::clockmem_inject_learn_g(&q, &k, &v, &g, &alpha, &beta, &ar, &ib, &z, floor, n, &s0).0;
    let grads = backward(&ops::sum_all(&out));
    let gz = grads.get(&z.id).cloned().unwrap();
    assert!(gz[0].abs() < 1e-4, "gate gradient at saturation should be ~0, got {}", gz[0]);
    let gar = grads.get(&ar.id).cloned().unwrap();
    let gib = grads.get(&ib.id).cloned().unwrap();
    let norm: f32 = gar.iter().map(|x| x * x).sum::<f32>() + gib.iter().map(|x| x * x).sum::<f32>();
    assert!(norm > 1e-6, "memory-path gradient is zero with the gate saturated closed: the floor guarantee is broken");
}

#[test]
fn gradcheck_clockmem_inject_learn_g() {
    // R2: out = q*cur*g + s*read with s = floor+(1-floor)*sigmoid(z). Checks
    // the gate gradient z, the scaled memory-path gradients (alpha_read,
    // inv_beta) and the physical alpha/beta (which only enter via the state
    // recurrence).
    let (s, d, n) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let mut q = mk(1.3, -2.0);
    let mut k = mk(0.6, 1.0);
    let mut v = mk(-0.8, 0.5);
    let mut g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let mut beta = param(vec![0.7], vec![1]);
    let mut ar = param(vec![0.4, 0.2, 0.6], vec![d]);
    let mut ib = param(vec![0.9, 1.1, 0.8], vec![d]);
    let mut z = param(vec![0.3], vec![1]);
    let s0 = vec![0.4, -0.9, 0.25];
    let floor = 0.05f32;

    let run = |qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor, a: &Tensor, b: &Tensor, aa: &Tensor, ii: &Tensor, zz: &Tensor| {
        ops::clockmem_inject_learn_g(qq, kk, vv, gg, a, b, aa, ii, zz, floor, n, &s0).0
    };

    let out = run(&q, &k, &v, &g, &alpha, &beta, &ar, &ib, &z);
    let grads = backward(&ops::sum_all(&out));
    let gq = grads.get(&q.id).cloned().unwrap();
    let gk = grads.get(&k.id).cloned().unwrap();
    let gv = grads.get(&v.id).cloned().unwrap();
    let gg2 = grads.get(&g.id).cloned().unwrap();
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gar = grads.get(&ar.id).cloned().unwrap();
    let gib = grads.get(&ib.id).cloned().unwrap();
    let gz = grads.get(&z.id).cloned().unwrap();

    let nq = numeric(&mut q, |p| {
        sum_of(&run(p, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nk = numeric(&mut k, |p| {
        sum_of(&run(&q.detach(), p, &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nv = numeric(&mut v, |p| {
        sum_of(&run(&q.detach(), &k.detach(), p, &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let ng = numeric(&mut g, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), p, &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let na = numeric(&mut alpha, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), p, &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nb = numeric(&mut beta, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), p, &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nar = numeric(&mut ar, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), p, &ib.detach(), &z.detach()))
    }, 1e-3);
    let nib = numeric(&mut ib, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), p, &z.detach()))
    }, 1e-3);
    let nz = numeric(&mut z, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), p))
    }, 1e-3);

    check(&gq, &nq, "q (inject_learn_g)");
    check(&gk, &nk, "k (inject_learn_g)");
    check(&gv, &nv, "v (inject_learn_g)");
    check(&gg2, &ng, "g (inject_learn_g)");
    check(&ga, &na, "alpha (inject_learn_g)");
    check(&gb, &nb, "beta (inject_learn_g)");
    check(&gar, &nar, "alpha_read (inject_learn_g)");
    check(&gib, &nib, "inv_beta (inject_learn_g)");
    check(&gz, &nz, "z (inject_learn_g)");
}

#[test]
fn r2_channel_contains_scalar() {
    // R2-channel (GPT control): with z constant across channels, the
    // per-channel gate reduces EXACTLY to the block scalar gate. The scalar
    // family is a subset of the channel family -- so a channel run can always
    // reproduce the block result, and any failure of channel is not
    // representability but optimization.
    let (s, d, n) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let q = mk(1.3, -2.0);
    let k = mk(0.6, 1.0);
    let v = mk(-0.8, 0.5);
    let g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let beta = param(vec![0.7], vec![1]);
    let ar = param(vec![0.4, 0.2, 0.6], vec![d]);
    let ib = param(vec![0.9, 1.1, 0.8], vec![d]);
    let z_scalar = param(vec![0.4], vec![1]);
    let z_chan = param(vec![0.4, 0.4, 0.4], vec![d]);
    let s0 = vec![0.4, -0.9, 0.25];
    let floor = 0.05f32;

    let a = ops::clockmem_inject_learn_g(&q, &k, &v, &g, &alpha, &beta, &ar, &ib, &z_scalar, floor, n, &s0).0;
    let b = ops::clockmem_inject_learn_gc(&q, &k, &v, &g, &alpha, &beta, &ar, &ib, &z_chan, floor, n, &s0).0;
    assert_eq!(a.shape, b.shape);
    for i in 0..a.data.len() {
        assert!((a.data[i] - b.data[i]).abs() < 1e-5, "channel!=scalar at {i}: {} vs {}", a.data[i], b.data[i]);
    }
}

#[test]
fn gradcheck_clockmem_inject_learn_gc() {
    // R2-channel: out = q*cur*g + s[c]*read, s[c] = floor+(1-floor)*sigmoid(z[c]),
    // z per channel. Checks all 9 gradients including z[c] (one per channel).
    let (s, d, n) = (6, 3, 3);
    let mk = |a: f32, b: f32| param((0..s * d).map(|j| ((j as f32) * a + b) / 7.0).collect(), vec![s, d]);
    let mut q = mk(1.3, -2.0);
    let mut k = mk(0.6, 1.0);
    let mut v = mk(-0.8, 0.5);
    let mut g = param((0..s * d).map(|j| ((j % 3) as f32 - 1.0) / 4.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.5, 0.3, 0.8], vec![d]);
    let mut beta = param(vec![0.7], vec![1]);
    let mut ar = param(vec![0.4, 0.2, 0.6], vec![d]);
    let mut ib = param(vec![0.9, 1.1, 0.8], vec![d]);
    let mut z = param(vec![0.3, -0.5, 1.1], vec![d]);
    let s0 = vec![0.4, -0.9, 0.25];
    let floor = 0.05f32;

    let run = |qq: &Tensor, kk: &Tensor, vv: &Tensor, gg: &Tensor, a: &Tensor, b: &Tensor, aa: &Tensor, ii: &Tensor, zz: &Tensor| {
        ops::clockmem_inject_learn_gc(qq, kk, vv, gg, a, b, aa, ii, zz, floor, n, &s0).0
    };

    let out = run(&q, &k, &v, &g, &alpha, &beta, &ar, &ib, &z);
    let grads = backward(&ops::sum_all(&out));
    let gq = grads.get(&q.id).cloned().unwrap();
    let gk = grads.get(&k.id).cloned().unwrap();
    let gv = grads.get(&v.id).cloned().unwrap();
    let gg2 = grads.get(&g.id).cloned().unwrap();
    let ga = grads.get(&alpha.id).cloned().unwrap();
    let gb = grads.get(&beta.id).cloned().unwrap();
    let gar = grads.get(&ar.id).cloned().unwrap();
    let gib = grads.get(&ib.id).cloned().unwrap();
    let gz = grads.get(&z.id).cloned().unwrap();

    let nq = numeric(&mut q, |p| {
        sum_of(&run(p, &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nk = numeric(&mut k, |p| {
        sum_of(&run(&q.detach(), p, &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nv = numeric(&mut v, |p| {
        sum_of(&run(&q.detach(), &k.detach(), p, &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let ng = numeric(&mut g, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), p, &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let na = numeric(&mut alpha, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), p, &beta.detach(), &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nb = numeric(&mut beta, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), p, &ar.detach(), &ib.detach(), &z.detach()))
    }, 1e-3);
    let nar = numeric(&mut ar, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), p, &ib.detach(), &z.detach()))
    }, 1e-3);
    let nib = numeric(&mut ib, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), p, &z.detach()))
    }, 1e-3);
    let nz = numeric(&mut z, |p| {
        sum_of(&run(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(), &ar.detach(), &ib.detach(), p))
    }, 1e-3);

    check(&gq, &nq, "q (inject_learn_gc)");
    check(&gk, &nk, "k (inject_learn_gc)");
    check(&gv, &nv, "v (inject_learn_gc)");
    check(&gg2, &ng, "g (inject_learn_gc)");
    check(&ga, &na, "alpha (inject_learn_gc)");
    check(&gb, &nb, "beta (inject_learn_gc)");
    check(&gar, &nar, "alpha_read (inject_learn_gc)");
    check(&gib, &nib, "inv_beta (inject_learn_gc)");
    check(&gz, &nz, "z (inject_learn_gc)");
}

#[test]
fn wread_init_delta_and_structured_finite_difference() {
    // B2.1 init: without EVA_READ_DF_N the window read starts as the delta
    // (w[0,c]=1, the base single-state read); with it, the two finite-
    // difference taps are placed on top: w[N-1,c]=1/beta, w[N,c]=-alpha_c/beta,
    // with alpha from the SAME alpha_squash the forward uses. K=1 (the
    // control) has no room for the taps and must stay a plain delta.
    use crate::model::clock::ClockMem;
    use crate::rng::Rng;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap();

    let save_win = std::env::var("EVA_READ_WIN").ok();
    let save_df = std::env::var("EVA_READ_DF_N").ok();
    let save_t = std::env::var("EVA_ALPHA_TEMP").ok();
    let save_a = std::env::var("EVA_ALPHA_ANTISAT").ok();
    std::env::remove_var("EVA_ALPHA_TEMP");
    std::env::remove_var("EVA_ALPHA_ANTISAT");

    let restore = |s: Option<String>, k: &str| match s {
        Some(v) => std::env::set_var(k, v),
        None => std::env::remove_var(k),
    };

    // K=1 control: must be exactly the base delta read.
    std::env::set_var("EVA_READ_WIN", "1");
    std::env::set_var("EVA_READ_DF_N", "3");
    let cm1 = ClockMem::new(8, &mut Rng::new(7));
    let w1 = cm1.wread.clone().unwrap();
    assert_eq!(w1.shape, vec![1, 8]);
    for c in 0..8 {
        assert!((w1.data[c] - 1.0).abs() < 1e-6, "K=1 w[0,c] must be 1 (c={c})");
    }

    // Delta (no DF): K=3 -> w[0]=1, w[1]=w[2]=0.
    std::env::set_var("EVA_READ_WIN", "3");
    std::env::remove_var("EVA_READ_DF_N");
    let cmd = ClockMem::new(8, &mut Rng::new(7));
    let wd = cmd.wread.clone().unwrap();
    assert_eq!(wd.shape, vec![3, 8]);
    for j in 0..3 {
        for c in 0..8 {
            let exp = if j == 0 { 1.0 } else { 0.0 };
            assert!((wd.data[j * 8 + c] - exp).abs() < 1e-6, "delta j={j} c={c}");
        }
    }

    // Structured: K=5, N=3 -> w[0]=1, w[2]=1/beta, w[3]=-alpha_c/beta, w[1]=w[4]=0.
    std::env::set_var("EVA_READ_WIN", "5");
    std::env::set_var("EVA_READ_DF_N", "3");
    let cm = ClockMem::new(8, &mut Rng::new(7));
    let w = cm.wread.clone().unwrap();
    assert_eq!(w.shape, vec![5, 8]);
    let beta0 = cm.beta.data[0];
    // Default squash (no temp, no antisat in this test): alpha = sigmoid(z).
    let alpha = ops::sigmoid(&cm.log_clock);
    for c in 0..8 {
        assert!((w.data[0 * 8 + c] - 1.0).abs() < 1e-6, "w[0,c] c={c}");
        assert!((w.data[2 * 8 + c] - 1.0 / beta0).abs() < 1e-5, "w[N-1,c] c={c}: {}", w.data[2 * 8 + c]);
        assert!(
            (w.data[3 * 8 + c] + alpha.data[c] / beta0).abs() < 1e-5,
            "w[N,c] c={c}: {} vs {}", w.data[3 * 8 + c], -alpha.data[c] / beta0
        );
    }
    for j in [1usize, 4] {
        for c in 0..8 {
            assert!(w.data[j * 8 + c].abs() < 1e-6, "tap {j} must be 0 (c={c})");
        }
    }

    restore(save_win, "EVA_READ_WIN");
    restore(save_df, "EVA_READ_DF_N");
    restore(save_t, "EVA_ALPHA_TEMP");
    restore(save_a, "EVA_ALPHA_ANTISAT");
}

#[test]
fn ssm_fixed_alpha_excludes_clock_and_is_constant() {
    // EVA_SSM_ALPHA=a (SSM baseline): (a) log_clock must NOT be a learned
    // parameter, (b) alpha_physical must be the constant `a` for every
    // channel, (c) a forward/forward_from run must produce a state that
    // follows exactly the fixed-alpha recurrence (no clock learning).
    use crate::model::clock::ClockMem;
    use crate::rng::Rng;
    use crate::nn::Module;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap();

    let save_a = std::env::var("EVA_SSM_ALPHA").ok();
    std::env::set_var("EVA_SSM_ALPHA", "0.999");
    let cm = ClockMem::new(8, &mut Rng::new(7));
    let restore = |s: Option<String>, k: &str| match s {
        Some(v) => std::env::set_var(k, v),
        None => std::env::remove_var(k),
    };

    assert_eq!(cm.ssm_alpha, Some(0.999));
    assert_eq!(cm.alpha_physical().shape, vec![8]);
    for c in 0..8 {
        assert!((cm.alpha_physical().data[c] - 0.999).abs() < 1e-6, "alpha[c] must be fixed 0.999 (c={c})");
    }
    for p in cm.parameters() {
        assert!(p.id != cm.log_clock.id, "log_clock must not be a parameter in SSM mode");
    }
    assert_eq!(cm.parameters().len(), cm.wq.parameters().len() + cm.wk.parameters().len()
        + cm.wv.parameters().len() + cm.wg.parameters().len() + 1, "only q/k/v/g + beta");

    // forward_from must follow exactly the fixed-alpha recurrence.
    let x = Tensor::new((0..64).map(|i| (i as f32) / 64.0 - 0.5).collect(), vec![8, 8]);
    let (_, fin) = cm.forward_from(&x, &vec![0.0; 8]);
    let beta_v = cm.beta.data[0];
    let k0 = cm.wk.forward(&x);
    let v0 = cm.wv.forward(&x);
    let a = 0.999f32;
    for c in 0..8 {
        // state[7,c] = beta * sum_{j=0..7} a^(7-j) * k[j,c]*v[j,c] (s0=0).
        let mut exp = 0.0f32;
        let mut pow = 1.0f32;
        for j in (0..8).rev() {
            exp += pow * k0.data[j * 8 + c] * v0.data[j * 8 + c];
            pow *= a;
        }
        exp *= beta_v;
        assert!((fin[c] - exp).abs() < 1e-3, "state final must follow fixed alpha (c={c})");
    }

    restore(save_a, "EVA_SSM_ALPHA");
}

#[test]
fn oracle_readout_is_exact_and_has_no_learned_param() {
    // EVA_READ_ORACLE=1 must (a) NOT create a learned wread parameter and
    // (b) produce taps equal to the B2.1 finite-difference init, recomputed    // from the CURRENT alpha/beta so the recovery stays exact even after
    // alpha has been learned (mutate log_clock and re-read: taps must track).
    use crate::model::clock::ClockMem;
    use crate::rng::Rng;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap();

    let save_win = std::env::var("EVA_READ_WIN").ok();
    let save_df = std::env::var("EVA_READ_DF_N").ok();
    let save_or = std::env::var("EVA_READ_ORACLE").ok();
    let save_t = std::env::var("EVA_ALPHA_TEMP").ok();
    let save_a = std::env::var("EVA_ALPHA_ANTISAT").ok();
    std::env::remove_var("EVA_ALPHA_TEMP");
    std::env::remove_var("EVA_ALPHA_ANTISAT");

    let restore = |s: Option<String>, k: &str| match s {
        Some(v) => std::env::set_var(k, v),
        None => std::env::remove_var(k),
    };

    std::env::set_var("EVA_READ_WIN", "5");
    std::env::set_var("EVA_READ_DF_N", "3");
    std::env::set_var("EVA_READ_ORACLE", "1");
    let mut cm = ClockMem::new(8, &mut Rng::new(7));
    assert!(cm.wread.is_none(), "oracle mode must not create a learned wread");
    let w = cm.oracle_read().unwrap();
    assert_eq!(w.shape, vec![5, 8]);
    let beta0 = cm.beta.data[0];
    let alpha = ops::sigmoid(&cm.log_clock);
    for c in 0..8 {
        assert!((w.data[0 * 8 + c] - 1.0).abs() < 1e-6, "oracle w[0,c] c={c}");
        assert!((w.data[2 * 8 + c] - 1.0 / beta0).abs() < 1e-5, "oracle w[N-1,c] c={c}");
        assert!(
            (w.data[3 * 8 + c] + alpha.data[c] / beta0).abs() < 1e-5,
            "oracle w[N,c] c={c}: {} vs {}", w.data[3 * 8 + c], -alpha.data[c] / beta0
        );
    }
    for j in [1usize, 4] {
        for c in 0..8 {
            assert!(w.data[j * 8 + c].abs() < 1e-6, "oracle tap {j} must be 0 (c={c})");
        }
    }
    // Taps must track a learned alpha: rewrite log_clock and re-read.
    std::sync::Arc::get_mut(&mut cm.log_clock.data)
        .unwrap()
        .iter_mut()
        .for_each(|v| *v = -3.0);
    let w2 = cm.oracle_read().unwrap();
    let alpha2 = ops::sigmoid(&cm.log_clock);
    for c in 0..8 {
        assert!(
            (w2.data[3 * 8 + c] + alpha2.data[c] / beta0).abs() < 1e-5,
            "oracle must track learned alpha c={c}"
        );
    }

    std::env::remove_var("EVA_READ_ORACLE");
    let cm2 = ClockMem::new(8, &mut Rng::new(7));
    assert!(cm2.wread.is_some(), "without oracle the learned wread must exist");

    restore(save_win, "EVA_READ_WIN");
    restore(save_df, "EVA_READ_DF_N");
    restore(save_or, "EVA_READ_ORACLE");
    restore(save_t, "EVA_ALPHA_TEMP");
    restore(save_a, "EVA_ALPHA_ANTISAT");
}

#[test]
fn a_carried_state_actually_changes_the_output() {
    // Control to make sure the initial state isn't being silently ignored.
    // Without this, a `clockmem_from` that discarded s0 would still pass
    // gradcheck (the gradient would be consistent with what it computes)
    // and would give results identical to the usual ones without anyone
    // noticing.
    let (s, d) = (4, 3);
    let q = param(vec![1.0; s * d], vec![s, d]);
    let k = param(vec![0.5; s * d], vec![s, d]);
    let v = param(vec![0.5; s * d], vec![s, d]);
    let g = param(vec![1.0; s * d], vec![s, d]);
    let alpha = param(vec![0.9, 0.9, 0.9], vec![d]);
    let beta = param(vec![1.0], vec![1]);

    let (zero, _) = ops::clockmem_from(&q, &k, &v, &g, &alpha, &beta, &vec![0.0; d]);
    let (with, fin) = ops::clockmem_from(&q, &k, &v, &g, &alpha, &beta, &vec![2.0; d]);

    assert!(zero.data[0] != with.data[0], "the initial state is being ignored");
    // And the final state has to come out nonzero to be chainable.
    assert!(fin.iter().all(|v| *v != 0.0), "did not return a usable final state");
}

#[allow(clippy::too_many_arguments)]
fn cm_lr_sum(
    q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, alpha: &Tensor, beta: &Tensor,
    pk: &Tensor, pv: &Tensor, pq: &Tensor, po: &Tensor, am: &Tensor, s0: &[f32],
) -> f32 {
    let (t, _) = ops::clockmem_lowrank_from(
        &q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach(),
        &pk.detach(), &pv.detach(), &pq.detach(), &po.detach(), &am.detach(), s0,
    );
    t.data.iter().sum()
}

/// Gradcheck for the low-rank outer-product write (task #58). Checks all
/// ELEVEN inputs, including the four projections and the r-sized clock of
/// the matrix state -- a backward that got only the element-wise half right
/// would pass a test that checked q/k/v/g alone, which is exactly the class
/// of bug this has to catch.
///
/// `s0` is deliberately NON-ZERO: the element-wise path reads the previous
/// window's state at t=0, and a backward that hardcodes zero there gets
/// alpha's gradient wrong only at the window boundary -- silent in training,
/// caught here.
#[test]
fn gradcheck_clockmem_lowrank() {
    let s = 6;
    let d = 4;
    let r = 3;
    let mut q = param((0..s * d).map(|i| (i as f32 - 9.0) / 12.0).collect(), vec![s, d]);
    let mut k = param((0..s * d).map(|i| (i as f32 * 1.3 - 5.0) / 13.0).collect(), vec![s, d]);
    let mut v = param((0..s * d).map(|i| (i as f32 * -0.7 + 3.0) / 9.0).collect(), vec![s, d]);
    let mut g = param((0..s * d).map(|i| ((i % 5) as f32 - 2.0) / 6.0).collect(), vec![s, d]);
    let mut alpha = param(vec![0.15, 0.4, 0.65, 0.9], vec![d]);
    let mut beta = param(vec![0.8], vec![1]);
    let mut pk = param((0..d * r).map(|i| (i as f32 - 5.0) / 7.0).collect(), vec![d, r]);
    let mut pv = param((0..d * r).map(|i| (i as f32 * 0.6 - 3.0) / 8.0).collect(), vec![d, r]);
    let mut pq = param((0..d * r).map(|i| (i as f32 * -0.5 + 4.0) / 10.0).collect(), vec![d, r]);
    let mut po = param((0..r * d).map(|i| (i as f32 * 0.9 - 4.0) / 11.0).collect(), vec![r, d]);
    let mut am = param(vec![0.2, 0.55, 0.85], vec![r]);
    let s0: Vec<f32> = (0..d).map(|c| (c as f32 - 1.5) / 5.0).collect();

    let (out, _) = ops::clockmem_lowrank_from(&q, &k, &v, &g, &alpha, &beta, &pk, &pv, &pq, &po, &am, &s0);
    let grads = backward(&ops::sum_all(&out));

    let take = |t: &Tensor| grads.get(&t.id).cloned().unwrap();
    let (gq, gk, gv, gg) = (take(&q), take(&k), take(&v), take(&g));
    let (ga, gb) = (take(&alpha), take(&beta));
    let (gpk, gpv, gpq, gpo, gam) = (take(&pk), take(&pv), take(&pq), take(&po), take(&am));

    let e = 1e-3;
    let nq = numeric(&mut q, |p| cm_lr_sum(p, &k, &v, &g, &alpha, &beta, &pk, &pv, &pq, &po, &am, &s0), e);
    let nk = numeric(&mut k, |p| cm_lr_sum(&q, p, &v, &g, &alpha, &beta, &pk, &pv, &pq, &po, &am, &s0), e);
    let nv = numeric(&mut v, |p| cm_lr_sum(&q, &k, p, &g, &alpha, &beta, &pk, &pv, &pq, &po, &am, &s0), e);
    let ng = numeric(&mut g, |p| cm_lr_sum(&q, &k, &v, p, &alpha, &beta, &pk, &pv, &pq, &po, &am, &s0), e);
    let na = numeric(&mut alpha, |p| cm_lr_sum(&q, &k, &v, &g, p, &beta, &pk, &pv, &pq, &po, &am, &s0), e);
    let nb = numeric(&mut beta, |p| cm_lr_sum(&q, &k, &v, &g, &alpha, p, &pk, &pv, &pq, &po, &am, &s0), e);
    let npk = numeric(&mut pk, |p| cm_lr_sum(&q, &k, &v, &g, &alpha, &beta, p, &pv, &pq, &po, &am, &s0), e);
    let npv = numeric(&mut pv, |p| cm_lr_sum(&q, &k, &v, &g, &alpha, &beta, &pk, p, &pq, &po, &am, &s0), e);
    let npq = numeric(&mut pq, |p| cm_lr_sum(&q, &k, &v, &g, &alpha, &beta, &pk, &pv, p, &po, &am, &s0), e);
    let npo = numeric(&mut po, |p| cm_lr_sum(&q, &k, &v, &g, &alpha, &beta, &pk, &pv, &pq, p, &am, &s0), e);
    let nam = numeric(&mut am, |p| cm_lr_sum(&q, &k, &v, &g, &alpha, &beta, &pk, &pv, &pq, &po, p, &s0), e);

    check(&gq, &nq, "lowrank q");
    check(&gk, &nk, "lowrank k");
    check(&gv, &nv, "lowrank v");
    check(&gg, &ng, "lowrank g");
    check(&ga, &na, "lowrank alpha");
    check(&gb, &nb, "lowrank beta");
    check(&gpk, &npk, "lowrank pk");
    check(&gpv, &npv, "lowrank pv");
    check(&gpq, &npq, "lowrank pq");
    check(&gpo, &npo, "lowrank po");
    check(&gam, &nam, "lowrank alpha_m");
}

/// The op must reduce EXACTLY to base ClockMem when the low-rank path is
/// switched off by zeroing the output projection. Guards against the new
/// path silently perturbing the old one -- the failure mode that would make
/// any A/B against `--arch clock` uninterpretable.
#[test]
fn lowrank_with_zero_po_equals_base_clockmem() {
    let (s, d, r) = (7, 5, 4);
    let q = param((0..s * d).map(|i| (i as f32 - 8.0) / 11.0).collect(), vec![s, d]);
    let k = param((0..s * d).map(|i| (i as f32 * 1.1 - 4.0) / 12.0).collect(), vec![s, d]);
    let v = param((0..s * d).map(|i| (i as f32 * -0.8 + 2.0) / 7.0).collect(), vec![s, d]);
    let g = param((0..s * d).map(|i| ((i % 4) as f32 - 1.0) / 5.0).collect(), vec![s, d]);
    let alpha = param(vec![0.1, 0.3, 0.5, 0.7, 0.95], vec![d]);
    let beta = param(vec![0.85], vec![1]);
    let pk = param((0..d * r).map(|i| (i as f32 - 3.0) / 6.0).collect(), vec![d, r]);
    let pv = param((0..d * r).map(|i| (i as f32 * 0.4 - 2.0) / 9.0).collect(), vec![d, r]);
    let pq = param((0..d * r).map(|i| (i as f32 * -0.3 + 1.0) / 8.0).collect(), vec![d, r]);
    let po = param(vec![0.0; r * d], vec![r, d]);
    let am = param(vec![0.25, 0.5, 0.75, 0.9], vec![r]);
    let s0: Vec<f32> = (0..d).map(|c| (c as f32 - 2.0) / 4.0).collect();

    let (lr, lr_state) = ops::clockmem_lowrank_from(&q, &k, &v, &g, &alpha, &beta, &pk, &pv, &pq, &po, &am, &s0);
    let (base, base_state) = ops::clockmem_from(&q, &k, &v, &g, &alpha, &beta, &s0);

    for i in 0..lr.data.len() {
        assert!((lr.data[i] - base.data[i]).abs() < 1e-6,
            "output {} differs with po=0: lowrank={} base={}", i, lr.data[i], base.data[i]);
    }
    for c in 0..d {
        assert!((lr_state[c] - base_state[c]).abs() < 1e-6,
            "carried state {} differs with po=0", c);
    }
}

/// Task #58, and this is the test that matters most: "the code touches the
/// new parameters" is NOT "the new parameters learn". This checks the
/// EFFECT -- that gradient actually reaches all five tensors -- and pins
/// down the one-step delay the zero init of `po` implies, so nobody
/// rediscovers it as a bug later.
///
/// At `po = 0` the low-rank path contributes exactly zero to the output, so
/// `grd = sum_c grad_out[c]*po[j,c] = 0` and NOTHING downstream of it gets
/// gradient: pk, pv, pq and log_clock_m are all legitimately zero on the
/// first step. `po` itself is not (its gradient is `rd * grad_out`), so it
/// leaves zero immediately and everything else starts learning from step 2.
/// That is a one-step delay, not the saturation trap of Section 4.6 -- the
/// distinction is the whole reason to check it rather than assume it.
#[test]
fn lowrank_params_actually_receive_gradient() {
    let (s, d, r) = (6, 5, 3);
    let mk = |f: &dyn Fn(usize) -> f32, n: usize, sh: Vec<usize>| param((0..n).map(f).collect(), sh);
    let q = mk(&|i| (i as f32 - 7.0) / 10.0, s * d, vec![s, d]);
    let k = mk(&|i| (i as f32 * 1.2 - 4.0) / 11.0, s * d, vec![s, d]);
    let v = mk(&|i| (i as f32 * -0.6 + 2.0) / 8.0, s * d, vec![s, d]);
    let g = mk(&|i| ((i % 4) as f32 - 1.0) / 5.0, s * d, vec![s, d]);
    let alpha = param(vec![0.2, 0.4, 0.6, 0.8, 0.95], vec![d]);
    let beta = param(vec![0.9], vec![1]);
    let pk = mk(&|i| (i as f32 - 4.0) / 6.0, d * r, vec![d, r]);
    let pv = mk(&|i| (i as f32 * 0.5 - 2.0) / 7.0, d * r, vec![d, r]);
    let pq = mk(&|i| (i as f32 * -0.4 + 3.0) / 9.0, d * r, vec![d, r]);
    let am = param(vec![0.3, 0.6, 0.9], vec![r]);
    let s0 = vec![0.0; d];

    let nonzero = |g: &[f32]| g.iter().any(|x| x.abs() > 1e-9);
    let run = |po: &Tensor| {
        let (out, _) = ops::clockmem_lowrank_from(&q, &k, &v, &g, &alpha, &beta, &pk, &pv, &pq, po, &am, &s0);
        let gr = backward(&ops::sum_all(&out));
        let t = |x: &Tensor| gr.get(&x.id).cloned().unwrap();
        (t(&pk), t(&pv), t(&pq), t(po), t(&am))
    };

    // Step 1, po = 0: only po moves. Documented, not a bug.
    let po0 = param(vec![0.0; r * d], vec![r, d]);
    let (a, b, c, o, m) = run(&po0);
    assert!(nonzero(&o), "po must receive gradient at its zero init, or the path is dead forever");
    for (name, gv) in [("pk", &a), ("pv", &b), ("pq", &c), ("log_clock_m", &m)] {
        assert!(!nonzero(gv), "{} should have zero gradient while po=0 (see doc comment)", name);
    }

    // Step 2, po off zero: every tensor of the new path learns.
    let po1 = param((0..r * d).map(|i| (i as f32 * 0.3 - 1.0) / 8.0).collect(), vec![r, d]);
    let (a, b, c, o, m) = run(&po1);
    for (name, gv) in [("pk", &a), ("pv", &b), ("pq", &c), ("po", &o), ("log_clock_m", &m)] {
        assert!(nonzero(gv), "{} receives NO gradient once po != 0 -- the path does not learn", name);
    }
}
