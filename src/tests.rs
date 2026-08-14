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
