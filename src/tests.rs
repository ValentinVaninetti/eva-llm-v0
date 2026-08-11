use crate::nn::param;
use crate::tensor::autograd::backward;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

fn numeric(t: &mut Tensor, f: impl Fn(&Tensor) -> f32, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; t.data.len()];
    for i in 0..t.data.len() {
        let orig = t.data[i];
        t.data[i] = orig + eps;
        let fp = f(t);
        t.data[i] = orig - eps;
        let fm = f(t);
        t.data[i] = orig;
        out[i] = (fp - fm) / (2.0 * eps);
    }
    out
}

fn check(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "len mismatch en {}", name);
    for i in 0..a.len() {
        let err = (a[i] - b[i]).abs();
        let scale = (a[i].abs() + b[i].abs()).max(1.0);
        assert!(err / scale < 1e-3, "{}[{}]: analitico={} numerico={} err={}", name, i, a[i], b[i], err);
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

fn cm_sum(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, alpha: &Tensor, beta: &Tensor) -> f32 {
    let t = ops::clockmem(&q.detach(), &k.detach(), &v.detach(), &g.detach(), &alpha.detach(), &beta.detach());
    t.data.iter().sum()
}
