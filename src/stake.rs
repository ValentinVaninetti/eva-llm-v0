//! 8. THAT IT BETS -- the per-span stake head.
//!
//! The decisive experiment that got defined with Claudio: does a head that
//! reads the hidden state AT THE START of a span predict whether the span is
//! going to go well, better than the free bar (mean p[argmax] approx 0.67)
//! and before spending anything?
//!
//! The head is a probe: a linear projection of the hidden state onto a
//! stake in (0,1), trained with `stake_loss` (the new op, forward+backward+
//! gradcheck, in the contract's style). The hidden state comes in DETACHED:
//! the probe doesn't send gradient back to the model. Letting the gradient
//! cross is decision 4, which is more expensive, and only gets measured if
//! there's signal here.
//!
//! It's scaffolding like `local::Heads`: it isn't saved in the checkpoint,
//! the model stays untouched.

use crate::model::EvaModel;
use crate::nn::param;
use crate::rng::Rng;
use crate::tensor::ops as ops;
use crate::tensor::Tensor;

/// The stake head: `s = sigmoid(w*h + b)`. D numbers + 1 bias.
pub struct StakeHead {
    pub w: Tensor,
    pub b: Tensor,
}

impl StakeHead {
    pub fn new(dim: usize, rng: &mut Rng) -> Self {
        let bound = 1.0 / (dim as f32).sqrt();
        StakeHead {
            w: param((0..dim).map(|_| rng.uniform(-bound, bound)).collect(), vec![dim]),
            b: param(vec![0.0], vec![1]),
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w, &mut self.b]
    }

    pub fn count(&self) -> usize {
        self.w.data.len() + self.b.data.len()
    }

    /// The stake for each span of a window, over the real hidden state.
    pub fn stakes(&self, hidden: &Tensor, span_len: usize) -> Vec<f32> {
        let d = hidden.shape[1];
        let s = hidden.shape[0];
        let mut out = Vec::new();
        let mut k = 0;
        while (k + 1) * span_len <= s {
            let start = k * span_len;
            let mut z = self.b.data[0];
            for j in 0..d {
                z += self.w.data[j] * hidden.data[start * d + j];
            }
            out.push(1.0 / (1.0 + (-z).exp()));
            k += 1;
        }
        out
    }
}

/// How well each span of a window went: the fraction of positions where the
/// model's argmax was correct. It's the `good` from the design doc, and it's
/// data, not something differentiable: it enters the loss through `saved_f`.
pub fn good_per_span(logits: &Tensor, targets: &[usize], span_len: usize) -> Vec<f32> {
    let (s, v) = (logits.shape[0], logits.shape[1]);
    let mut out = Vec::new();
    let mut k = 0;
    while (k + 1) * span_len <= s {
        let mut correct = 0usize;
        for t in k * span_len..(k + 1) * span_len {
            let row = &logits.data[t * v..(t + 1) * v];
            let argmax = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
                .0;
            if argmax == targets[t] {
                correct += 1;
            }
        }
        out.push(correct as f32 / span_len as f32);
        k += 1;
    }
    out
}

/// The representation that feeds the probe: the hidden state, detached, in
/// its raw form (pre-norm) or the one the model actually uses to predict
/// (post-norm, RMSNorm + gain). The choice is measured because the model
/// predicts on the second one; if the signal is there, it has to be there.
fn representation(model: &EvaModel, hidden: &Tensor, post_norm: bool) -> Tensor {
    if !post_norm {
        return hidden.detach();
    }
    // Applied by hand on the detached data: running `norm_out.forward`
    // would leave a node pointing at the model's w, and backpropagating
    // stake_loss would end up training the model's norm -- exactly what the
    // probe isn't allowed to touch.
    let (s, d) = (hidden.shape[0], hidden.shape[1]);
    let eps = model.cfg.eps;
    let wn = &model.norm_out.w.data;
    let mut out = vec![0.0; s * d];
    for i in 0..s {
        let mut sq = 0.0;
        for j in 0..d {
            let v = hidden.data[i * d + j];
            sq += v * v;
        }
        let inv = 1.0 / (sq / d as f32 + eps).sqrt();
        for j in 0..d {
            out[i * d + j] = hidden.data[i * d + j] * inv * wn[j];
        }
    }
    Tensor::new(out, vec![s, d])
}

/// Trains the probe and measures it. Returns (r on training, r on
/// validation, number of validation spans).
///
/// The model stays frozen: it never touches its parameters. Only `w` and
/// `b` get updated, through the graph's `stake_loss`.
pub fn train_and_measure(
    model: &EvaModel,
    ds: &crate::data::TextDataset,
    n_train: usize,
    n_val: usize,
    span_len: usize,
    epochs: usize,
    lr: f32,
    seed: u64,
    post_norm: bool,
) -> (f32, f32, usize) {
    let mut rng = Rng::new(seed);
    let mut head = StakeHead::new(model.cfg.dim, &mut rng);
    let mut opt = crate::optim::AdamW::new(lr, 0.0);

    for epoch in 0..epochs {
        let order = ds.shuffled_train_indices(n_train, &mut rng);
        let mut sum = 0.0f32;
        for (i, &wi) in order.iter().enumerate() {
            let (input, target) = ds.window(wi);
            let (logits, hidden) = model.forward_hidden(&input);
            let good = good_per_span(&logits, &target, span_len);
            // The hidden state comes in disconnected: the probe doesn't
            // send gradient to the model. That's the whole point of the
            // measurement.
            let h = representation(model, &hidden, post_norm);
            let loss = ops::stake_loss(&h, &head.w, &head.b, &good, span_len);
            let grads = crate::tensor::autograd::backward(&loss);
            let mut params = head.params_mut();
            opt.step(&mut params, &grads);
            sum += loss.data[0];
            if i % 500 == 0 {
                println!("  stake epoch {} | step {} | loss {:.4}", epoch + 1, i, loss.data[0]);
            }
        }
        println!("  stake epoch {} | mean loss {:.4}", epoch + 1, sum / order.len().max(1) as f32);
    }

    // Measured over the training windows (did it memorize noise?) and the
    // validation ones (the ones the model never saw, the number that
    // matters).
    let (r_train, _) = measure(&head, model, ds, 0, n_train, span_len, post_norm);
    let (r_val, n_val_spans) = measure(&head, model, ds, n_train, n_train + n_val, span_len, post_norm);
    println!("  head: {} params (w+b) | r(stake vs good) train {:.3} | val {:.3} over {n_val_spans} spans",
        head.count(), r_train, r_val);
    (r_train, r_val, n_val_spans)
}

/// r(stake, good) of the head over windows [from, to).
pub fn measure(
    head: &StakeHead,
    model: &EvaModel,
    ds: &crate::data::TextDataset,
    from: usize,
    to: usize,
    span_len: usize,
    post_norm: bool,
) -> (f32, usize) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let (logits, hidden) = model.forward_hidden(&input);
        let good = good_per_span(&logits, &target, span_len);
        let h = representation(model, &hidden, post_norm);
        let stakes = head.stakes(&h, span_len);
        for (x, y) in stakes.iter().zip(&good) {
            xs.push(*x as f64);
            ys.push(*y as f64);
        }
    }
    (crate::bet::pearson(&xs, &ys), xs.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn good_counts_the_argmax_not_the_probability() {
        // Each row has a single maximum; good looks at where it falls.
        let logits = param(
            vec![0.0, 1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0],
            vec![4, 3],
        );
        let targets = vec![1usize, 2, 0, 1];
        let good = good_per_span(&logits, &targets, 2);
        assert_eq!(good, vec![1.0, 1.0], "both rows of each span are correct");
    }

    #[test]
    fn good_keeps_the_fraction() {
        let logits = param(
            vec![0.0, 1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 1.0],
            vec![4, 3],
        );
        let targets = vec![1usize, 0, 0, 2];
        let good = good_per_span(&logits, &targets, 2);
        assert_eq!(good, vec![0.5, 1.0], "span 0 gets 1 out of 2 correct");
    }

    #[test]
    fn spans_do_not_use_leftovers() {
        // s=5, span=2 -> 2 complete spans; position 4 is discarded.
        let mut rng = Rng::new(3);
        let head = StakeHead::new(3, &mut rng);
        let hidden = param((0..15).map(|i| i as f32 * 0.1).collect(), vec![5, 3]);
        assert_eq!(head.stakes(&hidden, 2).len(), 2);
    }

    /// Full-pipeline sanity check: if the signal is in the state, the probe
    /// finds it. Without this test, an r approx 0 on the real model could
    /// mean a broken pipeline instead of a real result.
    #[test]
    fn the_probe_learns_a_signal_that_is_actually_there() {
        let (d, s, span) = (8usize, 8usize, 4usize);
        let mut rng = Rng::new(7);
        let mut head = StakeHead::new(d, &mut rng);
        let mut opt = crate::optim::AdamW::new(1e-2, 0.0);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..400 {
            let good = if rng.uniform(0.0, 1.0) > 0.5 { 1.0 } else { 0.0 };
            // The start of each span (rows 0 and d) encodes `good` in
            // coordinate 0; the rest is noise.
            let mut data = vec![0.0; s * d];
            data[0] = good * 4.0 - 2.0;
            data[d] = good * 4.0 - 2.0;
            let hidden = Tensor::new(data, vec![s, d]);
            let loss = ops::stake_loss(&hidden, &head.w, &head.b, &[good, good], span);
            let grads = crate::tensor::autograd::backward(&loss);
            let mut params = head.params_mut();
            opt.step(&mut params, &grads);
            xs.push(head.stakes(&hidden, span)[0] as f64);
            ys.push(good as f64);
        }
        let r = crate::bet::pearson(&xs, &ys);
        assert!(r > 0.9, "the probe did not learn a signal that is there: r={r}");
    }
}
