use std::time::{Duration, Instant};

use crate::data::TextDataset;
use crate::model::{EvaConfig, EvaModel};
use crate::nn::Module;
use crate::optim::AdamW;
use crate::model::block::Mixer;
use crate::recall::Recall;
use crate::save::{load_model, save_model};
use crate::tensor::autograd::backward;
use crate::tensor::Tensor;
use crate::tokenizer::ByteTokenizer;

// Second intervention on the flattening of `alpha`: a
// periodic trace of `log_clock`'s gradient DURING training, not just at
// the final checkpoint -- "I want to see where the trajectory starts to
// diverge." Free: these are the SAME gradients already computed for the
// optimizer step, no extra pass. Behind `EVA_ALPHA_TRACE=<n>` (every n
// steps), off by default.
struct Trace {
    every: usize,
    abs_sum: Vec<Vec<f64>>,
    n: usize,
}

impl Trace {
    fn from_env(n_blocks: usize, dim: usize) -> Option<Self> {
        let every = std::env::var("EVA_ALPHA_TRACE").ok()?.parse::<usize>().ok()?;
        if every == 0 { return None; }
        Some(Trace { every, abs_sum: vec![vec![0.0; dim]; n_blocks], n: 0 })
    }

    fn accumulate(&mut self, model: &EvaModel, grads: &std::collections::HashMap<usize, Vec<f32>>) {
        for (bi, block) in model.blocks.iter().enumerate() {
            if let Mixer::Clock(clock) = &block.mixer {
                if let Some(g) = grads.get(&clock.log_clock.id) {
                    for (c, &gv) in g.iter().enumerate() {
                        self.abs_sum[bi][c] += gv.abs() as f64;
                    }
                }
            }
        }
        self.n += 1;
    }

    fn maybe_print(&mut self, model: &EvaModel, step: usize) {
        if step % self.every != 0 { return; }
        for (bi, block) in model.blocks.iter().enumerate() {
            let Mixer::Clock(clock) = &block.mixer else { continue };
            // Use alpha_physical so the trace shows the REAL alpha in every
            // mode (learned squash, temperature, or the fixed SSM constant).
            let alpha = clock.alpha_physical();
            let dim = alpha.shape[0];
            let (mut fast, mut medium, mut slow) = (0usize, 0usize, 0usize);
            let mut alpha_sum = 0.0f64;
            for c in 0..dim {
                let a = alpha.data[c] as f64;
                alpha_sum += a;
                if a < 0.05 { fast += 1 } else if a < 0.3 { medium += 1 } else { slow += 1 }
            }
            // In SSM mode log_clock is not a parameter, so no gradient entry.
            let mean_grad: f64 = if clock.ssm_alpha.is_some() {
                0.0
            } else {
                self.abs_sum[bi].iter().sum::<f64>() / dim as f64 / self.n.max(1) as f64
            };
            println!("  [trace step {step}] block {bi}: fast={fast} medium={medium} slow={slow} | mean_alpha={:.4} | mean_|grad|={:.6} (n={})",
                alpha_sum / dim as f64, mean_grad, self.n);
        }
        for v in self.abs_sum.iter_mut() { v.iter_mut().for_each(|x| *x = 0.0); }
        self.n = 0;
    }
}

// R1 (2026-08-15): periodic trace of the windowed readout taps `wread`
// DURING training, so we can see WHEN the long read appears. Reports the RMS
// per tap (norm over channels) for the current-state tap (j=0), the two
// finite-difference taps (j=N-1, j=N) and the max/mean of the rest. Behind
// `EVA_WREAD_TRACE=<n>` (every n steps), off by default. Requires the
// readout to exist (EVA_READ_WIN set); otherwise it prints nothing.
struct WreadTrace {
    every: usize,
    n: usize,
    floor: f32,
    // Gradient accumulators (abs-sum since last print), only used when the
    // block has a structured read (R1c/R2): per block, per channel.
    grad_ar: Vec<Vec<f64>>,
    grad_ib: Vec<Vec<f64>>,
    grad_z: Vec<f64>,
    n_grad: usize,
}

impl WreadTrace {
    fn from_env(n_blocks: usize, dim: usize) -> Option<Self> {
        let every = std::env::var("EVA_WREAD_TRACE").ok()?.parse::<usize>().ok()?;
        if every == 0 { return None; }
        let n = std::env::var("EVA_READ_DF_N").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let floor = std::env::var("EVA_READ_R2_FLOOR")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.05);
        Some(WreadTrace {
            every,
            n,
            floor,
            grad_ar: vec![vec![0.0; dim]; n_blocks],
            grad_ib: vec![vec![0.0; dim]; n_blocks],
            grad_z: vec![0.0; n_blocks],
            n_grad: 0,
        })
    }

    /// R2/R1c: accumulate |grad| of the memory-path params (alpha_read,
    /// inv_beta) and of the gate (gate_z), same gradients already computed
    /// for the optimizer step. Answers "does memory_path keep receiving real
    /// gradient while the gate is closed" -- the property R2 tests.
    fn accumulate(&mut self, model: &EvaModel, grads: &std::collections::HashMap<usize, Vec<f32>>) {
        for (bi, block) in model.blocks.iter().enumerate() {
            let Mixer::Clock(clock) = &block.mixer else { continue };
            let Some(ar) = &clock.alpha_read else { continue };
            let Some(ib) = &clock.inv_beta else { continue };
            if let Some(g) = grads.get(&ar.id) {
                for (c, &gv) in g.iter().enumerate() {
                    self.grad_ar[bi][c] += gv.abs() as f64;
                }
            }
            if let Some(g) = grads.get(&ib.id) {
                for (c, &gv) in g.iter().enumerate() {
                    self.grad_ib[bi][c] += gv.abs() as f64;
                }
            }
            if let Some(z) = &clock.gate_z {
                if let Some(g) = grads.get(&z.id) {
                    self.grad_z[bi] += g[0].abs() as f64;
                }
            }
        }
        self.n_grad += 1;
    }

    fn maybe_print(&mut self, model: &EvaModel, step: usize) {
        if step % self.every != 0 { return; }
        let n = self.n;
        for (bi, block) in model.blocks.iter().enumerate() {
            let Mixer::Clock(clock) = &block.mixer else { continue };
            // R2 trace: the gate story, three numbers apart:
            //   z (raw pre-activation), s (gate WITH the floor), mean|s*read|
            //   (the effective contribution to the residual, measured in the
            //   op forward), and the mean |grad| of alpha_read/inv_beta/z
            //   since the last print (the property (2) test). s=floor+(1-floor)*sigmoid(z).
            if let (Some(ar), Some(_ib), Some(z)) = (&clock.alpha_read, &clock.inv_beta, &clock.gate_z) {
                let ng = self.n_grad.max(1);
                let d = ar.shape[0] as f64;
                let g_ar: f64 = self.grad_ar[bi].iter().sum::<f64>() / d / ng as f64;
                let g_ib: f64 = self.grad_ib[bi].iter().sum::<f64>() / d / ng as f64;
                let g_z: f64 = self.grad_z[bi] / ng as f64;
                let (eff, _eff_n) = crate::tensor::ops::r2_read_stats(ar.id).unwrap_or((0.0, 0));
                if z.shape == vec![ar.shape[0]] {
                    // R2-channel: the DISTRIBUTION of s[c] (save it, not
                    // just bpb). Summary: mean/min/max s and the fractions at
                    // the floor (closed) and near 1 (open).
                    let (mut s_mean, mut s_min, mut s_max) = (0.0f64, f64::MAX, 0.0f64);
                    let (mut n_closed, mut n_open) = (0usize, 0usize);
                    for &zv in z.data.iter() {
                        let s2 = 1.0 / (1.0 + (-zv as f64).exp());
                        let s_c = self.floor as f64 + (1.0 - self.floor as f64) * s2;
                        s_mean += s_c;
                        s_min = s_min.min(s_c);
                        s_max = s_max.max(s_c);
                        if s_c < self.floor as f64 + 0.1 {
                            n_closed += 1;
                        }
                        if s_c > 0.9 {
                            n_open += 1;
                        }
                    }
                    s_mean /= d;
                    println!(
                        "  [r2c trace step {step}] block {bi}: z mean={:.4} s mean={s_mean:.4} min={s_min:.4} max={s_max:.4} closed={}/{} open={}/{} | mean|s*read|={eff:.6} | mean|g| ar={g_ar:.2e} ib={g_ib:.2e} z={g_z:.2e} (n={})",
                        z.data.iter().map(|&x| x as f64).sum::<f64>() / d,
                        n_closed,
                        z.shape[0],
                        n_open,
                        z.shape[0],
                        self.n_grad
                    );
                } else {
                    let z_v = z.data[0] as f64;
                    let sg = 1.0 / (1.0 + (-z_v).exp());
                    let s_gate = self.floor as f64 + (1.0 - self.floor as f64) * sg;
                    println!(
                        "  [r2 trace step {step}] block {bi}: z={z_v:.4} s={s_gate:.4} mean|s*read|={eff:.6} | mean|g| ar={g_ar:.2e} ib={g_ib:.2e} z={g_z:.2e} (n={})",
                        self.n_grad
                    );
                }
                continue;
            }
            // R1c trace: how far the structured read (alpha_read, inv_beta)
            // is from the physical clock (alpha_c, 1). we wants these two
            // traced through training to distinguish "gradient absent"
            // (params frozen at init), "present but bad trajectory" (moves
            // away), and "converging to a different solution" (moves, loss
            // improves). dalpha = alpha_read - alpha, di = inv_beta - 1/beta.
            if let (Some(ar), Some(ib)) = (&clock.alpha_read, &clock.inv_beta) {
                let phys = clock.alpha_physical();
                let (mut a_mean, mut da_mean, mut da_max) = (0.0f64, 0.0f64, 0.0f64);
                let (mut i_mean, mut di_mean, mut di_max) = (0.0f64, 0.0f64, 0.0f64);
                for c in 0..ar.shape[0] {
                    let a = ar.data[c] as f64;
                    let i = ib.data[c] as f64;
                    let da = (a - phys.data[c] as f64).abs();
                    let di = (i - 1.0).abs();
                    a_mean += a; da_mean += da; da_max = da_max.max(da);
                    i_mean += i; di_mean += di; di_max = di_max.max(di);
                }
                let d = ar.shape[0] as f64;
                println!(
                    "  [r1c trace step {step}] block {bi}: alpha_read mean={:.6} |da|mean={:.6} max={:.6} | inv_beta mean={:.6} |di|mean={:.6} max={:.6}",
                    a_mean / d, da_mean / d, da_max, i_mean / d, di_mean / d, di_max
                );
                continue;
            }
            let Some(w) = &clock.wread else { continue };
            let (kwin, d) = (w.shape[0], w.shape[1]);
            let taps: Vec<f64> = (0..kwin)
                .map(|j| {
                    let s: f64 = (0..d).map(|c| (w.data[j * d + c] as f64).powi(2)).sum();
                    (s / d as f64).sqrt()
                })
                .collect();
            let (mut taps_max, mut taps_sum, mut cnt) = (0.0f64, 0.0f64, 0usize);
            for &t in &taps[1..] {
                taps_max = taps_max.max(t);
                taps_sum += t;
                cnt += 1;
            }
            let taps_mean = taps_sum / cnt.max(1) as f64;
            let jn1 = if n >= 1 && n - 1 < kwin { taps[n - 1] } else { f64::NAN };
            let jn = if n >= 1 && n < kwin { taps[n] } else { f64::NAN };
            let taps0 = taps[0];
            println!(
                "  [wread trace step {step}] block {bi}: j0={taps0:.6} mean_taps={taps_mean:.6} max_taps={taps_max:.6} | w[N-1]={jn1:.6} w[N]={jn:.6} (rms/tap)"
            );
        }
        self.grad_ar.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 0.0));
        self.grad_ib.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 0.0));
        self.grad_z.iter_mut().for_each(|x| *x = 0.0);
        self.n_grad = 0;
    }
}

// Final benchmark: the gate as a training regularizer, recipe
// consolidated independently -- "teach where the table is deterministic, don't
const GATE_CAP: f32 = 7.7; // same cap Seneca used in adaptive_lambda.rs
const GATE_H_HIGH: f32 = 2.0;
const GATE_EPS: f32 = 1e-9;

fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

/// [seq, vocab] bias to add to the raw logits before the loss: zero where
/// the table didn't answer, `beta*gate*bias` where it did. Built with
/// `Tensor::new` (requires_grad=false by default) -- constant for the
/// graph, the loss's gradient flows back to the model's logits intact.
fn inject_bias(table: &Recall, ds: &TextDataset, wi: usize, seq: usize, vocab: usize, beta: f32) -> Tensor {
    let mut data = vec![0.0f32; seq * vocab];
    for t in 0..seq {
        let global_pos = wi * seq + t + 1;
        let from = global_pos.saturating_sub(8);
        let ctx = &ds.ids[from..global_pos];
        if ctx.is_empty() {
            continue;
        }
        let Some((q, _count)) = table.lookup_detail(ctx, vocab) else { continue };
        let h = shannon_bits(&q);
        let gate = (1.0 - h / GATE_H_HIGH).clamp(0.0, 1.0);
        if gate == 0.0 {
            continue;
        }
        let lambda = beta * gate;
        let row = &mut data[t * vocab..(t + 1) * vocab];
        for (k, &qi) in q.iter().enumerate() {
            row[k] = lambda * (qi + GATE_EPS).ln().max(-GATE_CAP);
        }
    }
    Tensor::new(data, vec![seq, vocab])
}

pub struct TrainConfig {
    pub data_path: String,
    pub epochs: usize,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    pub log_every: usize,
    pub out_path: String,
    pub resume: Option<String>,
    /// What fraction of the windows is held out for validation. 0 turns it off.
    pub val_frac: f32,
    /// Surprise threshold. 0 = learn from everything (today's convention).
    pub surprise: f32,
    /// Local credit: each block with its own objective, no crossing gradient.
    pub local: bool,
    /// ClockMem's state doesn't reset between windows.
    pub persist: bool,
    /// Walk the corpus in order without persisting state. This is the
    /// CONTROL for `--persist`: without it two changes would be compared
    /// at once, order and memory, and there'd be no way to know which one
    /// produced the difference.
    pub inorder: bool,
    /// Beta of the table regularizer (0.0 = off, normal training, identical
    /// to before this recipe). Gated by the table's determinism, not by the
    /// model's confidence -- see the comment next to `inject_bias`.
    pub gate_beta: f32,
    /// The "survival" hypothesis: probability that EACH block
    /// gets skipped, independently, on every step (0.0 = off, identical to
    /// before). Reuses `forward_skips` (already public, the same one
    /// `ceiling` uses) -- this is Stochastic Depth (Huang et al. 2016) at
    /// the block level, deliberately without any bias toward a particular
    /// block: the question is whether the network redistributes dependence
    /// on its own, not whether we force it to.
    pub destroy_p: f32,
}

pub fn train(tcfg: &TrainConfig, mcfg: &EvaConfig) -> Result<(), String> {
    let ds = TextDataset::from_file(&tcfg.data_path, mcfg.seq_len)
        .map_err(|e| format!("could not read {}: {}", tcfg.data_path, e))?;
    let n_windows = ds.num_windows();
    if n_windows == 0 {
        return Err(format!("the dataset is too small for seq_len {}", mcfg.seq_len));
    }

    let mut model = match &tcfg.resume {
        Some(p) => load_model(p).map_err(|e| format!("could not load {}: {}", p, e))?,
        None => EvaModel::new(mcfg.clone()),
    };
    let mut rng = crate::rng::Rng::new(tcfg.seed);
    let mut opt = AdamW::new(tcfg.lr, tcfg.wd);
    // B2.1: separate/scaled LR for the windowed readout (env EVA_READ_LR).
    // Applied as a true per-parameter LR multiplier inside the AdamW step.
    if let Ok(v) = std::env::var("EVA_READ_LR") {
        if let Ok(mult) = v.parse::<f32>() {
            let mut n = 0usize;
            for (name, t) in model.named_parameters() {
                if name.ends_with(".wread") {
                    opt.lr_mult.insert(t.id, mult);
                    n += 1;
                }
            }
            println!("eva: EVA_READ_LR={mult}: readout lr scaled on {n} wread params");
        }
    }
    // Warmup is a short epoch: before that, the moving average would be
    // computed with a random model and would mean nothing.
    let mut gate = crate::learn::gate(tcfg.surprise, 200);
    // Counted separately because backward costs twice what forward does:
    // comparing runs by "steps" would hide exactly what we want to measure.
    let (mut fwd, mut bwd) = (0usize, 0usize);

    // Training scaffolding: one head per intermediate block. Not saved in
    // the checkpoint, so inference stays identical.
    let mut heads = tcfg.local.then(|| {
        crate::local::Heads::new(
            model.blocks.len().saturating_sub(1),
            mcfg.dim,
            mcfg.vocab,
            mcfg.eps,
            &mut crate::rng::Rng::new(tcfg.seed ^ 0xC0FFEE),
        )
    });
    if let Some(h) = &heads {
        println!("eva: LOCAL CREDIT, no gradient crosses from block to block");
        println!("eva: {} extra params in auxiliary heads (scaffolding, not part of the model)", h.count());
    }

    let total_params = model.param_count();
    println!("eva: dataset {} bytes, {} windows, {} params",
        ds.ids.len(), n_windows, total_params);

    // THE CUT GOES AT THE END AND IS CONTIGUOUS, not scattered. Neighboring
    // windows share context: with a random cut, the model would see during
    // training the text right next to what later gets treated as the exam,
    // and validation would score better than it should. An exam that leaks
    // doesn't measure anything.
    let n_val = if tcfg.val_frac > 0.0 {
        (((n_windows as f32) * tcfg.val_frac).round() as usize).clamp(1, n_windows / 2)
    } else {
        0
    };
    let n_train = n_windows - n_val;
    if n_train == 0 {
        return Err("no training windows left after the cut".into());
    }
    println!("eva: {n_train} windows to train on, {n_val} to validate");

    // Frozen table, built ONLY on train -- same discipline as today's eval
    // scripts. None if the regularizer is off (beta=0.0): zero extra cost,
    // training identical to before.
    let table: Option<Recall> = if tcfg.gate_beta > 0.0 {
        let bytes_train: Vec<usize> = ds.ids[..n_train * mcfg.seq_len].to_vec();
        let t = Recall::build(&bytes_train);
        println!("eva: table regularizer ACTIVE, beta={:.2}, H_HIGH={GATE_H_HIGH:.1} (gated by determinism, not confidence)", tcfg.gate_beta);
        Some(t)
    } else {
        None
    };

    let mut trace = Trace::from_env(model.blocks.len(), mcfg.dim);
    if trace.is_some() {
        println!("eva: alpha/gradient TRACE ACTIVE (EVA_ALPHA_TRACE)");
    }
    let mut wread_trace = WreadTrace::from_env(model.blocks.len(), mcfg.dim);
    if wread_trace.is_some() {
        println!("eva: wread TRACE ACTIVE (EVA_WREAD_TRACE)");
    }

    let total_steps = n_train * tcfg.epochs;
    let mut step = 0usize;
    let mut running = 0.0f32;
    let t0 = Instant::now();
    let mut last_log = Instant::now();
    let mut best_val_loss = f32::INFINITY;
    let mut best_step = 0usize;
    let best_path = {
        let p = std::path::Path::new(&tcfg.out_path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("weights");
        match p.parent() {
            Some(parent) if !parent.as_os_str().is_empty() =>
                format!("{}/{}_best.{}", parent.display(), stem, ext),
            _ => format!("{}_best.{}", stem, ext),
        }
    };

    // WITH PERSISTENT STATE YOU HAVE TO GO IN ORDER. Shuffling would fill
    // the state with context from a document that has nothing to do with
    // the next one: that's not memory, it's noise. And since order also
    // affects training, the baseline for comparison has to run in order
    // too -- otherwise two things would be getting compared at once.
    let mut states = model.fresh_states();

    for epoch in 0..tcfg.epochs {
        let order: Vec<usize> = if tcfg.persist || tcfg.inorder {
            (0..n_train).collect()
        } else {
            ds.shuffled_train_indices(n_train, &mut rng)
        };
        for &wi in &order {
            let (input, target) = ds.window(wi);

            fwd += 1;
            let loss_v = if let Some(h) = heads.as_mut() {
                bwd += 1;
                local_step(&mut model, h, &mut opt, &input, &target)
            } else {
                // THE GRAPH GETS FREED BEFORE THE OPTIMIZER, and it's not
                // cosmetic. The weights are shared with the graph through
                // `Arc`; if the graph is still alive, the optimizer's
                // `make_mut` copies EVERY parameter on EVERY step. It
                // wouldn't error: it would just run slow. That's why the
                // scope, and that's why the copy counter further below.
                let (loss_v, grads) = {
                    let logits = if tcfg.persist {
                        model.forward_carrying(&input, &mut states)
                    } else if tcfg.destroy_p > 0.0 {
                        let skips: Vec<usize> = (0..model.blocks.len())
                            .filter(|_| rng.next_f32() < tcfg.destroy_p)
                            .collect();
                        model.forward_skips(&input, &skips).0
                    } else {
                        model.forward(&input)
                    };
                    // The bias is constant (no graph): `add` only needs
                    // grad from `logits`, which it does have. Nothing new
                    // to gradcheck -- these are two existing ops composed.
                    let logits = match &table {
                        Some(t) => crate::tensor::ops::add(
                            &logits,
                            &inject_bias(t, &ds, wi, mcfg.seq_len, mcfg.vocab, tcfg.gate_beta),
                        ),
                        None => logits,
                    };
                    let loss = crate::tensor::ops::cross_entropy(&logits, &target);
                    let v = loss.data[0];
                    let g = if gate.should_learn(v) {
                        Some(crate::prof::time(crate::prof::P::Backward, || backward(&loss)))
                    } else {
                        None
                    };
                    (v, g)
                };
                if let Some(grads) = grads {
                    bwd += 1;
                    if let Some(tr) = trace.as_mut() {
                        tr.accumulate(&model, &grads);
                    }
                    if let Some(wt) = wread_trace.as_mut() {
                        wt.accumulate(&model, &grads);
                    }
                    let mut params = model.parameters_mut();
                    crate::prof::time(crate::prof::P::Optim, || opt.step(&mut params, &grads));
                }
                loss_v
            };

            running += loss_v;
            step += 1;

            if let Some(tr) = trace.as_mut() {
                tr.maybe_print(&model, step);
            }
            if let Some(wtr) = wread_trace.as_mut() {
                wtr.maybe_print(&model, step);
            }

            if step % tcfg.log_every == 0 {
                let avg = running / tcfg.log_every as f32;
                let elapsed = last_log.elapsed();
                let sps = if elapsed.as_secs_f32() > 0.0 {
                    tcfg.log_every as f32 / elapsed.as_secs_f32()
                } else {
                    0.0
                };
                println!("step {}/{} | epoch {} | loss {:.4} | {:.1} steps/s",
                    step, total_steps, epoch + 1, avg, sps);
                running = 0.0;
                last_log = Instant::now();
            }

            if n_val > 0 && step % (tcfg.log_every * 5) == 0 {
                let vl = eval_loss(&model, &ds, n_train, n_windows);
                println!("  validation: loss {:.4} | {:.3} bits/byte", vl, bits_per_byte(vl));
                if vl < best_val_loss {
                    best_val_loss = vl;
                    best_step = step;
                    save_model(&best_path, &model).map_err(|e| format!("could not save best: {}", e))?;
                    println!("  ** best checkpoint saved at step {} (loss {:.4})", step, vl);
                }
            }

            if step % (tcfg.log_every * 10) == 0 {
                let sample = generate_sample(&model, mcfg.seq_len, 64, &mut rng);
                println!("sample: {:?}", ByteTokenizer::decode(&sample));
                save_model(&tcfg.out_path, &model).map_err(|e| format!("could not save: {}", e))?;
            }
        }
    }

    save_model(&tcfg.out_path, &model).map_err(|e| format!("could not save: {}", e))?;
    println!("eva: process memory peak {:.1} MB", crate::local::peak_rss_mb());
    crate::tensor::held::report(total_params);
    let copies = crate::optim::parameter_copies();
    println!("  parameter copies from a shared Arc: {copies}  (has to be 0)");
    if n_val > 0 {
        if best_step > 0 {
            println!("eva: BEST validation loss {:.4} | {:.3} bits/byte at step {} (saved as {})",
                best_val_loss, bits_per_byte(best_val_loss), best_step, best_path);
        }
        // With clean state: comparable to any run.
        let vl = eval_loss(&model, &ds, n_train, n_windows);
        if tcfg.persist {
            // And with the state carried over from prior text, which is
            // the advantage being tested. Reported SEPARATELY: mixing them
            // would be claiming victory for a difference that isn't the
            // one actually believed to be there.
            let vp = eval_loss_carrying(&model, &ds, n_train, n_windows);
            println!("eva: validation with inherited state {:.4} | {:.3} bits/byte",
                vp, bits_per_byte(vp));
        }
        // THE NUMBER ARCHITECTURES ARE COMPARED ON. Training loss only says
        // how much it memorized.
        println!("eva: FINAL VALIDATION loss {:.4} | {:.3} bits/byte ({} windows, arch {})",
            vl, bits_per_byte(vl), n_windows - n_train, mcfg.arch.name());
        println!("eva: rule '{}' | {} forward, {} backward ({:.0}% learned)",
            gate.name(), fwd, bwd, 100.0 * bwd as f32 / fwd.max(1) as f32);
    }
    let elapsed: Duration = t0.elapsed();
    crate::prof::report(elapsed);
    println!("eva: training finished in {:.1}s, weights in {}", elapsed.as_secs_f32(), tcfg.out_path);
    Ok(())
}

/// One step with local credit. Returns the LAST block's loss, which is the
/// model's real output and therefore what's comparable to the baseline.
///
/// The key is the order: each block backpropagates, updates, and **frees**
/// before the next one starts. Summing the losses and doing one backward at
/// the end would give the same gradients and **no memory savings**, which
/// is the one thing actually being bought here.
fn local_step(
    model: &mut EvaModel,
    heads: &mut crate::local::Heads,
    opt: &mut AdamW,
    input: &[usize],
    target: &[usize],
) -> f32 {
    use crate::tensor::ops as ops;
    let n = model.blocks.len();
    let mut x = model.embed.embed(input);
    let mut last = 0.0;

    for i in 0..n {
        let y = model.blocks[i].forward(&x);
        let is_last = i + 1 == n;
        let logits = if is_last {
            ops::matmul(&model.norm_out.forward(&y), &model.head_w)
        } else {
            heads.logits(i, &y)
        };
        let loss = ops::cross_entropy(&logits, target);
        if is_last {
            last = loss.data[0];
        }

        let grads = crate::prof::time(crate::prof::P::Backward, || backward(&loss));

        let mut params: Vec<&mut crate::tensor::Tensor> = Vec::new();
        if i == 0 {
            params.push(&mut model.embed.table);
        }
        params.extend(model.blocks[i].parameters_mut());
        if is_last {
            params.extend(model.norm_out.parameters_mut());
            params.push(&mut model.head_w);
        } else {
            params.extend(heads.params_mut(i));
        }
        crate::prof::time(crate::prof::P::Optim, || opt.step(&mut params, &grads));

        // `detach` is what cuts the credit: the next block starts from a
        // tensor with no graph behind it, so this block's graph gets freed
        // right here.
        x = y.detach();
    }
    last
}

/// Mean loss over windows the model never saw. No backward.
fn eval_loss(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) -> f32 {
    let mut sum = 0.0;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        sum += crate::tensor::ops::cross_entropy(&logits, &target).data[0];
    }
    sum / (to - from) as f32
}

/// Like `eval_loss`, but letting the state carry from one window to the
/// next: the exam is taken with the memory the previous text left behind.
fn eval_loss_carrying(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) -> f32 {
    let mut states = model.fresh_states();
    let mut sum = 0.0;
    let mut peak = 0.0f32;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward_carrying(&input, &mut states);
        sum += crate::tensor::ops::cross_entropy(&logits, &target).data[0];
        for e in states.iter() {
            for v in e {
                peak = peak.max(v.abs());
            }
        }
    }
    // If this is huge, the slow channels (alpha near 1) are accumulating
    // without forgetting and the state saturated: the problem would be the
    // clock's initialization, not the idea of persisting.
    println!("eva: max magnitude of the inherited state: {peak:.2}");
    sum / (to - from) as f32
}

/// Bits per byte: the honest unit for a byte-level model, and comparable
/// across corpora and across architectures. Loss in nats doesn't tell
/// anyone anything.
fn bits_per_byte(loss: f32) -> f32 {
    loss / std::f32::consts::LN_2
}

pub fn generate_sample(model: &EvaModel, seq: usize, max: usize, rng: &mut crate::rng::Rng) -> Vec<usize> {
    let mut ids = crate::gen::generate(model, &[0], max, 0.9, 32, rng, seq);
    ids.retain(|&x| x != 0);
    ids
}
