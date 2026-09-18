//! Activation-localisation probe for the ORACLE readout.
//!
//! It began as a hunt for "signal death" between the recovered write and the
//! logits (state_probe B had shown the write ~84% linearly decodable). After
//! the oracle fix (EVA_READ_INJECT, w[N] tap) the premise is obsolete: the
//! write survives 100% and the model solves the task. Kept because the
//! per-stage decodability is still the map of where the signal is.
//!
//! The recovered write is placed into `read[t]` by construction (the oracle
//! taps are exact). Ridge probes are fit at masked positions t >= N on each
//! intermediate:
//!
//!   per block  read[t]  recovered write, pre-q*g
//!              m[t]     = q[t]*read[t]*g[t]        mixer output
//!              h2[t]    = h[t] + m[t]              post-residual
//!              out[t]   = h2[t] + glu(norm2(h2))   block output
//!   final      hidden   pre output RMSNorm
//!              head_in  post output RMSNorm, pre head linear
//!
//! Label = input[t-N+1], the untransformed value the write carries. The model's
//! own output is F(input[t-N+1]), so its top-1 is scored against F(label) while
//! the probes fit the untransformed token directly -- F is a fixed bijection.
//!
//! USAGE: cargo run --release --example activation_probe -- <weights> <data> <N> [train_windows]
//!   requires EVA_READ_WIN=K, EVA_READ_DF_N=N, EVA_READ_INJECT=1.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::nn::Module;
use eva_llm_v0::rng::Rng;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::ops as ops;
use eva_llm_v0::tensor::Tensor;

/// The physics permutation F (fixed, see examples/generate_synthetic.rs).
const PHYSICS_SEED: u64 = 0xF15A1CA;

fn physics_perm() -> Vec<usize> {
    let mut rng = Rng::new(PHYSICS_SEED);
    let mut idx: Vec<usize> = (0..256).collect();
    rng.shuffle(&mut idx);
    idx
}

fn main() {
    let weights = std::env::args().nth(1).expect("usage: activation_probe <weights> <data> <N> [train_windows]");
    let data = std::env::args().nth(2).expect("usage: activation_probe <weights> <data> <N> [train_windows]");
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok())
        .expect("usage: activation_probe <weights> <data> <N> [train_windows]");
    let ptrain = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    let nb = model.blocks.len();
    let d = model.cfg.dim;
    let kwin = std::env::var("EVA_READ_WIN").ok().and_then(|v| v.parse::<usize>().ok())
        .expect("set EVA_READ_WIN=K (the oracle window, K=N+1)");
    assert_eq!(kwin, n + 1, "oracle probe assumes K=N+1");

    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n_win = ds.num_windows();
    let n_val = (((n_win as f32) * 0.1).round() as usize).clamp(1, n_win / 2);
    let n_train = n_win - n_val;
    let ptrain = ptrain.min(n_train);
    let pstats = 30.min(ptrain);

    let antisat = std::env::var("EVA_ALPHA_ANTISAT").is_ok();
    let temp = std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
    let squash = |lc: f32| -> f32 {
        if antisat {
            0.5 * (1.0 + lc / (1.0 + lc * lc).sqrt())
        } else if temp != 1.0 {
            1.0 / (1.0 + (-lc / temp).exp())
        } else {
            1.0 / (1.0 + (-lc).exp())
        }
    };

    println!("eva activation_probe: {weights} | {data} | N={n} seq={seq} blocks={nb} dim={d} K={kwin}");
    println!("  probe region per window: t in [{n}, {seq}) -> {} positions (label = input[t-N+1])", seq - n);
    println!("  oracle taps: w[0]=1, w[N-1]=1/beta, w[N]=-alpha_c/beta (exact recovery)\n");

    // Per-block learned params used by the replicated forward.
    let mut alphas: Vec<Vec<f32>> = Vec::new();
    let mut betas: Vec<f32> = Vec::new();
    for b in &model.blocks {
        let Mixer::Clock(c) = &b.mixer else { panic!("not a ClockMem model") };
        alphas.push(c.log_clock.data.iter().map(|&lc| squash(lc)).collect());
        betas.push(c.beta.data[0]);
    }

    let mut lays: Vec<String> = Vec::new();
    for bi in 0..nb {
        for name in ["read", "m", "h2", "out"] {
            lays.push(format!("b{bi}.{name}"));
        }
    }
    lays.push("final.hidden".into());
    lays.push("final.head_in".into());
    let nl = lays.len();
    let fc = d + 1;
    let lams = [0.01f64, 0.1, 1.0, 10.0];

    // z-scaling stats.
    let mut mean = vec![vec![0.0f64; d]; nl];
    let mut var = vec![vec![0.0f64; d]; nl];
    let mut stats_cnt = 0usize;
    for wi in 0..pstats {
        let (input, _t) = ds.window(wi);
        let activ = forward_layers(&model, &input, &alphas, &betas, n, kwin);
        for (li, _) in lays.iter().enumerate() {
            for t in n..seq {
                let v = &activ[t][li];
                for c in 0..d {
                    mean[li][c] += v[c] as f64;
                    var[li][c] += v[c] as f64 * v[c] as f64;
                }
            }
        }
        stats_cnt += (seq - n);
    }
    let scl: Vec<Vec<f32>> = (0..nl)
        .map(|li| {
            (0..d)
                .map(|c| {
                    let m = mean[li][c] / stats_cnt as f64;
                    let v = (var[li][c] / stats_cnt as f64 - m * m).max(1e-12);
                    (v.sqrt() as f32).max(1e-8)
                })
                .collect()
        })
        .collect();

    // Fit moments.
    let mut xtx = vec![vec![0.0f64; fc * fc]; nl];
    let mut xty = vec![vec![0.0f64; fc * 256]; nl];
    let mut cnt = vec![0usize; nl];
    for wi in 0..ptrain {
        let (input, target) = ds.window(wi);
        let activ = forward_layers(&model, &input, &alphas, &betas, n, kwin);
        for t in n..seq {
            let label = target[t - n] as usize;
            for (li, _) in lays.iter().enumerate() {
                let z: Vec<f32> = (0..d).map(|c| activ[t][li][c] / scl[li][c]).collect();
                acc(&mut xtx[li], &mut xty[li], &mut cnt[li], &z, label, fc);
            }
        }
    }
    println!("samples (fit): {} per layer\n", cnt[0]);

    // Val features.
    let mut rows: Vec<Vec<(Vec<f32>, usize)>> = vec![Vec::new(); nl];
    for wi in n_train..n_win {
        let (input, target) = ds.window(wi);
        let activ = forward_layers(&model, &input, &alphas, &betas, n, kwin);
        for t in n..seq {
            let label = target[t - n] as usize;
            for (li, _) in lays.iter().enumerate() {
                rows[li].push(((0..d).map(|c| activ[t][li][c] / scl[li][c]).collect(), label));
            }
        }
    }
    println!("samples (val): {}\n", rows[0].len());

    println!("=== WHERE THE RECOVERED WRITE SURVIVES (ridge val acc / bpb) ===");
    for (li, name) in lays.iter().enumerate() {
        let (lam, bpb, accv) = best_lambda(&xtx[li], &xty[li], fc, &lams, &rows[li]);
        println!("  {name:<14} lam={lam:5}  VAL bpb={bpb:.4}  acc={accv:.3}%");
    }

    // Reference: the model's own top-1 acc at masked positions. The model
    // emits F(input[t-N+1]) (the generator applies F), so compare against F(label).
    let perm = physics_perm();
    let (mut ok, mut tot) = (0usize, 0usize);
    for wi in n_train..n_win {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        for t in n..seq {
            let row = &logits.data[t * 256..(t + 1) * 256];
            let mut best = 0usize;
            for j in 1..256 {
                if row[j] > row[best] {
                    best = j;
                }
            }
            if best == perm[target[t - n] as usize] {
                ok += 1;
            }
            tot += 1;
        }
    }
    println!("\n  MODEL OWN top-1 acc at masked positions: {:.3}% ({ok}/{tot})", 100.0 * ok as f32 / tot as f32);
    println!("\n(chance floor: 8.0000 bpb, 0.39% acc)\n");
}

/// One window forward with the ORACLE windowed read, collecting the probe
/// layers at every position. activ[t] = Vec of `nl` layer vectors.
fn forward_layers(
    model: &EvaModel,
    input: &[usize],
    alphas: &[Vec<f32>],
    betas: &[f32],
    n: usize,
    kwin: usize,
) -> Vec<Vec<Vec<f32>>> {
    let d = model.cfg.dim;
    let nb = model.blocks.len();
    let nl = nb * 4 + 2;
    let mut x = model.embed.embed(input);
    let mut activ: Vec<Vec<Vec<f32>>> = vec![Vec::new(); input.len()];
    for bi in 0..nb {
        let b = &model.blocks[bi];
        let h = ops::add(&x, &b.conv.forward(&b.norm0.forward(&x)));
        let mix = b.norm1.forward(&h);
        let Mixer::Clock(clock) = &b.mixer else { panic!("not a ClockMem model") };
        let q = clock.wq.forward(&mix);
        let k = clock.wk.forward(&mix);
        let v = clock.wv.forward(&mix);
        let g = ops::sigmoid(&clock.wg.forward(&mix));
        let alpha = &alphas[bi];
        let beta = betas[bi];
        let s = q.shape[0];
        // Oracle taps (exact recovery, recomputed from the CURRENT alpha).
        let mut w = vec![0.0f32; kwin * d];
        for c in 0..d {
            w[c] = 1.0;
            w[(n - 1) * d + c] = 1.0 / beta;
            w[n * d + c] = -alpha[c] / beta;
        }
        let inject = std::env::var("EVA_READ_INJECT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            != 0;
        // State recurrence + read + mixer output.
        let mut cur = vec![0.0f32; d];
        let mut state = vec![vec![0.0f32; d]; s];
        let mut read = vec![vec![0.0f32; d]; s];
        let mut m = vec![vec![0.0f32; d]; s];
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha[c] * cur[c] + beta * k.data[t * d + c] * v.data[t * d + c];
                state[t][c] = cur[c];
            }
            for c in 0..d {
                let jmax = kwin.min(t + 1);
                let mut r = 0.0f32;
                for j in 0..jmax {
                    r += w[j * d + c] * state[t - j][c];
                }
                read[t][c] = r;
                if inject && t >= n {
                    let rec = (state[t - n + 1][c] - alpha[c] * state[t - n][c]) / beta;
                    read[t][c] = rec;
                    m[t][c] = q.data[t * d + c] * cur[c] * g.data[t * d + c] + rec;
                } else {
                    m[t][c] = q.data[t * d + c] * r * g.data[t * d + c];
                }
            }
        }
        // Residual + GLU, collecting block outputs.
        let h2 = ops::add(&h, &Tensor::new(m.iter().flatten().cloned().collect(), vec![s, d]));
        let out = ops::add(&h2, &b.glu.forward(&b.norm2.forward(&h2)));
        for t in 0..s {
            let li = bi * 4;
            let mut layer: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; d]).collect();
            layer[0] = read[t].clone();
            layer[1] = m[t].clone();
            layer[2] = h2.data[t * d..(t + 1) * d].to_vec();
            layer[3] = out.data[t * d..(t + 1) * d].to_vec();
            if activ[t].is_empty() {
                activ[t] = vec![Vec::new(); nl];
            }
            for li2 in 0..4 {
                activ[t][li + li2] = layer[li2].clone();
            }
        }
        x = out;
    }
    let hidden = x.clone();
    let head_in = model.norm_out.forward(&x);
    for t in 0..input.len() {
        let base = nb * 4;
        activ[t][base] = hidden.data[t * d..(t + 1) * d].to_vec();
        activ[t][base + 1] = head_in.data[t * d..(t + 1) * d].to_vec();
    }
    activ
}

fn best_lambda(xtx: &[f64], xty: &[f64], fc: usize, lams: &[f64], rows: &[(Vec<f32>, usize)]) -> (f64, f32, f32) {
    let mut best = (lams[0], f32::MAX, 0.0f32);
    for &lam in lams {
        let w = solve_ridge(xtx, xty, fc, lam);
        let (bpb, acc) = eval_rows(&w, rows, fc);
        if bpb < best.1 {
            best = (lam, bpb, acc);
        }
    }
    best
}

fn acc(xtx: &mut [f64], xty: &mut [f64], cnt: &mut usize, feat: &[f32], label: usize, fc: usize) {
    let mut x = vec![0.0f64; fc];
    for (i, &v) in feat.iter().enumerate() {
        x[i] = v as f64;
    }
    x[fc - 1] = 1.0;
    for i in 0..fc {
        let xi = x[i];
        for j in 0..fc {
            xtx[i * fc + j] += xi * x[j];
        }
        xty[i * 256 + label] += xi;
    }
    *cnt += 1;
}

fn solve_ridge(xtx: &[f64], xty: &[f64], f: usize, lam: f64) -> Vec<f64> {
    let mut a = xtx.to_vec();
    let mut rhs = xty.to_vec();
    for i in 0..f {
        a[i * f + i] += lam;
    }
    for col in 0..f {
        let mut piv = col;
        for r in col + 1..f {
            if a[r * f + col].abs() > a[piv * f + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..f {
                a.swap(col * f + c, piv * f + c);
            }
            for y in 0..256 {
                rhs.swap(col * 256 + y, piv * 256 + y);
            }
        }
        let dd = a[col * f + col];
        if dd.abs() < 1e-12 {
            continue;
        }
        for r in col + 1..f {
            let mm = a[r * f + col] / dd;
            if mm.abs() < 1e-15 {
                continue;
            }
            for c in col..f {
                a[r * f + c] -= mm * a[col * f + c];
            }
            for y in 0..256 {
                rhs[r * 256 + y] -= mm * rhs[col * 256 + y];
            }
        }
    }
    let mut w = vec![0.0f64; f * 256];
    for col in (0..f).rev() {
        let dd = a[col * f + col];
        if dd.abs() < 1e-12 {
            continue;
        }
        for y in 0..256 {
            let mut s = rhs[col * 256 + y];
            for c in col + 1..f {
                s -= a[col * f + c] * w[c * 256 + y];
            }
            w[col * 256 + y] = s / dd;
        }
    }
    w
}

fn eval_rows(w: &[f64], rows: &[(Vec<f32>, usize)], fc: usize) -> (f32, f32) {
    let d = fc - 1;
    let mut nats = 0.0f64;
    let mut correct = 0usize;
    let count = rows.len();
    for (feat, label) in rows {
        let mut logits = [0.0f64; 256];
        for j in 0..256 {
            let mut s = w[d * 256 + j];
            for i in 0..d {
                s += w[i * 256 + j] * feat[i] as f64;
            }
            logits[j] = s;
        }
        nats += logsumexp(&logits) - logits[*label];
        if argmax(&logits) == *label {
            correct += 1;
        }
    }
    let bpb = (nats / count.max(1) as f64 / std::f64::consts::LN_2) as f32;
    (bpb, 100.0 * correct as f32 / count.max(1) as f32)
}

fn logsumexp(x: &[f64; 256]) -> f64 {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + x.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}

fn argmax(x: &[f64; 256]) -> usize {
    let mut bi = 0;
    for i in 1..256 {
        if x[i] > x[bi] {
            bi = i;
        }
    }
    bi
}
