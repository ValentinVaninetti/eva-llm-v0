//! STORAGE-vs-READ probe for the long-range benchmark. Run BEFORE touching
//! `src/`.
//!
//! Does the leaky ClockMem state (or the WRITES a windowed readout could
//! recover) linearly contain the value needed to predict target[t]?
//!
//!   target[t] = F(input[t-N+1])                      the recurrence
//!   cur[t]    = beta * sum_j alpha^j * k[t-j]*v[t-j] the leaky state
//!   w[tau,c]  = (cur[tau,c] - alpha_c*cur[tau-1,c])/beta
//!               the WRITE at tau, recovered by the finite difference a read
//!               window of size >= N+1 can express
//!
//! RIDGE probes, features z-scored on train, reported as val bpb, with the
//! disambiguation pre-registered before running:
//!
//!   A: cur[t]     -> input[t-N+1]    B: w[t-N+1] -> input[t-N+1]
//!   A fails, B fails -> STORAGE: the value never made it into the state
//!   A fails, B works -> READ: it is there, but out=q*cur*g cannot extract a
//!                       specific past position
//!   A works          -> even the current read has it in reach
//!
//! This REPLICATES the block forward exactly (same ops, same order, same alpha
//! squash) because the state lives inside `ops::clockmem`, and reads the same
//! EVA_ALPHA_TEMP / EVA_ALPHA_ANTISAT as the real forward. State is FRESH per
//! window, as in mask_eval, which keeps windows independent and lets it sample
//! train windows instead of forwarding the whole corpus.
//!
//! USAGE: cargo run --release --example state_probe -- <weights> <data> <N> [train_windows]
//!   data must come from generate_synthetic; the checkpoint's val cut is the
//!   last 10%. train_windows defaults to 200.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::nn::Module;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::ops as ops;
use eva_llm_v0::tensor::Tensor;

fn main() {
    let weights = std::env::args().nth(1).expect("usage: state_probe <weights> <data> <N> [train_windows]");
    let data = std::env::args().nth(2).expect("usage: state_probe <weights> <data> <N> [train_windows]");
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok())
        .expect("usage: state_probe <weights> <data> <N> [train_windows]");
    let ptrain = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    let model = load_model(&weights).expect("could not load the checkpoint");
    let seq = model.cfg.seq_len;
    assert!(n >= 1 && n < seq, "N has to be in [1, seq)");
    let ds = TextDataset::from_file(&data, seq).expect("could not read the corpus");
    let n_win = ds.num_windows();
    let n_val = (((n_win as f32) * 0.1).round() as usize).clamp(1, n_win / 2);
    let n_train = n_win - n_val;
    let ptrain = ptrain.min(n_train);
    let pstats = 30.min(ptrain);

    // Same squash as src/model/clock.rs (alpha_squash), reading the same envs.
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

    let nb = model.blocks.len();
    let d = model.cfg.dim;
    let fc = d + 1; // features + bias
    let lams = [0.01f64, 0.1, 1.0, 10.0];

    println!("eva state_probe: {weights} | {data} | N={n} seq={seq} blocks={nb} dim={d}");
    println!("  probe region per window: targets [{n}, {seq}-1) -> {} positions (label = target[t-N])", seq - n);
    println!("  fit on first {ptrain} train windows (stats on first {pstats}), eval on {n_val} val windows");
    println!("  state: fresh per window (mask_eval convention); squash temp={temp} antisat={antisat}\n");

    let mut alphas: Vec<Vec<f32>> = Vec::new();
    let mut betas: Vec<f32> = Vec::new();
    for b in &model.blocks {
        let Mixer::Clock(c) = &b.mixer else { panic!("not a ClockMem model") };
        alphas.push(c.log_clock.data.iter().map(|&lc| squash(lc)).collect());
        betas.push(c.beta.data[0]);
    }

    // Features for one position: A = cur[t] per block; B = recovered write.
    let feat_a = |curs: &[Vec<Vec<f32>>], bi: usize, t: usize| curs[bi][t].clone();
    let feat_b = |curs: &[Vec<Vec<f32>>], alphas: &[Vec<f32>], betas: &[f32], bi: usize, t: usize, n: usize| {
        (0..d)
            .map(|c| (curs[bi][t - n + 1][c] - alphas[bi][c] * curs[bi][t - n][c]) / betas[bi])
            .collect::<Vec<f32>>()
    };

    // Pass 0: per-feature mean/std over pstats windows, for A and B.
    let mut a_mean = vec![vec![0.0f64; d]; nb];
    let mut a_var = vec![vec![0.0f64; d]; nb];
    let mut b_mean = vec![vec![0.0f64; d]; nb];
    let mut b_var = vec![vec![0.0f64; d]; nb];
    let mut stats_cnt = 0usize;
    for wi in 0..pstats {
        let (input, _target) = ds.window(wi);
        let curs = forward_clock(&model, &input, &alphas, &betas);
        for t in n..seq - 1 {
            for bi in 0..nb {
                let a = feat_a(&curs, bi, t);
                let b = feat_b(&curs, &alphas, &betas, bi, t, n);
                for c in 0..d {
                    a_mean[bi][c] += a[c] as f64;
                    a_var[bi][c] += a[c] as f64 * a[c] as f64;
                    b_mean[bi][c] += b[c] as f64;
                    b_var[bi][c] += b[c] as f64 * b[c] as f64;
                }
            }
            stats_cnt += 1;
        }
        if wi % 10 == 0 { eprintln!("  stats: window {wi}/{pstats}"); }
    }
    let zscale = |mean: &[f64], var: &[f64]| -> Vec<f32> {
        (0..d).map(|c| {
            let m = mean[c] / stats_cnt as f64;
            let v = (var[c] / stats_cnt as f64 - m * m).max(1e-12);
            (v.sqrt() as f32).max(1e-8)
        }).collect()
    };
    let a_scl: Vec<Vec<f32>> = (0..nb).map(|bi| zscale(&a_mean[bi], &a_var[bi])).collect();
    let b_scl: Vec<Vec<f32>> = (0..nb).map(|bi| zscale(&b_mean[bi], &b_var[bi])).collect();

    // Pass 1: z-scored moments over ptrain windows.
    let mut a_xtx = vec![vec![0.0f64; fc * fc]; nb];
    let mut a_xty = vec![vec![0.0f64; fc * 256]; nb];
    let mut b_xtx = vec![vec![0.0f64; fc * fc]; nb];
    let mut b_xty = vec![vec![0.0f64; fc * 256]; nb];
    let mut a_cnt = vec![0usize; nb];
    let mut b_cnt = vec![0usize; nb];
    for wi in 0..ptrain {
        let (input, target) = ds.window(wi);
        let curs = forward_clock(&model, &input, &alphas, &betas);
        for t in n..seq - 1 {
            let label = target[t - n] as usize;
            for bi in 0..nb {
                let a = feat_a(&curs, bi, t);
                let za: Vec<f32> = (0..d).map(|c| a[c] / a_scl[bi][c]).collect();
                acc(&mut a_xtx[bi], &mut a_xty[bi], &mut a_cnt[bi], &za, label, fc);
                let b = feat_b(&curs, &alphas, &betas, bi, t, n);
                let zb: Vec<f32> = (0..d).map(|c| b[c] / b_scl[bi][c]).collect();
                acc(&mut b_xtx[bi], &mut b_xty[bi], &mut b_cnt[bi], &zb, label, fc);
            }
        }
        if wi % 100 == 0 { eprintln!("  fit: window {wi}/{ptrain}"); }
    }
    println!("samples (fit): A {} per block | B {} per block\n", a_cnt[0], b_cnt[0]);

    // Pass 2: val windows, collect z-scored features.
    let mut val_a: Vec<(Vec<Vec<f32>>, usize)> = Vec::new();
    let mut val_b: Vec<(Vec<Vec<f32>>, usize)> = Vec::new();
    for wi in n_train..n_win {
        let (input, target) = ds.window(wi);
        let curs = forward_clock(&model, &input, &alphas, &betas);
        for t in n..seq - 1 {
            let label = target[t - n] as usize;
            val_a.push((curs.iter().map(|_| Vec::new()).collect::<Vec<_>>(), label));
            val_b.push((curs.iter().map(|_| Vec::new()).collect::<Vec<_>>(), label));
        }
        // Store per-block z-scored features.
        let base = val_a.len() - (seq - n - 1);
        let mut idx = 0usize;
        for t in n..seq - 1 {
            for bi in 0..nb {
                let a = feat_a(&curs, bi, t);
                val_a[base + idx].0[bi] = (0..d).map(|c| a[c] / a_scl[bi][c]).collect();
                let b = feat_b(&curs, &alphas, &betas, bi, t, n);
                val_b[base + idx].0[bi] = (0..d).map(|c| b[c] / b_scl[bi][c]).collect();
            }
            idx += 1;
        }
        if wi % 20 == 0 { eprintln!("  val: window {wi}/{n_win}"); }
    }
    println!("samples (val): {}\n", val_a.len());

    println!("=== PROBE A: cur[t] -> input[t-N+1]  (single-state linear read) ===");
    for bi in 0..nb {
        let (lam, bpb, accv) = best_lambda(&a_xtx[bi], &a_xty[bi], fc, &lams, &val_a, bi);
        println!("  block {bi}: lam={lam:5}  VAL bpb={bpb:.4}  acc={accv:.3}%");
    }

    println!("\n=== PROBE B: w[t-N+1] finite-difference read (window K>=N+1) ===");
    for bi in 0..nb {
        let (lam, bpb, accv) = best_lambda(&b_xtx[bi], &b_xty[bi], fc, &lams, &val_b, bi);
        println!("  block {bi}: lam={lam:5}  VAL bpb={bpb:.4}  acc={accv:.3}%");
    }
    println!("\n(chance floor: 8.0000 bpb, 0.39% acc; best block matters, not the mean)\n");
}

fn best_lambda(
    xtx: &[f64], xty: &[f64], fc: usize, lams: &[f64],
    rows: &[(Vec<Vec<f32>>, usize)], bi: usize,
) -> (f64, f32, f32) {
    let mut best = (lams[0], f32::MAX, 0.0f32);
    for &lam in lams {
        let w = solve_ridge(xtx, xty, fc, lam);
        let (bpb, acc) = eval_block(&w, rows, bi, fc);
        if bpb < best.1 {
            best = (lam, bpb, acc);
        }
    }
    best
}

fn acc(xtx: &mut [f64], xty: &mut [f64], cnt: &mut usize, feat: &[f32], label: usize, fc: usize) {
    debug_assert_eq!(fc, feat.len());
    let mut x = vec![0.0f64; fc];
    for (i, &v) in feat.iter().enumerate() {
        x[i] = v as f64;
    }
    x[fc - 1] = 1.0; // bias column
    for i in 0..fc {
        let xi = x[i];
        for j in 0..fc {
            xtx[i * fc + j] += xi * x[j];
        }
        xty[i * 256 + label] += xi;
    }
    *cnt += 1;
}

/// Solve (X^T X + lam*I) W = X^T Y by Gaussian elimination with partial
/// pivoting. W comes back as [f][256].
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
        let d = a[col * f + col];
        if d.abs() < 1e-12 {
            continue;
        }
        for r in col + 1..f {
            let m = a[r * f + col] / d;
            if m.abs() < 1e-15 {
                continue;
            }
            for c in col..f {
                a[r * f + c] -= m * a[col * f + c];
            }
            for y in 0..256 {
                rhs[r * 256 + y] -= m * rhs[col * 256 + y];
            }
        }
    }
    let mut w = vec![0.0f64; f * 256];
    for col in (0..f).rev() {
        let d = a[col * f + col];
        if d.abs() < 1e-12 {
            continue;
        }
        for y in 0..256 {
            let mut s = rhs[col * 256 + y];
            for c in col + 1..f {
                s -= a[col * f + c] * w[c * 256 + y];
            }
            w[col * 256 + y] = s / d;
        }
    }
    w
}

/// Val bpb + accuracy for a probe on a single block, features = row[bi].
fn eval_block(w: &[f64], rows: &[(Vec<Vec<f32>>, usize)], bi: usize, fc: usize) -> (f32, f32) {
    let d = fc - 1;
    let mut nats = 0.0f64;
    let mut correct = 0usize;
    let mut count = 0usize;
    for (curs, label) in rows {
        let feat = &curs[bi];
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
        count += 1;
    }
    let bpb = (nats / count as f64 / std::f64::consts::LN_2) as f32;
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

/// Replicates the model's forward (same ops and order as src/model/block.rs
/// + src/tensor/ops.rs::clockmem_from) while collecting each block's clock
/// state per position. Fresh zero state per window. Returns [block][s][d].
fn forward_clock(
    model: &EvaModel,
    input: &[usize],
    alphas: &[Vec<f32>],
    betas: &[f32],
) -> Vec<Vec<Vec<f32>>> {
    let mut x = model.embed.embed(input);
    let mut all_cur: Vec<Vec<Vec<f32>>> = Vec::new();
    for (i, block) in model.blocks.iter().enumerate() {
        let h = ops::add(&x, &block.conv.forward(&block.norm0.forward(&x)));
        let mix = block.norm1.forward(&h);
        let Mixer::Clock(clock) = &block.mixer else { panic!("not a ClockMem model") };
        let q = clock.wq.forward(&mix);
        let k = clock.wk.forward(&mix);
        let v = clock.wv.forward(&mix);
        let g = ops::sigmoid(&clock.wg.forward(&mix));
        let (s, d) = (q.shape[0], q.shape[1]);
        let alpha = &alphas[i];
        let beta = betas[i];
        let mut cur = vec![0.0f32; d];
        let mut curs: Vec<Vec<f32>> = Vec::with_capacity(s);
        let mut m_vals = vec![0.0f32; s * d];
        for t in 0..s {
            for c in 0..d {
                cur[c] = alpha[c] * cur[c] + beta * k.data[t * d + c] * v.data[t * d + c];
            }
            for c in 0..d {
                m_vals[t * d + c] = q.data[t * d + c] * cur[c] * g.data[t * d + c];
            }
            curs.push(cur.clone());
        }
        all_cur.push(curs);
        let m = Tensor::new(m_vals, vec![s, d]);
        let h2 = ops::add(&h, &m);
        x = ops::add(&h2, &block.glu.forward(&block.norm2.forward(&h2)));
    }
    all_cur
}
