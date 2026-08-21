//! Claude's follow-up to GPT's diagnosis after `activation_probe.rs` localized
//! the signal loss to `m = q*read*g` (b0.read ~38.6% val acc vs b0.m ~0.84%).
//!
//! Hypothesis under test: a product of three learned gates (q, read, g) is a
//! multiplicative-gradient trap -- the gradient reaching q or g is
//! proportional to the OTHER two factors, so once either saturates near zero
//! for a channel, the gradient needed to reopen it also collapses. Same
//! family of bug as the `log_clock`/alpha saturation trap from last night,
//! one layer over (the output gate, not the decay).
//!
//! Per channel, per block: mean |q_c|, mean |g_c|, mean |q_c*g_c|, and a
//! single-channel linear probe (2 params: slope+bias over 256 classes) using
//! ONLY read[:,c] as a feature, at the masked long-range positions (t>=N,
//! label=input[t-N+1]). If oracle-useful channels (high single-channel probe
//! accuracy) are exactly the ones with |q_c| or |g_c| near zero, that is
//! direct evidence for the multiplicative-gate-starvation hypothesis.
//!
//! USAGE: cargo run --release --example gate_channel_probe -- <weights> <data> <N> [train_windows]
//!   requires EVA_READ_WIN=K, EVA_READ_DF_N=N, EVA_READ_ORACLE=1 set (same
//!   oracle checkpoint/recipe as activation_probe.rs).

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::nn::Module;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::ops as ops;

fn main() {
    let weights = std::env::args().nth(1).expect("usage: gate_channel_probe <weights> <data> <N> [train_windows]");
    let data = std::env::args().nth(2).expect("usage: gate_channel_probe <weights> <data> <N> [train_windows]");
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok())
        .expect("usage: gate_channel_probe <weights> <data> <N> [train_windows]");
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

    println!("eva gate_channel_probe: {weights} | {data} | N={n} seq={seq} blocks={nb} dim={d} K={kwin}");
    println!("  masked region per window: t in [{n}, {seq}) -> {} positions (label = input[t-N+1])\n", seq - n);

    for bi in 0..nb {
        let Mixer::Clock(clock) = &model.blocks[bi].mixer else { panic!("not a ClockMem model") };
        let alpha: Vec<f32> = clock.log_clock.data.iter().map(|&lc| squash(lc)).collect();
        let beta = clock.beta.data[0];

        // Oracle taps (exact recovery), fixed for this block.
        let mut w = vec![0.0f32; kwin * d];
        for c in 0..d {
            w[c] = 1.0;
            w[(n - 1) * d + c] = 1.0 / beta;
            w[n * d + c] = -alpha[c] / beta;
        }

        // Per-channel gate magnitude accumulators + single-channel ridge stats.
        let mut sum_absq = vec![0.0f64; d];
        let mut sum_absg = vec![0.0f64; d];
        let mut sum_absqg = vec![0.0f64; d];
        let mut gate_cnt = 0usize;
        // xtx: [a b; b c] per channel (a=sum x^2, b=sum x, c=count); xty: [256] per channel (sum x per class) + [256] (count per class, for the bias row).
        let mut sxx = vec![0.0f64; d];
        let mut sx = vec![0.0f64; d];
        let mut sxy = vec![vec![0.0f64; 256]; d];
        let mut scnt = vec![vec![0.0f64; 256]; d]; // per-class count, for the bias row's RHS
        let mut fit_cnt = 0usize;

        for wi in 0..n_win {
            if wi >= ptrain && wi < n_train {
                continue; // skip the unused middle slice, only fit on [0,ptrain) and eval on [n_train,n_win)
            }
            let (input, target) = ds.window(wi);
            let mut x = model.embed.embed(&input);
            for bj in 0..bi {
                let b = &model.blocks[bj];
                let h = ops::add(&x, &b.conv.forward(&b.norm0.forward(&x)));
                let mix = b.norm1.forward(&h);
                let Mixer::Clock(c2) = &b.mixer else { panic!("not a ClockMem model") };
                let a2: Vec<f32> = c2.log_clock.data.iter().map(|&lc| squash(lc)).collect();
                let beta2 = c2.beta.data[0];
                let q2 = c2.wq.forward(&mix);
                let k2 = c2.wk.forward(&mix);
                let v2 = c2.wv.forward(&mix);
                let g2 = ops::sigmoid(&c2.wg.forward(&mix));
                let s = q2.shape[0];
                let dd = model.cfg.dim;
                let mut w2 = vec![0.0f32; kwin * dd];
                for c in 0..dd {
                    w2[c] = 1.0;
                    w2[(n - 1) * dd + c] = 1.0 / beta2;
                    w2[n * dd + c] = -a2[c] / beta2;
                }
                let mut cur = vec![0.0f32; dd];
                let mut state = vec![vec![0.0f32; dd]; s];
                let mut m = vec![0.0f32; s * dd];
                for t in 0..s {
                    for c in 0..dd {
                        cur[c] = a2[c] * cur[c] + beta2 * k2.data[t * dd + c] * v2.data[t * dd + c];
                        state[t][c] = cur[c];
                    }
                    let jmax = kwin.min(t + 1);
                    for c in 0..dd {
                        let mut r = 0.0f32;
                        for j in 0..jmax {
                            r += w2[j * dd + c] * state[t - j][c];
                        }
                        m[t * dd + c] = q2.data[t * dd + c] * r * g2.data[t * dd + c];
                    }
                }
                let h2 = ops::add(&h, &eva_llm_v0::tensor::Tensor::new(m, vec![s, dd]));
                x = ops::add(&h2, &b.glu.forward(&b.norm2.forward(&h2)));
            }
            // Block `bi` itself: collect q, g, read.
            let b = &model.blocks[bi];
            let h = ops::add(&x, &b.conv.forward(&b.norm0.forward(&x)));
            let mix = b.norm1.forward(&h);
            let q = clock.wq.forward(&mix);
            let k = clock.wk.forward(&mix);
            let v = clock.wv.forward(&mix);
            let g = ops::sigmoid(&clock.wg.forward(&mix));
            let s = q.shape[0];
            let mut cur = vec![0.0f32; d];
            let mut state = vec![vec![0.0f32; d]; s];
            let mut read = vec![vec![0.0f32; d]; s];
            for t in 0..s {
                for c in 0..d {
                    cur[c] = alpha[c] * cur[c] + beta * k.data[t * d + c] * v.data[t * d + c];
                    state[t][c] = cur[c];
                }
                let jmax = kwin.min(t + 1);
                for c in 0..d {
                    let mut r = 0.0f32;
                    for j in 0..jmax {
                        r += w[j * d + c] * state[t - j][c];
                    }
                    read[t][c] = r;
                }
            }

            let is_fit = wi < ptrain;
            for t in n..seq {
                let label = target[t - n] as usize;
                if is_fit {
                    for c in 0..d {
                        let x_ = read[t][c] as f64;
                        sxx[c] += x_ * x_;
                        sx[c] += x_;
                        sxy[c][label] += x_;
                        scnt[c][label] += 1.0;
                    }
                    fit_cnt += 1;
                } else {
                    for c in 0..d {
                        sum_absq[c] += (q.data[t * d + c] as f64).abs();
                        sum_absg[c] += (g.data[t * d + c] as f64).abs();
                        sum_absqg[c] += (q.data[t * d + c] as f64 * g.data[t * d + c] as f64).abs();
                    }
                    gate_cnt += 1;
                }
            }
        }

        // Solve the 2-param (slope, bias) softmax regression per channel via ridge (lam=1.0 fixed, small problem).
        // Closed-form only -- no forward pass needed here, all channels at once.
        let lam = 1.0f64;
        let total_fit = fit_cnt as f64;
        let mut slope = vec![vec![0.0f64; 256]; d];
        let mut bias = vec![vec![0.0f64; 256]; d];
        for c in 0..d {
            let a = sxx[c] + lam;
            let b_ = sx[c];
            let cc = total_fit + lam;
            let det = a * cc - b_ * b_;
            if det.abs() < 1e-9 {
                continue;
            }
            for y in 0..256 {
                let rx = sxy[c][y];
                let ry = scnt[c][y];
                slope[c][y] = (cc * rx - b_ * ry) / det;
                bias[c][y] = (a * ry - b_ * rx) / det;
            }
        }

        // Single pass over val windows: forward ONCE per window (all channels
        // together, same as the fit loop), then score every channel's probe
        // against its own precomputed slope/bias.
        let mut correct = vec![0usize; d];
        let mut tot = 0usize;
        for wi in n_train..n_win {
            let (input, target) = ds.window(wi);
            let mut x = model.embed.embed(&input);
            for bj in 0..bi {
                let bb = &model.blocks[bj];
                let h = ops::add(&x, &bb.conv.forward(&bb.norm0.forward(&x)));
                let mix = bb.norm1.forward(&h);
                let Mixer::Clock(c2) = &bb.mixer else { panic!("not a ClockMem model") };
                let a2: Vec<f32> = c2.log_clock.data.iter().map(|&lc| squash(lc)).collect();
                let beta2 = c2.beta.data[0];
                let q2 = c2.wq.forward(&mix);
                let k2 = c2.wk.forward(&mix);
                let v2 = c2.wv.forward(&mix);
                let g2 = ops::sigmoid(&c2.wg.forward(&mix));
                let s = q2.shape[0];
                let dd = model.cfg.dim;
                let mut w2 = vec![0.0f32; kwin * dd];
                for cc2 in 0..dd {
                    w2[cc2] = 1.0;
                    w2[(n - 1) * dd + cc2] = 1.0 / beta2;
                    w2[n * dd + cc2] = -a2[cc2] / beta2;
                }
                let mut cur = vec![0.0f32; dd];
                let mut state = vec![vec![0.0f32; dd]; s];
                let mut m = vec![0.0f32; s * dd];
                for t in 0..s {
                    for cc2 in 0..dd {
                        cur[cc2] = a2[cc2] * cur[cc2] + beta2 * k2.data[t * dd + cc2] * v2.data[t * dd + cc2];
                        state[t][cc2] = cur[cc2];
                    }
                    let jmax = kwin.min(t + 1);
                    for cc2 in 0..dd {
                        let mut r = 0.0f32;
                        for j in 0..jmax {
                            r += w2[j * dd + cc2] * state[t - j][cc2];
                        }
                        m[t * dd + cc2] = q2.data[t * dd + cc2] * r * g2.data[t * dd + cc2];
                    }
                }
                let h2 = ops::add(&h, &eva_llm_v0::tensor::Tensor::new(m, vec![s, dd]));
                x = ops::add(&h2, &bb.glu.forward(&bb.norm2.forward(&h2)));
            }
            let bb = &model.blocks[bi];
            let h = ops::add(&x, &bb.conv.forward(&bb.norm0.forward(&x)));
            let mix = bb.norm1.forward(&h);
            let k = clock.wk.forward(&mix);
            let v = clock.wv.forward(&mix);
            let s = k.shape[0];
            let mut cur = vec![0.0f32; d];
            let mut state = vec![vec![0.0f32; d]; s];
            let mut read = vec![vec![0.0f32; d]; s];
            for t in 0..s {
                for c in 0..d {
                    cur[c] = alpha[c] * cur[c] + beta * k.data[t * d + c] * v.data[t * d + c];
                    state[t][c] = cur[c];
                }
                let jmax = kwin.min(t + 1);
                for c in 0..d {
                    let mut r = 0.0f32;
                    for j in 0..jmax {
                        r += w[j * d + c] * state[t - j][c];
                    }
                    read[t][c] = r;
                }
            }
            for t in n..seq {
                let label = target[t - n] as usize;
                for c in 0..d {
                    let r = read[t][c] as f64;
                    let mut best = 0usize;
                    let mut best_val = slope[c][0] * r + bias[c][0];
                    for y in 1..256 {
                        let v = slope[c][y] * r + bias[c][y];
                        if v > best_val {
                            best_val = v;
                            best = y;
                        }
                    }
                    if best == label {
                        correct[c] += 1;
                    }
                }
                tot += 1;
            }
        }
        let acc_per_c: Vec<f32> = correct.iter().map(|&c| 100.0 * c as f32 / tot.max(1) as f32).collect();

        let mean_absq: Vec<f32> = sum_absq.iter().map(|&s| (s / gate_cnt.max(1) as f64) as f32).collect();
        let mean_absg: Vec<f32> = sum_absg.iter().map(|&s| (s / gate_cnt.max(1) as f64) as f32).collect();
        let mean_absqg: Vec<f32> = sum_absqg.iter().map(|&s| (s / gate_cnt.max(1) as f64) as f32).collect();

        // Pearson correlation: single-channel probe acc vs |q|, |g|, |q*g|.
        let corr = |xs: &[f32], ys: &[f32]| -> f32 {
            let n = xs.len() as f64;
            let mx: f64 = xs.iter().map(|&v| v as f64).sum::<f64>() / n;
            let my: f64 = ys.iter().map(|&v| v as f64).sum::<f64>() / n;
            let mut sxy = 0.0f64;
            let mut sxx = 0.0f64;
            let mut syy = 0.0f64;
            for i in 0..xs.len() {
                let dx = xs[i] as f64 - mx;
                let dy = ys[i] as f64 - my;
                sxy += dx * dy;
                sxx += dx * dx;
                syy += dy * dy;
            }
            (sxy / (sxx.sqrt() * syy.sqrt() + 1e-12)) as f32
        };

        println!("=== block {bi} ===");
        println!("  corr(single-channel acc, |q|)   = {:+.3}", corr(&acc_per_c, &mean_absq));
        println!("  corr(single-channel acc, |g|)   = {:+.3}", corr(&acc_per_c, &mean_absg));
        println!("  corr(single-channel acc, |q*g|) = {:+.3}", corr(&acc_per_c, &mean_absqg));

        let mut idx: Vec<usize> = (0..d).collect();
        idx.sort_by(|&a, &b| acc_per_c[b].partial_cmp(&acc_per_c[a]).unwrap());
        println!("  top 8 channels by single-channel long-range acc:");
        for &c in idx.iter().take(8) {
            println!("    ch{c:<4} acc={:6.2}%  |q|={:.4}  |g|={:.4}  |q*g|={:.4}", acc_per_c[c], mean_absq[c], mean_absg[c], mean_absqg[c]);
        }
        println!("  bottom 8 channels by single-channel long-range acc:");
        for &c in idx.iter().rev().take(8) {
            println!("    ch{c:<4} acc={:6.2}%  |q|={:.4}  |g|={:.4}  |q*g|={:.4}", acc_per_c[c], mean_absq[c], mean_absg[c], mean_absqg[c]);
        }
        println!();
    }
}
