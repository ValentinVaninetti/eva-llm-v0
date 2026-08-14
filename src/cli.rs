use crate::gen::generate;
use crate::gpu::Gpu;
use crate::model::EvaConfig;
use crate::rng::Rng;
use crate::save::{describe, load_model};
use crate::tokenizer::ByteTokenizer;
use crate::train::{train, TrainConfig};

pub fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        print_help();
        return Ok(());
    }
    match args[0].as_str() {
        "train" => cmd_train(&args[1..]),
        "gen" => cmd_gen(&args[1..]),
        "info" => cmd_info(&args[1..]),
        "gpu" => cmd_gpu(&args[1..]),
        "recall" => cmd_recall(&args[1..]),
        "ceiling" => cmd_ceiling(&args[1..]),
        "bet" => cmd_bet(&args[1..]),
        "stake" => cmd_stake(&args[1..]),
        "settle" => cmd_settle(&args[1..]),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown command: {} (try `eva help`)", other)),
    }
}

fn cmd_train(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["data", "dim", "ffn", "blocks", "kernel", "eps", "seq", "arch", "epochs", "lr", "wd", "seed", "log", "out", "resume", "val", "surprise", "local", "persist", "inorder", "gate-beta", "destroy-p"])?;
    let data = flag(args, "data").ok_or("train requires --data <file>")?;
    let mcfg = EvaConfig {
        vocab: 256,
        dim: flag_num(args, "dim", 256)?,
        ffn_dim: flag_num(args, "ffn", 512)?,
        blocks: flag_num(args, "blocks", 6)?,
        conv_kernel: flag_num(args, "kernel", 5)?,
        eps: flag_num(args, "eps", 1e-5)?,
        seq_len: flag_num(args, "seq", 64)?,
        arch: crate::model::Arch::from_str(&flag(args, "arch").unwrap_or_else(|| "clock".into()))?,
    };
    let tcfg = TrainConfig {
        data_path: data,
        epochs: flag_num(args, "epochs", 10)?,
        lr: flag_num(args, "lr", 3e-4)?,
        wd: flag_num(args, "wd", 0.01)?,
        seed: flag_num(args, "seed", 0)? as u64,
        log_every: flag_num(args, "log", 20)?,
        out_path: flag(args, "out").unwrap_or_else(|| "eva.weights".to_string()),
        resume: flag(args, "resume"),
        val_frac: flag_num(args, "val", 0.1)?,
        surprise: flag_num(args, "surprise", 0.0)?,
        local: args.iter().any(|a| a == "--local"),
        persist: args.iter().any(|a| a == "--persist"),
        inorder: args.iter().any(|a| a == "--inorder"),
        gate_beta: flag_num(args, "gate-beta", 0.0)?,
        destroy_p: flag_num(args, "destroy-p", 0.0)?,
    };
    train(&tcfg, &mcfg)
}

fn cmd_gen(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "prompt", "tokens", "temp", "topk", "seed"])?;
    let weights = flag(args, "weights").ok_or("gen requires --weights <file>")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let prompt = flag(args, "prompt").unwrap_or_default();
    let prompt = if prompt.is_empty() { " ".to_string() } else { prompt };
    let tokens = flag_num(args, "tokens", 128)?;
    let temp = flag_num(args, "temp", 0.8)?;
    let topk = flag_num(args, "topk", 40)?;
    let seed = flag_num(args, "seed", 0)? as u64;
    let mut rng = Rng::new(seed);
    let ids = ByteTokenizer::encode(&prompt);
    let out = generate(&model, &ids, tokens, temp, topk, &mut rng, model.cfg.seq_len);
    let text = ByteTokenizer::decode(&out);
    println!("{}", text);
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights"])?;
    let weights = flag(args, "weights").ok_or("info requires --weights <file>")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    println!("{}", describe(&model));
    Ok(())
}

/// Measures whether a small model PLUS a k-gram table reaches a large one.
///
/// Three splits and not two: the mixing weight is chosen on DEVELOPMENT and
/// reported on VALIDATION. Choosing lambda while looking at validation
/// would be tuning against the exam -- it would give the best possible
/// number and mean nothing.
fn cmd_recall(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "data", "seq"])?;
    let weights = flag(args, "weights").ok_or("recall requires --weights")?;
    let data = flag(args, "data").ok_or("recall requires --data")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let seq = flag_num(args, "seq", model.cfg.seq_len)?;

    let ds = crate::data::TextDataset::from_file(&data, seq).map_err(|e| e.to_string())?;
    let n = ds.num_windows();
    // 80 / 10 / 10, contiguous. The model has to have been trained with
    // --val 0.2 so it never saw development or validation.
    let end_train = n * 8 / 10;
    let end_dev = n * 9 / 10;

    let bytes_train: Vec<usize> = ds.ids[..end_train * seq].to_vec();
    let table = crate::recall::Recall::build(&bytes_train);
    println!("model {} params | table {} entries, ~{:.1} KB",
        model.param_count(), table.entries(), table.bytes() as f64 / 1024.0);

    let evaluate = |from: usize, to: usize, lambda: f32| -> (f32, f32) {
        let mut sum = 0.0;
        let mut count = 0;
        let mut cov = (0usize, 0usize);
        for wi in from..to {
            let (input, target) = ds.window(wi);
            let logits = model.forward(&input);
            let before: Vec<usize> = if wi == 0 { Vec::new() } else { ds.ids[..wi * seq].to_vec() };
            sum += crate::recall::mixed_loss(
                &logits.data, model.cfg.vocab, &before, &target, &table, lambda, &mut cov);
            count += 1;
        }
        (sum / count as f32, 100.0 * cov.0 as f32 / cov.1.max(1) as f32)
    };

    // Sweep over DEVELOPMENT.
    let mut best = (0.0f32, f32::INFINITY);
    println!("  lambda   development");
    for step in 0..=10 {
        let l = step as f32 * 0.05;
        let (p, _) = evaluate(end_train, end_dev, l);
        println!("   {l:.2}     {p:.4}");
        if p < best.1 {
            best = (l, p);
        }
    }

    // And a single pass over VALIDATION, with lambda already chosen.
    let (alone, coverage) = evaluate(end_dev, n, 0.0);
    let (with, _) = evaluate(end_dev, n, best.0);
    let bpb = |x: f32| x / std::f32::consts::LN_2;
    println!("\nVALIDATION (lambda {:.2} chosen on development)", best.0);
    println!("  model alone      {:.4}  |  {:.3} bits/byte", alone, bpb(alone));
    println!("  model + table    {:.4}  |  {:.3} bits/byte", with, bpb(with));
    println!("  improvement      {:.1}%", 100.0 * (alone - with) / alone);
    // If this is close to 100%, the test text is too similar to what was
    // stored and the result wouldn't hold up on new text.
    println!("  the table had something to say for {coverage:.1}% of the positions");
    let profile: Vec<String> = table
        .hit_profile()
        .iter()
        .filter(|(_, p)| *p > 0.05)
        .map(|(k, p)| format!("{k}:{p:.0}%"))
        .collect();
    println!("  where the hits come from: {}", profile.join("  "));
    Ok(())
}

/// 7. THE CEILING -- the first number of compute-by-influence.
///
/// Measures, on unseen text, how much it costs to remove each block on its
/// own: if there's a block that can be skipped while losing little, there's
/// real room for a mechanism that anticipates it; if removing any of them
/// destroys quality, the line dies here without ever writing the scheduler.
fn cmd_ceiling(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "data", "val", "skip"])?;
    let weights = flag(args, "weights").ok_or("ceiling requires --weights")?;
    let data = flag(args, "data").ok_or("ceiling requires --data")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let val: f32 = flag_num(args, "val", 0.1)?;

    let ds = crate::data::TextDataset::from_file(&data, model.cfg.seq_len).map_err(|e| e.to_string())?;
    let n = ds.num_windows();
    if n == 0 {
        return Err("the dataset is too small for the model's seq_len".into());
    }
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    if let Some(raw) = flag(args, "skip") {
        let mut skips: Vec<usize> = raw.split(',').map(|s| s.trim().parse::<usize>()
            .map_err(|_| format!("invalid --skip: '{raw}' (example: 2,3,4)")))
            .collect::<Result<_, _>>()?;
        skips.sort_unstable();
        if skips.windows(2).any(|w| w[0] == w[1]) {
            return Err("--skip cannot repeat blocks".into());
        }
        println!("eva: the model has to have been trained with --val {val:.1} to not have seen validation");
        let r = crate::ceiling::measure_combo(&model, &ds, n_train, n, &skips)?;
        println!("eva: REAL COMBINATION, blocks skipped {:?} (one pass)", r.skips);
        println!("  full           {:.4} bits/byte | {:.2} ms", r.full_bits_per_byte, r.full_time_s * 1e3);
        println!("  combination    {:.4} bits/byte | {:+.4} ({:+.1}%) | argmax changes {:.1}%",
            r.bits_per_byte, r.delta_bits, 100.0 * r.rel_delta, 100.0 * r.pred_change);
        println!("  real time      {:.2} ms | real savings {:.1}% | weights not read {:.1} KB",
            r.time_total_s * 1e3, 100.0 * (r.full_time_s - r.time_total_s) / r.full_time_s.max(1e-12),
            r.weight_bytes as f64 / 1024.0);
        return Ok(());
    }
    println!("eva: model {} params | {} blocks | arch {} | dim {} ffn {} kernel {} | seq {}",
        model.param_count(), model.blocks.len(), model.cfg.arch.name(),
        model.cfg.dim, model.cfg.ffn_dim, model.cfg.conv_kernel, model.cfg.seq_len);
    println!("eva: {} windows, {} trained, {} validation (contiguous cut at the end)",
        n, n_train, n_val);
    println!("eva: the model has to have been trained with --val {val:.1} to not have seen validation");
    println!("eva: each variant runs once per window, state at zero (bet's convention)");

    let rep = crate::ceiling::measure(&model, &ds, n_train, n)?;

    println!("\n=== RETROSPECTIVE CEILING (one block out at a time, unseen text) ===");
    println!("  {n_val} windows, {} positions | full {:.4} bits/byte in {:.2} ms",
        rep.n_pos, rep.full_bits_per_byte, rep.full_time_s * 1e3);
    println!("  variant        bits/byte   D bits/byte   pred changes   time       saves   weights not read");
    println!("  full           {:.4}         -              -          {:6.2} ms      -          -",
        rep.full_bits_per_byte, rep.full_time_s * 1e3);
    for b in &rep.blocks {
        println!("  without block {:<2}    {:.4}      {:+.4} ({:+.1}%)   {:6.1}%     {:6.2} ms  {:5.1}%     {:6.1} KB",
            b.idx, b.bits_per_byte, b.delta_bits, 100.0 * b.rel_delta,
            100.0 * b.pred_change, b.time_total_s * 1e3,
            100.0 * b.block_time_s / rep.full_time_s,
            b.weight_bytes as f64 / 1024.0);
    }

    println!("\n=== IDEAL RULES (retrospective -- the maximum ceiling, not what a runtime gate would achieve) ===");
    println!("  the projected loss of skipping several is the SUM of the individual ones; the combined");
    println!("  run gets checked separately if the ceiling looks good. Each block costs the same in weights.");
    for threshold in [0.01f32, 0.02, 0.05] {
        let (_, t, w, projected) = crate::ceiling::ideal_rule(&rep, threshold);
        let who: Vec<String> = rep
            .blocks
            .iter()
            .filter(|b| b.rel_delta <= threshold)
            .map(|b| format!("b{}", b.idx))
            .collect();
        let who = if who.is_empty() { "none".to_string() } else { who.join(", ") };
        println!("  loss <= {:>3.0}%  -> skips [{}]  saves {t:5.1}% of time, {w:4.1}% of weight | projected {projected:.4} bits/byte",
            100.0 * threshold, who);
    }

    let (skip, t, _, _) = crate::ceiling::ideal_rule(&rep, 0.05);
    if skip > 0 {
        println!("\n  VERDICT: there are block(s) that can be skipped for <=5% bits/byte loss ->");
        println!("  real room for conditional compute (ceiling {t:.1}% of time).");
        println!("  Next step: how to anticipate influence without paying for the block.");
    } else {
        println!("\n  VERDICT: removing any block costs more than 5% bits/byte ->");
        println!("  low ceiling, the conditional compute line dies here.");
    }
    Ok(())
}

/// 8. THAT IT BETS -- the first number that kills it.
///
/// Measures whether the confidence the model DECLARES tracks what it
/// actually gets right, on unseen text. The cut is the same contiguous one
/// `train` used: the last `--val` windows are the exam, never touched.
fn cmd_bet(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "data", "seq", "val", "bins", "train"])?;
    let weights = flag(args, "weights").ok_or("bet requires --weights")?;
    let data = flag(args, "data").ok_or("bet requires --data")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let seq = flag_num(args, "seq", model.cfg.seq_len)?;
    let val: f32 = flag_num(args, "val", 0.1)?;
    let bins = flag_num(args, "bins", 10)?;
    let measure_train = args.iter().any(|a| a == "--train");

    let ds = crate::data::TextDataset::from_file(&data, seq).map_err(|e| e.to_string())?;
    let n = ds.num_windows();
    if n == 0 {
        return Err("the dataset is too small for the model's seq_len".into());
    }
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    println!("eva: {} windows, {} trained, {} validation (contiguous cut at the end)",
        n, n_train, n_val);
    println!("eva: the model has to have been trained with --val {val:.1} to not have seen validation");

    if measure_train {
        let t = crate::bet::measure(&model, &ds, 0, n_train, bins);
        print_calibration(&t, "TRAINING (text the model DID see)", bins);
    }
    let obs = crate::bet::scan(&model, &ds, n_train, n);
    let mut cal = crate::bet::Calibration::new(bins);
    for o in &obs {
        cal.add(o.conf, o.margin, o.correct, o.p_target);
    }
    print_calibration(&cal, "VALIDATION (text the model did NOT see)", bins);

    println!("\n=== SPANS (can a span's confidence be predicted from what's already there?) ===");
    println!("  good = fraction of correct positions within the span");
    println!("  r(mean) = average of p[argmax]   r(min) = weakest link");
    println!("  r(geo_truth) = p[truth] of the span (CEILING, uses the answer)");
    println!("  r(geo_argmax) = p[argmax] of the span (BAR usable at generation)");
    println!("  r(magnitude) = length of the vector AT THE START");
    for &l in &[8usize, 16, 32] {
        let s = crate::bet::span_analysis(&obs, model.cfg.seq_len, l);
        println!("  span {l:>2}: n={:>5}  good {:.1}%  | r(mean) {:.3}  r(min) {:.3}  r(geo_truth) {:.3}  r(geo_argmax) {:.3}  r(magnitude) {:.3}",
            s.n, 100.0 * s.good_global, s.r_media, s.r_min, s.r_geomean, s.r_geomean_argmax, s.r_magnitude);
    }
    Ok(())
}

/// 8. THAT IT BETS -- the decisive experiment: the stake head.
///
/// Trains a PROBE (linear projection of the hidden state at the start of
/// the span) over an already-trained, frozen model, and measures it on the
/// same validation as `bet`: does r(stake vs good) beat the free bar (mean
/// p[argmax] approx 0.67)? If it doesn't beat it, the bet is done for free
/// with the geomean and #8 closes. If it does, the signal exists before
/// spending anything and the head earns its cost.
fn cmd_stake(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "data", "val", "span", "epochs", "lr", "seed", "norm"])?;
    let weights = flag(args, "weights").ok_or("stake requires --weights")?;
    let data = flag(args, "data").ok_or("stake requires --data")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let span = flag_num(args, "span", 16)?;
    let epochs = flag_num(args, "epochs", 1)?;
    let lr = flag_num(args, "lr", 3e-4)?;
    let seed = flag_num(args, "seed", 0)? as u64;
    let val: f32 = flag_num(args, "val", 0.1)?;
    let post_norm = args.iter().any(|a| a == "--norm");

    let ds = crate::data::TextDataset::from_file(&data, model.cfg.seq_len).map_err(|e| e.to_string())?;
    let n = ds.num_windows();
    if n == 0 {
        return Err("the dataset is too small for the model's seq_len".into());
    }
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    println!("eva: model {} params | head {} params (w+b) | probe over {} state",
        model.param_count(), model.cfg.dim + 1, if post_norm { "POST-norm" } else { "PRE-norm" });
    println!("eva: {} windows, {} train the probe, {} validation (same contiguous cut as bet)",
        n, n_train, n_val);
    println!("eva: span = {span} positions, the model stays frozen (hidden detached)");

    let (r_train, r_stake, n_spans) =
        crate::stake::train_and_measure(&model, &ds, n_train, n_val, span, epochs, lr, seed, post_norm);

    // The free bars, measured with the same machinery as bet, on the same
    // cut. The head has to beat the bar; getting close to the ceiling is a
    // bonus.
    let obs = crate::bet::scan(&model, &ds, n_train, n);
    let s = crate::bet::span_analysis(&obs, model.cfg.seq_len, span);
    println!("\n=== THE NUMBER ===");
    println!("  r(stake vs good)   = {r_stake:.3}  (the head, before spending; train {r_train:.3})");
    println!("  r(mean p[argmax])  = {:.3}  (usable post-hoc BAR)", s.r_media);
    println!("  r(geo p[truth])    = {:.3}  (CEILING, uses the answer)", s.r_geomean);
    println!("  r(magnitude)       = {:.3}  (the free signal that does NOT exist)", s.r_magnitude);
    let verdict = if r_stake > s.r_media {
        format!("THE HEAD BEATS THE BAR {:.3} -> the per-span bet lives", s.r_media)
    } else {
        format!("THE HEAD DOES NOT BEAT THE BAR {:.3} -> the bet is done for free with the mean", s.r_media)
    };
    println!("  VERDICT ({n_spans} spans): {verdict}");
    Ok(())
}

/// 9. FREEZE AND UNFREEZE -- the first number that kills it.
///
/// Measures, window by window, what fraction of the parameters stay still
/// for SEVERAL consecutive windows (not just once -- that happens all the
/// time from noise and means nothing). If that fraction is zero or doesn't
/// grow with training, there's nothing to freeze and point 9 closes without
/// ever writing the mechanism.
fn cmd_settle(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["data", "dim", "ffn", "blocks", "seq", "epochs", "lr", "wd", "seed", "val", "every", "streak"])?;
    let cfg = crate::settle::SettleConfig {
        data_path: flag(args, "data").ok_or("settle requires --data <file>")?,
        dim: flag_num(args, "dim", 256)?,
        ffn: flag_num(args, "ffn", 512)?,
        blocks: flag_num(args, "blocks", 4)?,
        seq_len: flag_num(args, "seq", 64)?,
        epochs: flag_num(args, "epochs", 6)?,
        lr: flag_num(args, "lr", 3e-4)?,
        wd: flag_num(args, "wd", 0.01)?,
        seed: flag_num(args, "seed", 7)? as u64,
        val_frac: flag_num(args, "val", 0.1)?,
        every: flag_num(args, "every", 100)?,
        streak: flag_num(args, "streak", 3)?,
    };
    crate::settle::run(&cfg)
}

fn print_calibration(cal: &crate::bet::Calibration, title: &str, bins: usize) {
    let acc = cal.accuracy();
    println!("\n=== {title} ===");
    println!("  {:<4} {:>8} {:>10} {:>10}", "bin", "n", "conf", "accuracy");
    for b in 0..bins {
        if cal.bin_n(b) == 0 {
            continue;
        }
        println!("  {:<4} {:>8} {:>10.3} {:>10.1}%",
            b, cal.bin_n(b), cal.bin_conf(b), 100.0 * cal.bin_accuracy(b));
    }
    println!("  overall accuracy {:.1}%  |  ECE {:.3}", 100.0 * acc, cal.ece());
    println!("  r(mean conf vs accuracy per bin) = {:.3}", cal.bin_corr());
    if let Some((top, bottom)) = cal.top_vs_bottom() {
        println!("  accuracy of most confident bin {:.1}%  vs  least confident bin {:.1}%",
            100.0 * top, 100.0 * bottom);
    }
    println!("  r(margin vs accuracy) = {:.3}", cal.margin_r());
    let (yes, no) = cal.pt_calibration();
    println!("  p(truth) when correct {:.3}  vs  when not {:.3}", yes, no);
    let r = cal.bin_corr();
    let verdict = match (r, cal.top_vs_bottom()) {
        (r, Some((top, bottom))) if r >= 0.5 && top > bottom => "CONFIDENCE TRACKS ACCURACY -> #8 lives",
        _ => "CONFIDENCE DOES NOT TRACK ACCURACY -> #8 dies here",
    };
    println!("  VERDICT: {verdict}");
}

fn cmd_gpu(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["m", "k", "n", "iters"])?;
    let m = flag_num(args, "m", 512)?;
    let k = flag_num(args, "k", 512)?;
    let n = flag_num(args, "n", 512)?;
    let iters = flag_num(args, "iters", 20)?;

    let gpu = Gpu::init()?;
    let info = gpu.info();
    println!("GPU: {} | vendor 0x{:04x} device 0x{:04x} | api {}.{}.{}",
        info.name,
        info.vendor_id,
        info.device_id,
        (info.api_version >> 22) & 0x7f,
        (info.api_version >> 12) & 0x3ff,
        info.api_version & 0xfff,
    );

    let mut rng = Rng::new(0xE7A1);
    let a: Vec<f32> = (0..m * k).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.uniform(-1.0, 1.0)).collect();

    // The first call is loaded down with buffer creation; the rest is what
    // a training run sees, multiplying the same shapes thousands of times.
    // Measuring only once mixed the two things into a number that was
    // neither.
    let mut c = Vec::new();
    let (gpu_cold, gpu_steady) = timed(iters, || {
        c = gpu.matmul(&a, &b, m, k, n)?;
        Ok(())
    })?;

    let mut cref = vec![0.0f32; m * n];
    let (_, cpu_steady) = timed(iters, || {
        crate::math::matmul(&a, &b, m, k, n, &mut cref);
        Ok(())
    })?;

    let max_err = c.iter().zip(&cref).fold(0.0f32, |w, (x, y)| w.max((x - y).abs()));
    println!(
        "matmul {m}x{k}x{n} ({iters} runs)\n  \
         GPU  {:7.3} ms  (first {:7.3} ms, including buffer creation)\n  \
         CPU  {:7.3} ms\n  \
         max_err {max_err:.2e}",
        ms(gpu_steady),
        ms(gpu_cold),
        ms(cpu_steady),
    );
    if max_err > 1e-2 {
        return Err(format!("GPU matmul differs from CPU: max_err {max_err:.2e}"));
    }
    let (fast, slow, times) = if gpu_steady < cpu_steady {
        ("GPU", "CPU", ms(cpu_steady) / ms(gpu_steady))
    } else {
        ("CPU", "GPU", ms(gpu_steady) / ms(cpu_steady))
    };
    println!("OK: results correct. In steady state {fast} wins by {times:.2}x over {slow}.");
    Ok(())
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Runs `f` `iters` times and returns (the first, the average of the rest).
///
/// Separating them is the point: the first includes everything that gets
/// paid once, and averaging it in would hide exactly what we want to see.
fn timed(
    iters: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<(std::time::Duration, std::time::Duration), String> {
    let t0 = std::time::Instant::now();
    f()?;
    let first = t0.elapsed();
    if iters <= 1 {
        return Ok((first, first));
    }
    let t0 = std::time::Instant::now();
    for _ in 1..iters {
        f()?;
    }
    Ok((first, t0.elapsed() / (iters - 1) as u32))
}

fn flag(args: &[String], name: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == format!("--{}", name) {
            return args.get(i + 1).cloned();
        }
    }
    None
}

fn flag_num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match flag(args, name) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|_| format!("--{} expects a number, got: '{}'", name, v)),
    }
}

fn check_unknown(args: &[String], known: &[&str]) -> Result<(), String> {
    let known: Vec<String> = known.iter().map(|k| format!("--{}", k)).collect();
    for a in args {
        if a.starts_with('-') && !known.contains(a) {
            return Err(format!("unknown argument: {}", a));
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "eva v0 - from-scratch LLM (zero deps, no CUDA, AMD-friendly)\n\n\
         USAGE:\n\
         \x20 eva train --data <file> [options]\n\
         \x20 eva gen --weights <file> [--prompt text] [--tokens N] [--temp F] [--topk N]\n\
         \x20 eva info --weights <file>\n\
         \x20 eva ceiling --weights <file> --data <file> [--val F]\n\
         \x20 eva bet --weights <file> --data <file> [--val F] [--bins N]\n\
         \x20 eva stake --weights <file> --data <file> [--span N] [--epochs N] [--lr F]\n\
         \x20 eva help\n\n\
         TRAIN OPTIONS:\n\
         \x20 --seq N       context window (default 64)\n\
         \x20 --dim N       hidden dimension (default 256)\n\
         \x20 --ffn N       FFN dimension (default 512)\n\
         \x20 --blocks N    number of blocks (default 6)\n\
         \x20 --kernel N    causal conv kernel (default 5)\n\
         \x20 --epochs N    epochs (default 10)\n\
         \x20 --lr F        learning rate (default 3e-4)\n\
         \x20 --wd F        weight decay (default 0.01)\n\
         \x20 --seed N      seed (default 0)\n\
         \x20 --log N       log every N steps (default 20)\n\
         \x20 --out PATH    where to save weights (default eva.weights)\n\
         \x20 --resume PATH resume from saved weights"
    );
}
