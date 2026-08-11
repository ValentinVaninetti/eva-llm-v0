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
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("comando desconocido: {} (usá `eva help`)", other)),
    }
}

fn cmd_train(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["data", "dim", "ffn", "blocks", "kernel", "eps", "seq", "epochs", "lr", "wd", "seed", "log", "out", "resume"])?;
    let data = flag(args, "data").ok_or("train requiere --data <archivo>")?;
    let mcfg = EvaConfig {
        vocab: 256,
        dim: flag_num(args, "dim", 256)?,
        ffn_dim: flag_num(args, "ffn", 512)?,
        blocks: flag_num(args, "blocks", 6)?,
        conv_kernel: flag_num(args, "kernel", 5)?,
        eps: flag_num(args, "eps", 1e-5)?,
        seq_len: flag_num(args, "seq", 64)?,
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
    };
    train(&tcfg, &mcfg)
}

fn cmd_gen(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "prompt", "tokens", "temp", "topk", "seed"])?;
    let weights = flag(args, "weights").ok_or("gen requiere --weights <archivo>")?;
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
    let weights = flag(args, "weights").ok_or("info requiere --weights <archivo>")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    println!("{}", describe(&model));
    Ok(())
}

fn cmd_gpu(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["m", "k", "n"])?;
    let m = flag_num(args, "m", 512)?;
    let k = flag_num(args, "k", 512)?;
    let n = flag_num(args, "n", 512)?;

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

    let t0 = std::time::Instant::now();
    let c = gpu.matmul(&a, &b, m, k, n)?;
    let t_gpu = t0.elapsed();

    let t0 = std::time::Instant::now();
    let mut cref = vec![0.0f32; m * n];
    crate::math::matmul(&a, &b, m, k, n, &mut cref);
    let t_cpu = t0.elapsed();

    let mut max_err = 0.0f32;
    for i in 0..m * n {
        max_err = max_err.max((c[i] - cref[i]).abs());
    }
    println!("matmul {}x{}x{}: GPU {:.3}ms | CPU {:.3}ms | max_err {:.2e}",
        m, k, n,
        t_gpu.as_secs_f64() * 1e3,
        t_cpu.as_secs_f64() * 1e3,
        max_err,
    );
    if max_err > 1e-2 {
        return Err(format!("el matmul de GPU difiere del CPU: max_err {:.2e}", max_err));
    }
    println!("OK: el backend Vulkan produce resultados correctos.");
    Ok(())
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
            .map_err(|_| format!("--{} espera un número, recibí: '{}'", name, v)),
    }
}

fn check_unknown(args: &[String], known: &[&str]) -> Result<(), String> {
    let known: Vec<String> = known.iter().map(|k| format!("--{}", k)).collect();
    for a in args {
        if a.starts_with('-') && !known.contains(a) {
            return Err(format!("argumento desconocido: {}", a));
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "eva v0 - LLM propio desde cero (cero deps, sin CUDA, AMD-friendly)\n\n\
         USO:\n\
         \x20 eva train --data <archivo> [opciones]\n\
         \x20 eva gen --weights <archivo> [--prompt texto] [--tokens N] [--temp F] [--topk N]\n\
         \x20 eva info --weights <archivo>\n\
         \x20 eva help\n\n\
         OPCIONES DE TRAIN:\n\
         \x20 --seq N       ventana de contexto (default 64)\n\
         \x20 --dim N       dimensión oculta (default 256)\n\
         \x20 --ffn N       dimensión FFN (default 512)\n\
         \x20 --blocks N    cantidad de bloques (default 6)\n\
         \x20 --kernel N    kernel de la conv causal (default 5)\n\
         \x20 --epochs N    épocas (default 10)\n\
         \x20 --lr F        learning rate (default 3e-4)\n\
         \x20 --wd F        weight decay (default 0.01)\n\
         \x20 --seed N      semilla (default 0)\n\
         \x20 --log N       cada cuántos pasos loguear (default 20)\n\
         \x20 --out RUTA    dónde guardar pesos (default eva.weights)\n\
         \x20 --resume RUTA reanudar desde pesos guardados"
    );
}
