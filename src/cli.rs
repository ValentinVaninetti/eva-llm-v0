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
        "bet" => cmd_bet(&args[1..]),
        "stake" => cmd_stake(&args[1..]),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("comando desconocido: {} (usá `eva help`)", other)),
    }
}

fn cmd_train(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["data", "dim", "ffn", "blocks", "kernel", "eps", "seq", "arch", "epochs", "lr", "wd", "seed", "log", "out", "resume", "val", "surprise", "local", "persist", "inorder"])?;
    let data = flag(args, "data").ok_or("train requiere --data <archivo>")?;
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

/// Mide si un modelo chico MÁS una tabla de k-gramas alcanza a uno grande.
///
/// Tres particiones y no dos: el peso de mezcla se elige sobre DESARROLLO y se
/// reporta sobre VALIDACIÓN. Elegir lambda mirando validación sería ajustar
/// contra el examen -- daría el mejor número posible y no significaría nada.
fn cmd_recall(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "data", "seq"])?;
    let weights = flag(args, "weights").ok_or("recall requiere --weights")?;
    let data = flag(args, "data").ok_or("recall requiere --data")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let seq = flag_num(args, "seq", model.cfg.seq_len)?;

    let ds = crate::data::TextDataset::from_file(&data, seq).map_err(|e| e.to_string())?;
    let n = ds.num_windows();
    // 80 / 10 / 10, contiguo. El modelo tiene que haberse entrenado con
    // --val 0.2 para que no haya visto ni desarrollo ni validación.
    let fin_train = n * 8 / 10;
    let fin_dev = n * 9 / 10;

    let bytes_train: Vec<usize> = ds.ids[..fin_train * seq].to_vec();
    let tabla = crate::recall::Recall::build(&bytes_train);
    println!("modelo {} params | tabla {} entradas, ~{:.1} KB",
        model.param_count(), tabla.entries(), tabla.bytes() as f64 / 1024.0);

    let evaluar = |desde: usize, hasta: usize, lambda: f32| -> (f32, f32) {
        let mut suma = 0.0;
        let mut cuenta = 0;
        let mut cob = (0usize, 0usize);
        for wi in desde..hasta {
            let (input, target) = ds.window(wi);
            let logits = model.forward(&input);
            let antes: Vec<usize> = if wi == 0 { Vec::new() } else { ds.ids[..wi * seq].to_vec() };
            suma += crate::recall::mixed_loss(
                &logits.data, model.cfg.vocab, &antes, &target, &tabla, lambda, &mut cob);
            cuenta += 1;
        }
        (suma / cuenta as f32, 100.0 * cob.0 as f32 / cob.1.max(1) as f32)
    };

    // Barrido sobre DESARROLLO.
    let mut mejor = (0.0f32, f32::INFINITY);
    println!("  lambda   desarrollo");
    for paso in 0..=10 {
        let l = paso as f32 * 0.05;
        let (p, _) = evaluar(fin_train, fin_dev, l);
        println!("   {l:.2}     {p:.4}");
        if p < mejor.1 {
            mejor = (l, p);
        }
    }

    // Y una sola pasada por VALIDACIÓN, con el lambda ya elegido.
    let (solo, cobertura) = evaluar(fin_dev, n, 0.0);
    let (con, _) = evaluar(fin_dev, n, mejor.0);
    let bpb = |x: f32| x / std::f32::consts::LN_2;
    println!("\nVALIDACIÓN (lambda {:.2} elegido en desarrollo)", mejor.0);
    println!("  modelo solo      {:.4}  |  {:.3} bits/byte", solo, bpb(solo));
    println!("  modelo + tabla   {:.4}  |  {:.3} bits/byte", con, bpb(con));
    println!("  mejora           {:.1}%", 100.0 * (solo - con) / solo);
    // Si esto es casi 100%, el texto de prueba se parece demasiado al
    // guardado y el resultado no se sostendría con texto nuevo.
    println!("  la tabla tuvo algo que decir en el {cobertura:.1}% de las posiciones");
    let perfil: Vec<String> = tabla
        .hit_profile()
        .iter()
        .filter(|(_, p)| *p > 0.05)
        .map(|(k, p)| format!("{k}:{p:.0}%"))
        .collect();
    println!("  de dónde vienen los aciertos: {}", perfil.join("  "));
    Ok(())
}

/// 8. QUE APUESTE — el primer número que lo mata.
///
/// Mide si la confianza que el modelo DECLARA rastrea lo que realmente
/// acierta, sobre texto que no vio. El corte es el mismo contiguo que usó
/// `train`: las últimas `--val` ventanas son el examen, nunca tocadas.
fn cmd_bet(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "data", "seq", "val", "bins", "train"])?;
    let weights = flag(args, "weights").ok_or("bet requiere --weights")?;
    let data = flag(args, "data").ok_or("bet requiere --data")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let seq = flag_num(args, "seq", model.cfg.seq_len)?;
    let val: f32 = flag_num(args, "val", 0.1)?;
    let bins = flag_num(args, "bins", 10)?;
    let medir_train = args.iter().any(|a| a == "--train");

    let ds = crate::data::TextDataset::from_file(&data, seq).map_err(|e| e.to_string())?;
    let n = ds.num_windows();
    if n == 0 {
        return Err("el dataset es muy chico para el seq_len del modelo".into());
    }
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    println!("eva: {} ventanas, {} entrenadas, {} validación (corte contiguo al final)",
        n, n_train, n_val);
    println!("eva: el modelo tiene que haberse entrenado con --val {val:.1} para no haber visto validación");

    if medir_train {
        let t = crate::bet::measure(&model, &ds, 0, n_train, bins);
        print_calibration(&t, "ENTRENAMIENTO (texto que el modelo SÍ vio)", bins);
    }
    let obs = crate::bet::scan(&model, &ds, n_train, n);
    let mut cal = crate::bet::Calibration::new(bins);
    for o in &obs {
        cal.add(o.conf, o.margin, o.correct, o.p_target);
    }
    print_calibration(&cal, "VALIDACIÓN (texto que el modelo NO vio)", bins);

    println!("\n=== TRAMOS (¿la confianza del tramo se predice con lo que ya hay?) ===");
    println!("  bien = fracción de posiciones acertadas dentro del tramo");
    println!("  r(media) = promedio de p[argmax]   r(min) = eslabón débil");
    println!("  r(geo_verdad) = p[verdad] del tramo (TECHO, usa la respuesta)");
    println!("  r(geo_argmax) = p[argmax] del tramo (VARA usable al generar)");
    println!("  r(magnitud) = largo del vector AL ARRANCAR");
    for &l in &[8usize, 16, 32] {
        let s = crate::bet::span_analysis(&obs, model.cfg.seq_len, l);
        println!("  tramo {l:>2}: n={:>5}  bien {:.1}%  | r(media) {:.3}  r(min) {:.3}  r(geo_verdad) {:.3}  r(geo_argmax) {:.3}  r(magnitud) {:.3}",
            s.n, 100.0 * s.bien_global, s.r_media, s.r_min, s.r_geomean, s.r_geomean_argmax, s.r_magnitud);
    }
    Ok(())
}

/// 8. QUE APUESTE — el experimento decisivo: la cabeza de stake.
///
/// Entrena una SONDA (proyección lineal del estado oculto al arrancar el tramo)
/// sobre un modelo ya entrenado y congelado, y la mide en la misma validación
/// que `bet`: ¿r(stake vs bien) supera la vara libre (media p[argmax] ≈ 0.67)?
/// Si no la supera, la apuesta se hace gratis con la geomean y el 8 se cierra.
/// Si la supera, la señal existe antes de gastar y la cabeza vale su costo.
fn cmd_stake(args: &[String]) -> Result<(), String> {
    check_unknown(args, &["weights", "data", "val", "span", "epochs", "lr", "seed"])?;
    let weights = flag(args, "weights").ok_or("stake requiere --weights")?;
    let data = flag(args, "data").ok_or("stake requiere --data")?;
    let model = load_model(&weights).map_err(|e| e.to_string())?;
    let span = flag_num(args, "span", 16)?;
    let epochs = flag_num(args, "epochs", 1)?;
    let lr = flag_num(args, "lr", 3e-4)?;
    let seed = flag_num(args, "seed", 0)? as u64;
    let val: f32 = flag_num(args, "val", 0.1)?;

    let ds = crate::data::TextDataset::from_file(&data, model.cfg.seq_len).map_err(|e| e.to_string())?;
    let n = ds.num_windows();
    if n == 0 {
        return Err("el dataset es muy chico para el seq_len del modelo".into());
    }
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    println!("eva: modelo {} params | cabeza {} params (w+b)", model.param_count(), model.cfg.dim + 1);
    println!("eva: {} ventanas, {} entrenan la sonda, {} validación (mismo corte contiguo que bet)",
        n, n_train, n_val);
    println!("eva: tramo = {span} posiciones, el modelo queda congelado (hidden detached)");

    let (r_stake, n_tramos) =
        crate::stake::entrenar_y_medir(&model, &ds, n_train, n_val, span, epochs, lr, seed);

    // Las varas libres, medidas con la misma maquinaria que bet, sobre el
    // mismo corte. La cabeza tiene que superar la vara; acercarse al techo es
    // bonus.
    let obs = crate::bet::scan(&model, &ds, n_train, n);
    let s = crate::bet::span_analysis(&obs, model.cfg.seq_len, span);
    println!("\n=== EL NÚMERO ===");
    println!("  r(stake vs bien)   = {r_stake:.3}  (la cabeza, antes de gastar)");
    println!("  r(media p[argmax]) = {:.3}  (VARA usable post-hoc)", s.r_media);
    println!("  r(geo p[verdad])   = {:.3}  (TECHO, usa la respuesta)", s.r_geomean);
    println!("  r(magnitud)        = {:.3}  (la señal gratis que NO existe)", s.r_magnitud);
    let voto = if r_stake > s.r_media {
        format!("LA CABEZA SUPERA LA VARA {:.3} -> la apuesta por tramo vive", s.r_media)
    } else {
        format!("LA CABEZA NO SUPERA LA VARA {:.3} -> la apuesta se hace gratis con la media", s.r_media)
    };
    println!("  VEREDICTO ({n_tramos} tramos): {voto}");
    Ok(())
}

fn print_calibration(cal: &crate::bet::Calibration, titulo: &str, bins: usize) {
    let acc = cal.accuracy();
    println!("\n=== {titulo} ===");
    println!("  {:<4} {:>8} {:>10} {:>10}", "bin", "n", "conf", "acierto");
    for b in 0..bins {
        if cal.bin_n(b) == 0 {
            continue;
        }
        println!("  {:<4} {:>8} {:>10.3} {:>10.1}%",
            b, cal.bin_n(b), cal.bin_conf(b), 100.0 * cal.bin_accuracy(b));
    }
    println!("  acierto global {:.1}%  |  ECE {:.3}", 100.0 * acc, cal.ece());
    println!("  r(conf media vs acierto por bin) = {:.3}", cal.bin_corr());
    if let Some((top, bottom)) = cal.top_vs_bottom() {
        println!("  acierto bin más seguro {:.1}%  vs  bin menos seguro {:.1}%",
            100.0 * top, 100.0 * bottom);
    }
    println!("  r(margen vs acierto) = {:.3}", cal.margin_r());
    let (yes, no) = cal.pt_calibration();
    println!("  p(verdad) cuando acierta {:.3}  vs  cuando no {:.3}", yes, no);
    let r = cal.bin_corr();
    let voto = match (r, cal.top_vs_bottom()) {
        (r, Some((top, bottom))) if r >= 0.5 && top > bottom => "LA CONFIANZA RASTREA EL ACIERTO -> el 8 vive",
        _ => "LA CONFIANZA NO RASTREA EL ACIERTO -> el 8 muere acá",
    };
    println!("  VEREDICTO: {voto}");
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

    // La primera llamada carga con la creación de los buffers; el resto es lo
    // que ve un entrenamiento, que multiplica las mismas formas miles de veces.
    // Medir una sola vez mezclaba las dos cosas en un número que no era ninguna.
    let mut c = Vec::new();
    let (gpu_frio, gpu_regimen) = timed(iters, || {
        c = gpu.matmul(&a, &b, m, k, n)?;
        Ok(())
    })?;

    let mut cref = vec![0.0f32; m * n];
    let (_, cpu_regimen) = timed(iters, || {
        crate::math::matmul(&a, &b, m, k, n, &mut cref);
        Ok(())
    })?;

    let max_err = c.iter().zip(&cref).fold(0.0f32, |w, (x, y)| w.max((x - y).abs()));
    println!(
        "matmul {m}x{k}x{n} ({iters} corridas)\n  \
         GPU  {:7.3} ms  (primera {:7.3} ms, con la creación de buffers)\n  \
         CPU  {:7.3} ms\n  \
         max_err {max_err:.2e}",
        ms(gpu_regimen),
        ms(gpu_frio),
        ms(cpu_regimen),
    );
    if max_err > 1e-2 {
        return Err(format!("el matmul de GPU difiere del CPU: max_err {max_err:.2e}"));
    }
    let (rapido, lento, veces) = if gpu_regimen < cpu_regimen {
        ("GPU", "CPU", ms(cpu_regimen) / ms(gpu_regimen))
    } else {
        ("CPU", "GPU", ms(gpu_regimen) / ms(cpu_regimen))
    };
    println!("OK: resultados correctos. En régimen gana {rapido} por {veces:.2}x sobre {lento}.");
    Ok(())
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Corre `f` `iters` veces y devuelve (la primera, el promedio del resto).
///
/// Separarlas es el punto: la primera incluye todo lo que se paga una sola vez
/// y promediarla adentro esconde justamente lo que se quiere ver.
fn timed(
    iters: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<(std::time::Duration, std::time::Duration), String> {
    let t0 = std::time::Instant::now();
    f()?;
    let primera = t0.elapsed();
    if iters <= 1 {
        return Ok((primera, primera));
    }
    let t0 = std::time::Instant::now();
    for _ in 1..iters {
        f()?;
    }
    Ok((primera, t0.elapsed() / (iters - 1) as u32))
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
         \x20 eva bet --weights <archivo> --data <archivo> [--val F] [--bins N]\n\
         \x20 eva stake --weights <archivo> --data <archivo> [--span N] [--epochs N] [--lr F]\n\
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
