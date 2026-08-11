use std::time::{Duration, Instant};

use crate::data::TextDataset;
use crate::model::{EvaConfig, EvaModel};
use crate::optim::AdamW;
use crate::save::{load_model, save_model};
use crate::tensor::autograd::backward;
use crate::tokenizer::ByteTokenizer;

pub struct TrainConfig {
    pub data_path: String,
    pub epochs: usize,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    pub log_every: usize,
    pub out_path: String,
    pub resume: Option<String>,
    /// Qué fracción de las ventanas se reserva para validar. 0 la apaga.
    pub val_frac: f32,
    /// Umbral de sorpresa. 0 = aprender de todo (la convención de hoy).
    pub surprise: f32,
}

pub fn train(tcfg: &TrainConfig, mcfg: &EvaConfig) -> Result<(), String> {
    let ds = TextDataset::from_file(&tcfg.data_path, mcfg.seq_len)
        .map_err(|e| format!("no se pudo leer {}: {}", tcfg.data_path, e))?;
    let n_windows = ds.num_windows();
    if n_windows == 0 {
        return Err(format!("el dataset es muy chico para seq_len {}", mcfg.seq_len));
    }

    let mut model = match &tcfg.resume {
        Some(p) => load_model(p).map_err(|e| format!("no se pudo cargar {}: {}", p, e))?,
        None => EvaModel::new(mcfg.clone()),
    };
    let mut rng = crate::rng::Rng::new(tcfg.seed);
    let mut opt = AdamW::new(tcfg.lr, tcfg.wd);
    // El calentamiento es una época corta: antes de eso la media móvil se
    // calcularía con un modelo aleatorio y no significaría nada.
    let mut gate = crate::learn::gate(tcfg.surprise, 200);
    // Se cuentan por separado porque el backward cuesta el doble que el
    // forward: comparar corridas por "pasos" escondería justo lo que se quiere
    // medir.
    let (mut fwd, mut bwd) = (0usize, 0usize);

    let total_params = model.param_count();
    println!("eva: dataset {} bytes, {} windows, {} params",
        ds.ids.len(), n_windows, total_params);

    // EL CORTE VA AL FINAL Y CONTIGUO, no salteado. Las ventanas vecinas
    // comparten contexto: con un corte aleatorio, el modelo ve en entrenamiento
    // el texto pegado a lo que después se le toma como examen, y la validación
    // da mejor de lo que corresponde. Un examen que filtra no mide nada.
    let n_val = if tcfg.val_frac > 0.0 {
        (((n_windows as f32) * tcfg.val_frac).round() as usize).clamp(1, n_windows / 2)
    } else {
        0
    };
    let n_train = n_windows - n_val;
    if n_train == 0 {
        return Err("no quedan ventanas de entrenamiento después del corte".into());
    }
    println!("eva: {n_train} ventanas para entrenar, {n_val} para validar");

    let total_steps = n_train * tcfg.epochs;
    let mut step = 0usize;
    let mut running = 0.0f32;
    let t0 = Instant::now();
    let mut last_log = Instant::now();

    for epoch in 0..tcfg.epochs {
        let order = ds.shuffled_train_indices(n_train, &mut rng);
        for &wi in &order {
            let (input, target) = ds.window(wi);

            let logits = model.forward(&input);
            let loss = crate::tensor::ops::cross_entropy(&logits, &target);
            let loss_v = loss.data[0];

            fwd += 1;
            if gate.should_learn(loss_v) {
                bwd += 1;
                let grads = crate::prof::time(crate::prof::P::Backward, || backward(&loss));
                let mut params = model.parameters_mut();
                crate::prof::time(crate::prof::P::Optim, || opt.step(&mut params, &grads));
            }

            running += loss_v;
            step += 1;

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
                println!("  validación: loss {:.4} | {:.3} bits/byte", vl, bits_per_byte(vl));
            }

            if step % (tcfg.log_every * 10) == 0 {
                let sample = generate_sample(&model, mcfg.seq_len, 64, &mut rng);
                println!("sample: {:?}", ByteTokenizer::decode(&sample));
                save_model(&tcfg.out_path, &model).map_err(|e| format!("no se pudo guardar: {}", e))?;
            }
        }
    }

    save_model(&tcfg.out_path, &model).map_err(|e| format!("no se pudo guardar: {}", e))?;
    if n_val > 0 {
        let vl = eval_loss(&model, &ds, n_train, n_windows);
        // EL NÚMERO CON EL QUE SE COMPARAN ARQUITECTURAS. La pérdida de
        // entrenamiento sólo dice cuánto memorizó.
        println!("eva: VALIDACIÓN FINAL loss {:.4} | {:.3} bits/byte ({} ventanas, arch {})",
            vl, bits_per_byte(vl), n_windows - n_train, mcfg.arch.name());
        println!("eva: regla '{}' | {} forward, {} backward ({:.0}% aprendidos)",
            gate.name(), fwd, bwd, 100.0 * bwd as f32 / fwd.max(1) as f32);
    }
    let elapsed: Duration = t0.elapsed();
    crate::prof::report(elapsed);
    println!("eva: entrenamiento terminado en {:.1}s, pesos en {}", elapsed.as_secs_f32(), tcfg.out_path);
    Ok(())
}

/// Pérdida media sobre ventanas que el modelo nunca vio. Sin backward.
fn eval_loss(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) -> f32 {
    let mut sum = 0.0;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        sum += crate::tensor::ops::cross_entropy(&logits, &target).data[0];
    }
    sum / (to - from) as f32
}

/// Bits por byte: la unidad honesta para un modelo byte-level, y comparable
/// entre corpus y entre arquitecturas. La pérdida en nats no le dice nada a
/// nadie.
fn bits_per_byte(loss: f32) -> f32 {
    loss / std::f32::consts::LN_2
}

pub fn generate_sample(model: &EvaModel, seq: usize, max: usize, rng: &mut crate::rng::Rng) -> Vec<usize> {
    let mut ids = crate::gen::generate(model, &[0], max, 0.9, 32, rng, seq);
    ids.retain(|&x| x != 0);
    ids
}
