use std::time::{Duration, Instant};

use crate::data::TextDataset;
use crate::model::{EvaConfig, EvaModel};
use crate::nn::Module;
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
    /// Crédito local: cada bloque con su objetivo, sin gradiente que cruce.
    pub local: bool,
    /// El estado de ClockMem no se reinicia entre ventanas.
    pub persist: bool,
    /// Recorrer el corpus en orden sin persistir estado. Es el CONTROL de
    /// `--persist`: sin esto se compararían dos cambios a la vez, el orden y
    /// la memoria, y no se sabría cuál produjo la diferencia.
    pub inorder: bool,
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

    // Andamio de entrenamiento: una cabeza por bloque intermedio. No se
    // guarda en el checkpoint, así la inferencia queda idéntica.
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
        println!("eva: CRÉDITO LOCAL, ningún gradiente cruza de bloque a bloque");
        println!("eva: {} params extra en cabezas auxiliares (andamio, no van al modelo)", h.count());
    }

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

    // CON ESTADO PERSISTENTE HAY QUE IR EN ORDEN. Barajar llenaría el estado
    // con contexto de un documento que no tiene nada que ver con el siguiente:
    // eso no es memoria, es ruido. Y como el orden también afecta al
    // entrenamiento, la línea base para comparar tiene que correr en orden
    // igual -- si no, se estarían comparando dos cosas a la vez.
    let mut estados = model.fresh_states();

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
                paso_local(&mut model, h, &mut opt, &input, &target)
            } else {
                let logits = if tcfg.persist {
                    model.forward_carrying(&input, &mut estados)
                } else {
                    model.forward(&input)
                };
                let loss = crate::tensor::ops::cross_entropy(&logits, &target);
                let loss_v = loss.data[0];
                if gate.should_learn(loss_v) {
                    bwd += 1;
                    let grads = crate::prof::time(crate::prof::P::Backward, || backward(&loss));
                    let mut params = model.parameters_mut();
                    crate::prof::time(crate::prof::P::Optim, || opt.step(&mut params, &grads));
                }
                loss_v
            };

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
    println!("eva: pico de memoria del proceso {:.1} MB", crate::local::peak_rss_mb());
    crate::tensor::held::informe(total_params);
    if n_val > 0 {
        // Con estado limpio: comparable con cualquier corrida.
        let vl = eval_loss(&model, &ds, n_train, n_windows);
        if tcfg.persist {
            // Y con el estado que viene del texto anterior, que es la ventaja
            // que se está probando. Van SEPARADOS: mezclarlos sería cantar
            // victoria por una diferencia que no es la que se cree.
            let vp = eval_loss_carrying(&model, &ds, n_train, n_windows);
            println!("eva: validación con estado heredado {:.4} | {:.3} bits/byte",
                vp, bits_per_byte(vp));
        }
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

/// Un paso con crédito local. Devuelve la pérdida del ÚLTIMO bloque, que es la
/// salida real del modelo y por lo tanto lo comparable con la línea base.
///
/// La clave está en el orden: cada bloque retropropaga, actualiza y **libera**
/// antes de que empiece el siguiente. Sumar las pérdidas y hacer un backward
/// al final daría los mismos gradientes y **ningún ahorro de memoria**, que es
/// justamente lo único que se está tratando de comprar acá.
fn paso_local(
    model: &mut EvaModel,
    heads: &mut crate::local::Heads,
    opt: &mut AdamW,
    input: &[usize],
    target: &[usize],
) -> f32 {
    use crate::tensor::ops as ops;
    let n = model.blocks.len();
    let mut x = model.embed.embed(input);
    let mut ultima = 0.0;

    for i in 0..n {
        let y = model.blocks[i].forward(&x);
        let final_ = i + 1 == n;
        let logits = if final_ {
            ops::matmul(&model.norm_out.forward(&y), &model.head_w)
        } else {
            heads.logits(i, &y)
        };
        let loss = ops::cross_entropy(&logits, target);
        if final_ {
            ultima = loss.data[0];
        }

        let grads = crate::prof::time(crate::prof::P::Backward, || backward(&loss));

        let mut params: Vec<&mut crate::tensor::Tensor> = Vec::new();
        if i == 0 {
            params.push(&mut model.embed.table);
        }
        params.extend(model.blocks[i].parameters_mut());
        if final_ {
            params.extend(model.norm_out.parameters_mut());
            params.push(&mut model.head_w);
        } else {
            params.extend(heads.params_mut(i));
        }
        crate::prof::time(crate::prof::P::Optim, || opt.step(&mut params, &grads));

        // `detach` es lo que corta el crédito: el bloque siguiente arranca de
        // un tensor sin grafo detrás, así el de este bloque se libera acá.
        x = y.detach();
    }
    ultima
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

/// Como `eval_loss`, pero dejando correr el estado de una ventana a la otra:
/// el examen se toma con la memoria que el texto anterior dejó.
fn eval_loss_carrying(model: &EvaModel, ds: &TextDataset, from: usize, to: usize) -> f32 {
    let mut estados = model.fresh_states();
    let mut sum = 0.0;
    let mut pico = 0.0f32;
    for wi in from..to {
        let (input, target) = ds.window(wi);
        let logits = model.forward_carrying(&input, &mut estados);
        sum += crate::tensor::ops::cross_entropy(&logits, &target).data[0];
        for e in estados.iter() {
            for v in e {
                pico = pico.max(v.abs());
            }
        }
    }
    // Si esto es enorme, los canales lentos (alpha cerca de 1) están
    // acumulando sin olvidar y el estado saturó: el problema sería la
    // inicialización del reloj, no la idea de persistir.
    println!("eva: magnitud máxima del estado heredado: {pico:.2}");
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
