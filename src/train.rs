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

// Ronda 4, segunda intervención de GPT sobre el achatamiento de `alpha`:
// traza periódica del gradiente de `log_clock` DURANTE el entrenamiento,
// no sólo en el checkpoint final -- "quiero ver dónde empieza a
// separarse la trayectoria". Gratis: son los MISMOS gradientes que ya
// se calculan para el paso del optimizador, ninguna pasada extra.
// Detrás de `EVA_ALPHA_TRACE=<n>` (cada n pasos), apagado por defecto.
struct Traza {
    cada: usize,
    suma_abs: Vec<Vec<f64>>,
    n: usize,
}

impl Traza {
    fn desde_env(n_blocks: usize, dim: usize) -> Option<Self> {
        let cada = std::env::var("EVA_ALPHA_TRACE").ok()?.parse::<usize>().ok()?;
        if cada == 0 { return None; }
        Some(Traza { cada, suma_abs: vec![vec![0.0; dim]; n_blocks], n: 0 })
    }

    fn acumular(&mut self, model: &EvaModel, grads: &std::collections::HashMap<usize, Vec<f32>>) {
        for (bi, block) in model.blocks.iter().enumerate() {
            if let Mixer::Clock(clock) = &block.mixer {
                if let Some(g) = grads.get(&clock.log_clock.id) {
                    for (c, &gv) in g.iter().enumerate() {
                        self.suma_abs[bi][c] += gv.abs() as f64;
                    }
                }
            }
        }
        self.n += 1;
    }

    fn tal_vez_imprimir(&mut self, model: &EvaModel, step: usize) {
        if step % self.cada != 0 { return; }
        for (bi, block) in model.blocks.iter().enumerate() {
            let Mixer::Clock(clock) = &block.mixer else { continue };
            let dim = clock.log_clock.data.len();
            let (mut rapido, mut medio, mut lento) = (0usize, 0usize, 0usize);
            let mut suma_alpha = 0.0f64;
            let antisat = std::env::var("EVA_ALPHA_ANTISAT").is_ok();
            let temp = std::env::var("EVA_ALPHA_TEMP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
            for c in 0..dim {
                let lc = clock.log_clock.data[c];
                let a = if antisat {
                    0.5 * (1.0 + lc / (1.0 + lc * lc).sqrt())
                } else {
                    1.0 / (1.0 + (-lc / temp).exp())
                };
                suma_alpha += a as f64;
                if a < 0.05 { rapido += 1 } else if a < 0.3 { medio += 1 } else { lento += 1 }
            }
            let ga_medio: f64 = self.suma_abs[bi].iter().sum::<f64>() / dim as f64 / self.n.max(1) as f64;
            println!("  [traza step {step}] bloque {bi}: rápido={rapido} medio={medio} lento={lento} | alpha_media={:.4} | |grad|_medio={:.6} (n={})",
                suma_alpha / dim as f64, ga_medio, self.n);
        }
        for v in self.suma_abs.iter_mut() { v.iter_mut().for_each(|x| *x = 0.0); }
        self.n = 0;
    }
}

// Ronda 3, benchmark final: el gate como regularizador de entrenamiento,
// receta consolidada por Dante -- "enseñar donde la tabla es determinística,
// no tocar donde nadie gana". A diferencia de `mixed_ce` (que mezclaba la
// tabla en el TARGET y murió medido, monótono negativo), esto suma un sesgo
// CONSTANTE (sin gradiente propio, `Tensor::new` no pide grad) a los logits
// ANTES del softmax -- kNN-LM-en-eval trasladado a entrenamiento, con el
// gradiente real fluyendo de vuelta al modelo a través de `add` (backward de
// suma = identidad en ambos operandos, pero sólo `logits` tiene grafo).
//
// Gate por DETERMINISMO (H de Shannon de la tabla), no por confianza del
// modelo (`p[argmax]`) como en el gate de eval -- a diferencia de eval, acá
// hace falta una señal disponible desde el primer paso, cuando el modelo
// todavía no dice nada útil. H_HIGH=2.0 es una lectura de las bandas ya
// medidas hoy (c=2: 0.43 bits, 3-9: 0.87, 10-49: 1.47, 50+: 2.16) -- la
// banda 50+ (irresoluble, nadie le gana) cae cerca de gate≈0, la banda c=2
// (casi determinística) cerca de gate≈1. Interpretación explícita para que
// se corrija si no es la intención: no barrida contra val, elegida por
// lectura directa de un número ya publicado, no por ajuste.
const GATE_CAP: f32 = 7.7; // mismo cap de Séneca que en lambda_adaptativo.rs
const GATE_H_HIGH: f32 = 2.0;
const GATE_EPS: f32 = 1e-9;

fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

/// Sesgo [seq, vocab] a sumar a los logits crudos antes de la pérdida: cero
/// donde la tabla no contestó, `β·gate·bias` donde sí. Construido con
/// `Tensor::new` (requires_grad=false por defecto) -- constante para el
/// grafo, el gradiente de la pérdida vuelve intacto a los logits del modelo.
fn injectar_bias(tabla: &Recall, ds: &TextDataset, wi: usize, seq: usize, vocab: usize, beta: f32) -> Tensor {
    let mut data = vec![0.0f32; seq * vocab];
    for t in 0..seq {
        let pos_global = wi * seq + t + 1;
        let desde = pos_global.saturating_sub(8);
        let ctx = &ds.ids[desde..pos_global];
        if ctx.is_empty() {
            continue;
        }
        let Some((q, _count)) = tabla.lookup_detail(ctx, vocab) else { continue };
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
    /// β del regularizador de tabla (0.0 = apagado, entrenamiento normal,
    /// idéntico a antes de esta receta). Gate por determinismo de la tabla,
    /// no por confianza del modelo -- ver comentario junto a `injectar_bias`.
    pub gate_beta: f32,
    /// Ronda 4, hipótesis de GPT ("supervivencia"): probabilidad de que CADA
    /// bloque se saltee, independiente, en cada paso (0.0 = apagado,
    /// idéntico a antes). Reusa `forward_skips` (ya público, el mismo que
    /// usa `techo`) -- es Stochastic Depth (Huang et al. 2016) a nivel de
    /// bloque, sin sesgo hacia ningún bloque en particular a propósito: la
    /// pregunta es si la red redistribuye dependencia sola, no si la
    /// forzamos a hacerlo.
    pub destroy_p: f32,
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

    // Tabla congelada, construida SOLO sobre train -- misma disciplina que
    // los scripts de eval de hoy. None si el regularizador está apagado
    // (β=0.0): cero costo extra, entrenamiento idéntico al de siempre.
    let tabla: Option<Recall> = if tcfg.gate_beta > 0.0 {
        let bytes_train: Vec<usize> = ds.ids[..n_train * mcfg.seq_len].to_vec();
        let t = Recall::build(&bytes_train);
        println!("eva: regularizador de tabla ACTIVO, β={:.2}, H_HIGH={GATE_H_HIGH:.1} (gate por determinismo, no confianza)", tcfg.gate_beta);
        Some(t)
    } else {
        None
    };

    let mut traza = Traza::desde_env(model.blocks.len(), mcfg.dim);
    if traza.is_some() {
        println!("eva: TRAZA de alpha/gradiente ACTIVA (EVA_ALPHA_TRACE)");
    }

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
                // EL GRAFO SE LIBERA ANTES DEL OPTIMIZADOR, y no es cosmético.
                // Los pesos se comparten con el grafo por `Arc`; si el grafo
                // sigue vivo, `make_mut` del optimizador copia CADA parámetro
                // en CADA paso. No daría error: sólo andaría lento. Por eso el
                // ámbito, y por eso el contador de copias de más abajo.
                let (loss_v, grads) = {
                    let logits = if tcfg.persist {
                        model.forward_carrying(&input, &mut estados)
                    } else if tcfg.destroy_p > 0.0 {
                        let skips: Vec<usize> = (0..model.blocks.len())
                            .filter(|_| rng.next_f32() < tcfg.destroy_p)
                            .collect();
                        model.forward_skips(&input, &skips).0
                    } else {
                        model.forward(&input)
                    };
                    // El sesgo es constante (sin grafo): `add` sólo necesita
                    // grad de `logits`, que sí lo tiene. Nada nuevo que
                    // gradcheckear -- son dos ops existentes compuestas.
                    let logits = match &tabla {
                        Some(t) => crate::tensor::ops::add(
                            &logits,
                            &injectar_bias(t, &ds, wi, mcfg.seq_len, mcfg.vocab, tcfg.gate_beta),
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
                    if let Some(tr) = traza.as_mut() {
                        tr.acumular(&model, &grads);
                    }
                    let mut params = model.parameters_mut();
                    crate::prof::time(crate::prof::P::Optim, || opt.step(&mut params, &grads));
                }
                loss_v
            };

            running += loss_v;
            step += 1;

            if let Some(tr) = traza.as_mut() {
                tr.tal_vez_imprimir(&model, step);
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
    let copias = crate::optim::copias_de_parametros();
    println!("  copias de parámetros por Arc compartido: {copias}  (tiene que ser 0)");
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
