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

    let total_params = model.param_count();
    let total_steps = n_windows * tcfg.epochs;
    println!("eva: dataset {} bytes, {} windows, {} params, {} steps",
        ds.ids.len(), n_windows, total_params, total_steps);

    let mut step = 0usize;
    let mut running = 0.0f32;
    let t0 = Instant::now();
    let mut last_log = Instant::now();

    for epoch in 0..tcfg.epochs {
        let order = ds.shuffled_indices(&mut rng);
        for &wi in &order {
            let (input, target) = ds.window(wi);

            let logits = model.forward(&input);
            let loss = crate::tensor::ops::cross_entropy(&logits, &target);
            let loss_v = loss.data[0];

            let grads = backward(&loss);

            let mut params = model.parameters_mut();
            opt.step(&mut params, &grads);

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

            if step % (tcfg.log_every * 10) == 0 {
                let sample = generate_sample(&model, mcfg.seq_len, 64, &mut rng);
                println!("sample: {:?}", ByteTokenizer::decode(&sample));
                save_model(&tcfg.out_path, &model).map_err(|e| format!("no se pudo guardar: {}", e))?;
            }
        }
    }

    save_model(&tcfg.out_path, &model).map_err(|e| format!("no se pudo guardar: {}", e))?;
    let elapsed: Duration = t0.elapsed();
    println!("eva: entrenamiento terminado en {:.1}s, pesos en {}", elapsed.as_secs_f32(), tcfg.out_path);
    Ok(())
}

pub fn generate_sample(model: &EvaModel, seq: usize, max: usize, rng: &mut crate::rng::Rng) -> Vec<usize> {
    let mut ids = crate::gen::generate(model, &[0], max, 0.9, 32, rng, seq);
    ids.retain(|&x| x != 0);
    ids
}
