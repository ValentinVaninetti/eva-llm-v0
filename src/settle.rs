//! Punto 9: ¿qué fracción de los parámetros deja de moverse, y cuándo?
//!
//! Antes de escribir el congelamiento hay que saber si hay algo que congelar.
//! La pregunta no es "¿el gradiente da chico alguna vez?" -- eso pasa todo el
//! tiempo y no dice nada. La pregunta es: **¿un parámetro se queda quieto
//! VARIAS ventanas seguidas de entrenamiento real, y esa cantidad crece a
//! medida que el modelo aprende?** Si la respuesta es no, no hay nada que
//! congelar y el punto 9 se cierra acá, como se cerró la versión barata del 8.
//!
//! CÓMO SE MIDE: se parte el entrenamiento en ventanas de `every` pasos. Al
//! principio y al final de cada ventana se saca una foto de TODOS los
//! parámetros (un clon del buffer, no del grafo -- esto no toca `optim.rs` ni
//! `autograd.rs`, así que no colisiona con nada de lo que está tocando Dante).
//! Por cada parámetro se calcula cuánto se movió, relativo a su propio tamaño:
//!
//! ```text
//! ratio = |fin - inicio| / (|inicio| + 1e-6)
//! ```
//!
//! Un parámetro está "quieto en esta ventana" si `ratio` no llegó al umbral.
//! Un parámetro está "asentado" si llevaba `streak` ventanas quieto SEGUIDAS
//! -- si se mueve una sola ventana, la racha se corta. Esto es a propósito:
//! es la misma pregunta que haría el mecanismo de descongelado de Valentín
//! ("si falla, se ablanda"), sólo que del lado de congelar. No es sticky.
//!
//! EL UMBRAL ES UNA ELECCIÓN, así que no se reporta uno solo: se corren tres
//! (relajado, medio, estricto) sobre la MISMA corrida, porque las fotos ya se
//! sacaron y no cuesta nada recalcular. Si los tres cuentan una fracción
//! parecida y creciente, el número es real. Si sólo el relajado ve algo,
//! el resultado es un artefacto del umbral, no un hallazgo.

use crate::data::TextDataset;
use crate::model::{EvaConfig, EvaModel};
use crate::optim::AdamW;
use crate::rng::Rng;
use crate::tensor::autograd::backward;

/// Relajado / medio / estricto. Fijos y no parametrizables: la gracia de este
/// experimento es mirar los tres a la vez, no elegir uno.
const REL_EPS: [f32; 3] = [1e-2, 1e-3, 1e-4];

pub struct SettleConfig {
    pub data_path: String,
    pub dim: usize,
    pub ffn: usize,
    pub blocks: usize,
    pub seq_len: usize,
    pub epochs: usize,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    pub val_frac: f32,
    /// Pasos por ventana de medición.
    pub every: usize,
    /// Ventanas quietas seguidas para contar "asentado".
    pub streak: u32,
}

/// Dónde vive cada tensor dentro del vector plano de snapshot.
struct Layout {
    name: String,
    offset: usize,
    len: usize,
}

/// Cuánto se movió un parámetro, relativo a su propio tamaño.
fn ratio(antes: f32, despues: f32) -> f32 {
    (despues - antes).abs() / (antes.abs() + 1e-6)
}

fn flatten(model: &EvaModel) -> (Vec<f32>, Vec<Layout>) {
    let named = model.named_parameters();
    let mut flat = Vec::with_capacity(named.iter().map(|(_, t)| t.numel()).sum());
    let mut layout = Vec::with_capacity(named.len());
    for (name, t) in named {
        let offset = flat.len();
        flat.extend_from_slice(&t.data);
        layout.push(Layout { name, offset, len: t.numel() });
    }
    (flat, layout)
}

pub fn run(cfg: &SettleConfig) -> Result<(), String> {
    let ds = TextDataset::from_file(&cfg.data_path, cfg.seq_len)
        .map_err(|e| format!("no se pudo leer {}: {}", cfg.data_path, e))?;
    let n_windows = ds.num_windows();
    if n_windows == 0 {
        return Err(format!("el dataset es muy chico para seq_len {}", cfg.seq_len));
    }
    let n_val = (((n_windows as f32) * cfg.val_frac).round() as usize).clamp(1, n_windows / 2);
    let n_train = n_windows - n_val;

    let mcfg = EvaConfig {
        vocab: 256,
        dim: cfg.dim,
        ffn_dim: cfg.ffn,
        blocks: cfg.blocks,
        conv_kernel: 5,
        eps: 1e-5,
        seq_len: cfg.seq_len,
        arch: crate::model::Arch::Clock,
    };
    let mut model = EvaModel::new(mcfg);
    let mut rng = Rng::new(cfg.seed);
    let mut opt = AdamW::new(cfg.lr, cfg.wd);
    let total_params = model.param_count();
    println!("eva settle: {total_params} params | {n_train} ventanas de entrenamiento | ventana de medición cada {} pasos", cfg.every);

    // Racha por umbral, por elemento. Empieza en 0: nadie es "viejo" antes de
    // que arranque el entrenamiento.
    let mut streaks: Vec<Vec<u32>> = REL_EPS.iter().map(|_| vec![0u32; total_params]).collect();
    let (mut start_flat, layout) = flatten(&model);

    let mut step = 0usize;
    let mut running = 0.0f32;
    let mut in_window = 0usize;
    let t0 = std::time::Instant::now();

    // Historial para ver la tendencia: (step, fracción asentada por umbral).
    let mut historia: Vec<(usize, [f32; 3])> = Vec::new();

    for epoch in 0..cfg.epochs {
        let order = ds.shuffled_train_indices(n_train, &mut rng);
        for &wi in &order {
            let (input, target) = ds.window(wi);

            let (loss_v, grads) = {
                let logits = model.forward(&input);
                let loss = crate::tensor::ops::cross_entropy(&logits, &target);
                let v = loss.data[0];
                let g = backward(&loss);
                (v, g)
            };
            {
                let mut params = model.parameters_mut();
                opt.step(&mut params, &grads);
            }

            running += loss_v;
            step += 1;
            in_window += 1;

            if in_window == cfg.every {
                in_window = 0;
                let (end_flat, _) = flatten(&model);
                let mut asentados = [0usize; 3];
                for i in 0..total_params {
                    let r = ratio(start_flat[i], end_flat[i]);
                    for (u, eps) in REL_EPS.iter().enumerate() {
                        if r < *eps {
                            streaks[u][i] += 1;
                        } else {
                            streaks[u][i] = 0;
                        }
                        if streaks[u][i] >= cfg.streak {
                            asentados[u] += 1;
                        }
                    }
                }
                let frac = [
                    asentados[0] as f32 / total_params as f32,
                    asentados[1] as f32 / total_params as f32,
                    asentados[2] as f32 / total_params as f32,
                ];
                historia.push((step, frac));
                println!(
                    "step {step:>6} | epoch {} | loss {:.4} | asentado (>={} ventanas quietas): relajado {:.1}%  medio {:.1}%  estricto {:.1}%",
                    epoch + 1,
                    running / cfg.every as f32,
                    cfg.streak,
                    100.0 * frac[0],
                    100.0 * frac[1],
                    100.0 * frac[2],
                );
                running = 0.0;
                start_flat = end_flat;
            }
        }
    }

    println!("\n=== por tensor, al final ({} umbral medio, {:.0e}) ===", REL_EPS[1], REL_EPS[1]);
    let mut filas: Vec<(String, f32)> = layout
        .iter()
        .map(|l| {
            let asentado = (l.offset..l.offset + l.len).filter(|&i| streaks[1][i] >= cfg.streak).count();
            (l.name.clone(), asentado as f32 / l.len as f32)
        })
        .collect();
    filas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, frac) in &filas {
        println!("  {name:<16} {:.1}%", 100.0 * frac);
    }

    println!("\n=== evolución de la fracción asentada, umbral medio ===");
    for (s, f) in historia.iter().step_by((historia.len() / 12).max(1)) {
        println!("  step {s:>6}: {:.1}%", 100.0 * f[1]);
    }
    if let Some((_, ultima)) = historia.last() {
        let crece = historia.len() >= 2 && ultima[1] > historia[historia.len() / 2].1[1];
        println!("\nVEREDICTO: fracción final asentada (relajado/medio/estricto) = {:.1}% / {:.1}% / {:.1}%",
            100.0 * ultima[0], 100.0 * ultima[1], 100.0 * ultima[2]);
        println!("  {}", if ultima[1] < 0.01 {
            "PRÁCTICAMENTE NADA SE ASIENTA -> el punto 9 no tiene de dónde sacar ahorro, se cierra acá"
        } else if crece {
            "CRECE con el entrenamiento -> hay señal real para congelar, vale seguir al siguiente paso"
        } else {
            "hay una fracción asentada pero NO crece claramente -> medir más épocas antes de construir"
        });
    }
    println!("\neva settle: terminado en {:.1}s", t0.elapsed().as_secs_f32());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_parameter_that_did_not_move_has_zero_ratio() {
        assert_eq!(0.0, ratio(0.5, 0.5));
    }

    #[test]
    fn a_parameter_that_doubled_has_ratio_one() {
        assert!((ratio(1.0, 2.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_row_that_never_had_gradient_does_not_blow_up() {
        // embed.table de un byte que nunca aparece: antes y después son cero.
        // Sin el +1e-6 esto sería 0.0/0.0 = NaN, y NaN < eps es siempre falso
        // -- ese parámetro jamás contaría como asentado, que es el resultado
        // contrario al que dio la corrida real. El epsilon es lo que evita eso.
        assert_eq!(0.0, ratio(0.0, 0.0));
    }

    #[test]
    fn flatten_covers_every_parameter_exactly_once() {
        let mcfg = EvaConfig {
            vocab: 256,
            dim: 8,
            ffn_dim: 16,
            blocks: 2,
            conv_kernel: 3,
            eps: 1e-5,
            seq_len: 16,
            arch: crate::model::Arch::Clock,
        };
        let model = EvaModel::new(mcfg);
        let (flat, layout) = flatten(&model);
        assert_eq!(model.param_count(), flat.len(), "el vector plano no cubre todos los parámetros");
        let cubierto: usize = layout.iter().map(|l| l.len).sum();
        assert_eq!(flat.len(), cubierto, "el layout no suma lo mismo que el plano");
        // Contiguo y sin huecos: cada tensor empieza donde terminó el anterior.
        let mut esperado = 0;
        for l in &layout {
            assert_eq!(esperado, l.offset, "hueco o superposición en {}", l.name);
            esperado += l.len;
        }
    }

    #[test]
    fn a_streak_breaks_the_instant_it_moves() {
        // La racha no es sticky: tres ventanas quietas y una que se mueve
        // tienen que dar racha 0, no 3. Es la propiedad central del diseño.
        let eps = 1e-3;
        let mut streak = 0u32;
        for &r in &[0.0, 0.0, 0.0, 0.5] {
            if r < eps { streak += 1 } else { streak = 0 }
        }
        assert_eq!(0, streak, "la racha sobrevivió a un movimiento grande");
    }
}
