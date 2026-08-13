//! Número gratis ofrecido a GPT/Dante: ¿los `alpha` de ClockMem (el olvido
//! por canal) muestran una jerarquía temporal real después de entrenar, o
//! convergen todos parecidos? Sólo lectura de un checkpoint ya entrenado --
//! cero código nuevo, cero entrenamiento.
//!
//! CORRECCIÓN IMPORTANTE a como se planteó la pregunta: la dispersión NO es
//! algo que "emergería" de la nada. `ClockMem::new` (src/model/clock.rs)
//! siembra `log_clock` con un espectro geométrico DISEÑADO a propósito, de
//! alpha≈0.01 (canal rápido) a alpha≈0.9999 (canal lento) -- así que hay
//! jerarquía por INICIALIZACIÓN, no por descubrimiento. La pregunta
//! honesta y falsable no es "¿aparece jerarquía?" (ya está puesta), es:
//! **¿el entrenamiento la CONSERVA (o la afila), o la ACHATA hacia un valor
//! único, tirando el diseño inicial?** Si se achata, el gradiente está
//! diciendo que la jerarquía de escalas no aportaba. Si se conserva o se
//! afila, el modelo la está usando de verdad.

use eva_llm_v0::model::block::Mixer;
use eva_llm_v0::save::load_model;

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let model = load_model(&weights).expect("no pude cargar el checkpoint");

    println!("eva alpha_dispersion: {} bloques, dim {}\n", model.blocks.len(), model.cfg.dim);

    for (bi, block) in model.blocks.iter().enumerate() {
        let Mixer::Clock(clock) = &block.mixer else {
            println!("bloque {bi}: no es ClockMem (arch distinta), salteado");
            continue;
        };
        // alpha = sigmoid(log_clock), igual que en forward().
        let alphas: Vec<f32> = clock.log_clock.data.iter().map(|&lc| 1.0 / (1.0 + (-lc).exp())).collect();
        let mut ordenado = alphas.clone();
        ordenado.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = ordenado.len();
        let media = alphas.iter().sum::<f32>() / n as f32;
        let var = alphas.iter().map(|a| (a - media).powi(2)).sum::<f32>() / n as f32;
        let desvio = var.sqrt();

        // memoria efectiva ~ 1/(1-alpha) tokens (aprox. de la serie geométrica).
        let mem_ef = |a: f32| 1.0 / (1.0 - a).max(1e-6);

        println!("=== bloque {bi} ===");
        println!("  alpha: min {:.4} (mem~{:.0} tok)  p25 {:.4}  mediana {:.4}  p75 {:.4}  max {:.4} (mem~{:.0} tok)",
            ordenado[0], mem_ef(ordenado[0]),
            ordenado[n / 4],
            ordenado[n / 2],
            ordenado[3 * n / 4],
            ordenado[n - 1], mem_ef(ordenado[n - 1]));
        println!("  media {media:.4}  desvío estándar {desvio:.4}  (init geométrico teórico: alpha_min=0.01, alpha_max=0.9999)");

        // Reparto por banda de memoria efectiva, para ver si sobrevive una
        // jerarquía de verdad (rápido/medio/lento) o si colapsa a una banda.
        let bandas = [
            ("rápido (mem<10 tok)", 0.0f32, 0.9f32),
            ("medio (10-100 tok)", 0.9, 0.99),
            ("lento (100-1000 tok)", 0.99, 0.999),
            ("muy lento (>1000 tok)", 0.999, 1.0),
        ];
        print!("  reparto por banda:");
        for (nombre, lo, hi) in bandas {
            let c = alphas.iter().filter(|&&a| a >= lo && a < hi).count();
            print!("  {nombre}={c}({:.0}%)", 100.0 * c as f32 / n as f32);
        }
        println!("\n");
    }

    println!("=== VEREDICTO ===");
    println!("  Si el desvío estándar es chico (canales convergieron parecido) y casi");
    println!("  todo cae en una sola banda: el entrenamiento ACHATÓ el espectro inicial --");
    println!("  la jerarquía de escalas no se sostuvo como útil, a esta escala/corpus.");
    println!("  Si el desvío es grande y hay masa real en varias bandas: la jerarquía");
    println!("  sobrevivió (o se afiló) -- el modelo SÍ está usando escalas temporales");
    println!("  distintas por canal, y ahí es donde seguiría la pregunta de GPT (¿esa");
    println!("  historia acumulada aporta algo que el estado \"actual\" solo no daría?).");
}
