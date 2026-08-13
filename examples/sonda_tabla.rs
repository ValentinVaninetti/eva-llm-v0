//! Ronda 3, Paso 1: la sonda detached, contra la vara REAL (~53%, no ~97%).
//!
//! Misma pregunta que mató al stake antes-de-gastar, aplicada a algo
//! distinto: no es "¿sé si voy a acertar?", es "¿sé si la tabla tiene
//! razón acá?". Cabecita `g = sigmoid(w·h+b)` (misma forma que `StakeHead`,
//! reusada tal cual -- cero op nueva), DETACHED del modelo: no le manda
//! gradiente, así que si el modelo ya entrenado no tiene la señal, esto lo
//! prueba sin gastar un solo paso de entrenamiento real.
//!
//! NO toca `src/`: todo lo que usa ya es público (`forward_hidden`,
//! `Recall::lookup_detail`, `ops::stake_loss` con span_len=1, `StakeHead`).

use eva_llm_v0::bet::classify_row;
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::rng::Rng;
use eva_llm_v0::save::load_model;
use eva_llm_v0::stake::StakeHead;
use eva_llm_v0::tensor::autograd::backward;
use eva_llm_v0::tensor::ops as ops;
use eva_llm_v0::tensor::Tensor;

/// Una fila de hidden ya detached + su label + el count que la generó.
struct Ejemplo {
    hidden: Vec<f32>,
    correcto: f32,
    count: u32,
    /// p[argmax] que el modelo YA declara gratis (`bet::classify_row`) --
    /// la vara a batir. Si esto sólo redescubre esto, no agrega nada nuevo.
    conf_modelo: f32,
}

fn recolectar(
    model: &eva_llm_v0::model::EvaModel,
    ds: &TextDataset,
    tabla: &Recall,
    ventanas: &[usize],
    seq: usize,
    dim: usize,
) -> Vec<Ejemplo> {
    let mut out = Vec::new();
    for &wi in ventanas {
        let (input, target) = ds.window(wi);
        let (logits, hidden) = model.forward_hidden(&input);
        let vocab = model.cfg.vocab;
        for t in 0..input.len() {
            // Mismo indexado que el Paso 0, ya corregido una vez ahí.
            let pos_global = wi * seq + t + 1;
            let desde = pos_global.saturating_sub(8);
            let ctx = &ds.ids[desde..pos_global];
            if ctx.is_empty() {
                continue;
            }
            let Some((q, count)) = tabla.lookup_detail(ctx, 256) else { continue };
            // Sólo posiciones con hit: es la misma población que midió el
            // Paso 0 (la vara), para que la comparación sea manzanas con
            // manzanas. Un miss no plantea la pregunta "¿la tabla tiene razón?".
            let argmax = q.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
                .0;
            let correcto = (argmax == target[t]) as u8 as f32;
            let (conf_modelo, _, _, _) = classify_row(&logits.data[t * vocab..(t + 1) * vocab], target[t]);
            out.push(Ejemplo {
                hidden: hidden.data[t * dim..(t + 1) * dim].to_vec(),
                correcto,
                count,
                conf_modelo,
            });
        }
    }
    out
}

fn banda(count: u32) -> &'static str {
    match count {
        2..=4 => "2-4",
        5..=9 => "5-9",
        10..=49 => "10-49",
        _ => "50+",
    }
}

fn auroc(pares: &[(f64, bool)]) -> f64 {
    let mut v: Vec<(f64, bool)> = pares.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n_pos = v.iter().filter(|(_, p)| *p).count() as f64;
    let n_neg = v.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let mut suma_rangos_pos = 0.0f64;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 { j += 1; }
        let rango_medio = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            if v[k].1 { suma_rangos_pos += rango_medio; }
        }
        i = j;
    }
    (suma_rangos_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("no pude cargar el checkpoint");
    let dim = model.cfg.dim;
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();
    let tabla = Recall::build(&bytes_train);

    // 50/50 de las ventanas de VALIDACIÓN, no de train: la sonda entrena y
    // evalúa dentro de la misma población que midió el Paso 0, pero sin
    // testear sobre lo mismo que entrenó.
    let mitad = n_train + (n_val / 2);
    let ventanas_sonda_train: Vec<usize> = (n_train..mitad).collect();
    let ventanas_sonda_eval: Vec<usize> = (mitad..n).collect();
    println!("eva sonda_tabla: {} params modelo | {} ventanas sonda-train | {} ventanas sonda-eval",
        model.param_count(), ventanas_sonda_train.len(), ventanas_sonda_eval.len());

    let ejemplos_train = recolectar(&model, &ds, &tabla, &ventanas_sonda_train, seq, dim);
    let ejemplos_eval = recolectar(&model, &ds, &tabla, &ventanas_sonda_eval, seq, dim);
    println!("eva sonda_tabla: {} ejemplos (hit) para entrenar la sonda, {} para evaluarla",
        ejemplos_train.len(), ejemplos_eval.len());

    // Un solo tensor grande (S=n_ejemplos, D=dim), hidden ya detached --
    // stake_loss con span_len=1 da un stake por fila.
    let hidden_train = Tensor::new(
        ejemplos_train.iter().flat_map(|e| e.hidden.iter().copied()).collect(),
        vec![ejemplos_train.len(), dim],
    );
    let bien_train: Vec<f32> = ejemplos_train.iter().map(|e| e.correcto).collect();

    let mut rng = Rng::new(11);
    let mut head = StakeHead::new(dim, &mut rng);
    let mut opt = eva_llm_v0::optim::AdamW::new(1e-2, 0.0);

    println!("\n=== entrenando la cabecita (detached, cero gradiente al modelo) ===");
    for epoch in 0..60 {
        let loss = ops::stake_loss(&hidden_train, &head.w, &head.b, &bien_train, 1);
        let loss_v = loss.data[0];
        let grads = backward(&loss);
        let mut params = head.params_mut();
        opt.step(&mut params, &grads);
        if epoch % 10 == 0 || epoch == 59 {
            println!("  epoch {epoch:>2} | loss (MSE) {loss_v:.4}");
        }
    }

    // Evaluación: AUROC pooled y por banda de count, sobre ejemplos que la
    // sonda NUNCA vio entrenando (otra mitad de las ventanas de validación).
    let hidden_eval = Tensor::new(
        ejemplos_eval.iter().flat_map(|e| e.hidden.iter().copied()).collect(),
        vec![ejemplos_eval.len(), dim],
    );
    let stakes = head.stakes(&hidden_eval, 1);

    let mut pool: Vec<(f64, bool)> = Vec::new();
    let mut por_banda: std::collections::HashMap<&str, Vec<(f64, bool)>> = std::collections::HashMap::new();
    for (e, &g) in ejemplos_eval.iter().zip(&stakes) {
        let par = (g as f64, e.correcto > 0.5);
        pool.push(par);
        por_banda.entry(banda(e.count)).or_default().push(par);
    }

    let vara_global = ejemplos_eval.iter().filter(|e| e.correcto > 0.5).count() as f32
        / ejemplos_eval.len().max(1) as f32;

    println!("\n=== RESULTADO: AUROC de la sonda contra table_correct, en eval ===");
    println!("  vara (fracción de aciertos de la tabla en eval): {:.1}%", 100.0 * vara_global);
    println!("  AUROC pooled: {:.3}  (n={})", auroc(&pool), pool.len());
    println!("\n  banda      n       vara       AUROC");
    for b in ["2-4", "5-9", "10-49", "50+"] {
        if let Some(pares) = por_banda.get(b) {
            let vara_b = pares.iter().filter(|(_, c)| *c).count() as f32 / pares.len().max(1) as f32;
            println!("  {b:<8}  {:>5}   {:5.1}%    {:.3}", pares.len(), 100.0 * vara_b, auroc(pares));
        }
    }

    // Riesgo que dejé anotado en mi propia propuesta: que la cabecita sólo
    // redescubra algo ya gratis (la confianza que el modelo YA declara,
    // bet::classify_row) en vez de aprender algo nuevo sobre la tabla.
    let pool_conf: Vec<(f64, bool)> = ejemplos_eval.iter()
        .map(|e| (e.conf_modelo as f64, e.correcto > 0.5))
        .collect();
    let auroc_conf_libre = auroc(&pool_conf);

    let global = auroc(&pool);
    println!("\n=== CONTROL: ¿esto es nuevo, o es la confianza gratis del modelo? ===");
    println!("  AUROC de p[argmax] SOLO (bet::classify_row, cero parámetros nuevos): {auroc_conf_libre:.3}");
    println!("  AUROC de la cabecita entrenada:                                      {global:.3}");
    if global > auroc_conf_libre + 0.03 {
        println!("  La cabecita le gana a la confianza gratis -- aprendió algo que p[argmax]");
        println!("  no tenía. Vale la pena como mecanismo, no es sólo redescubrir lo gratis.");
    } else {
        println!("  La cabecita NO le gana a p[argmax] solo -- lo que mide ya estaba gratis en");
        println!("  la confianza que el modelo declara. No hace falta entrenar nada nuevo: usar");
        println!("  p[argmax] directo como el gate, sin gastar dim+1 parámetros extra.");
    }

    println!("\n=== VEREDICTO ===");
    if global < 0.55 {
        println!("  AUROC pooled {global:.3} -- azar contra la vara (~{:.0}%). El modelo NO sabe", 100.0*vara_global);
        println!("  cuándo la tabla tiene razón. La cabecita muere acá, sin tocar train.rs.");
    } else if global > 0.6 {
        println!("  AUROC pooled {global:.3} -- supera la vara claramente. La señal existe en el");
        println!("  estado oculto. Sigue viva: entrenamiento conjunto con β chico, control β=0.");
    } else {
        println!("  AUROC pooled {global:.3} -- zona gris entre 0.55 y 0.6. Ni muerta ni viva del");
        println!("  todo; decidir con Dante/Valentín si vale una segunda semilla antes de construir.");
    }
}
