//! Ronda 3, receta de Dante: λ adaptativo por posición (usando la señal
//! GRATIS de p[argmax], el resultado del Paso 1) contra el λ global fijo,
//! estilo kNN-LM -- las dos en la MISMA forma log-lineal, para aislar una
//! sola variable (adaptativo vs fijo) y no mezclarla con "log-lineal vs
//! espacio de probabilidad".
//!
//! FÓRMULA USADA (interpretación explícita, para que se pueda corregir si
//! no es la intención exacta -- el mensaje de Dante reusa "β" en dos
//! fórmulas y una lectura literal lo eleva al cuadrado en el caso
//! adaptativo, que casi seguro no es la intención):
//!
//!   bias_i_k = max(ln(q_i_k + eps), -7.7)      -- cap de Séneca, sin β adentro
//!   FIJO:      z_i = z_m,i + β · bias_i
//!   ADAPTATIVO: z_i = z_m,i + β · gate_i · bias_i
//!   gate_i = 0                       si p[argmax]_i >= 0.8 (piso)
//!          = clamp(1 - p[argmax]_i, 0, 1)   si no
//!
//! Un solo β, barrido sobre DESARROLLO (la mitad de ventanas ya usada para
//! entrenar la sonda en `sonda_tabla.rs`), elegido por separado para fijo y
//! para adaptativo -- nunca mirando la mitad de VALIDACIÓN, que se toca una
//! sola vez para el número final.
//!
//! NO toca `src/`: sólo `forward_hidden`, `Recall::lookup_detail`,
//! `bet::classify_row`, todo público. Cero op nueva, tal como pidió Dante.

use eva_llm_v0::bet::classify_row;
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::model::EvaModel;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::save::load_model;

const CAP: f32 = 7.7;
const PISO_CONFIANZA: f32 = 0.8;
const EPS: f32 = 1e-9;

/// Todo lo que hace falta por posición para recombinar con cualquier β sin
/// volver a correr el modelo.
struct Pos {
    target: usize,
    logits_m: Vec<f32>,
    /// bias_i_k ya con el cap aplicado, o None si la tabla no contestó acá
    /// (posición sin hit: se combina igual que el modelo solo).
    bias: Option<Vec<f32>>,
    p_argmax: f32,
}

fn recolectar(model: &EvaModel, ds: &TextDataset, tabla: &Recall, ventanas: &[usize], seq: usize) -> Vec<Pos> {
    let vocab = model.cfg.vocab;
    let mut out = Vec::new();
    for &wi in ventanas {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_hidden(&input);
        for t in 0..input.len() {
            let row = logits.data[t * vocab..(t + 1) * vocab].to_vec();
            let (p_argmax, _, _, _) = classify_row(&row, target[t]);

            let pos_global = wi * seq + t + 1;
            let desde = pos_global.saturating_sub(8);
            let ctx = &ds.ids[desde..pos_global];
            let bias = if ctx.is_empty() {
                None
            } else {
                tabla.lookup_detail(ctx, vocab).map(|(q, _count)| {
                    q.iter().map(|&qi| (qi + EPS).ln().max(-CAP)).collect::<Vec<f32>>()
                })
            };
            out.push(Pos { target: target[t], logits_m: row, bias, p_argmax });
        }
    }
    out
}

/// bits/byte de la combinación, sobre un conjunto de posiciones, para un β
/// y un modo (adaptativo o fijo). Devuelve (bpb_total, bpb_delegacion,
/// bpb_segura, n_delegacion, n_segura) para el desglose que pidió Dante.
fn bpb(posiciones: &[Pos], beta: f32, adaptativo: bool) -> (f64, f64, f64, usize, usize) {
    let mut nats_total = 0.0f64;
    let (mut nats_deleg, mut nats_segura) = (0.0f64, 0.0f64);
    let (mut n_deleg, mut n_segura) = (0usize, 0usize);

    for p in posiciones {
        let lambda = match &p.bias {
            None => 0.0,
            Some(_) if !adaptativo => beta,
            Some(_) => beta * if p.p_argmax >= PISO_CONFIANZA { 0.0 } else { (1.0 - p.p_argmax).clamp(0.0, 1.0) },
        };
        let mx_m = p.logits_m.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // z combinado, restando el máximo del modelo solo para estabilidad
        // (el bias es acotado por el cap, no puede desestabilizar más).
        let mut sum = 0.0f32;
        let mut z_tgt = 0.0f32;
        for (k, &zm) in p.logits_m.iter().enumerate() {
            let zc = zm - mx_m + lambda * p.bias.as_ref().map(|b| b[k]).unwrap_or(0.0);
            let e = zc.exp();
            sum += e;
            if k == p.target {
                z_tgt = zc;
            }
        }
        let nats = -(z_tgt - sum.ln());
        nats_total += nats as f64;
        if p.p_argmax < PISO_CONFIANZA {
            nats_deleg += nats as f64;
            n_deleg += 1;
        } else {
            nats_segura += nats as f64;
            n_segura += 1;
        }
    }
    let ln2 = std::f64::consts::LN_2;
    let n = posiciones.len().max(1) as f64;
    (
        nats_total / n / ln2,
        nats_deleg / n_deleg.max(1) as f64 / ln2,
        nats_segura / n_segura.max(1) as f64 / ln2,
        n_deleg,
        n_segura,
    )
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("no pude cargar el checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();
    let tabla = Recall::build(&bytes_train);

    // Mismo split que sonda_tabla.rs: mitad de val para elegir β (desarrollo),
    // la otra mitad se toca una sola vez para el número final.
    let mitad = n_train + (n_val / 2);
    let ventanas_dev: Vec<usize> = (n_train..mitad).collect();
    let ventanas_val: Vec<usize> = (mitad..n).collect();
    println!("eva lambda_adaptativo: {} ventanas desarrollo | {} ventanas validación final",
        ventanas_dev.len(), ventanas_val.len());

    let pos_dev = recolectar(&model, &ds, &tabla, &ventanas_dev, seq);
    let pos_val = recolectar(&model, &ds, &tabla, &ventanas_val, seq);
    println!("eva lambda_adaptativo: {} posiciones dev | {} posiciones val\n",
        pos_dev.len(), pos_val.len());

    let (base_dev, ..) = bpb(&pos_dev, 0.0, false);
    println!("baseline (modelo solo, β=0), dev:  {base_dev:.4} bits/byte");

    let betas = [0.0f32, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];
    println!("\n=== barrido de β sobre DESARROLLO ===");
    println!("   β     fijo (bpb)   adaptativo (bpb)");
    let (mut mejor_fijo, mut mejor_fijo_bpb) = (0.0f32, f64::INFINITY);
    let (mut mejor_adap, mut mejor_adap_bpb) = (0.0f32, f64::INFINITY);
    for &b in &betas {
        let (f, ..) = bpb(&pos_dev, b, false);
        let (a, ..) = bpb(&pos_dev, b, true);
        println!("  {b:>4.1}    {f:.4}        {a:.4}");
        if f < mejor_fijo_bpb { mejor_fijo_bpb = f; mejor_fijo = b; }
        if a < mejor_adap_bpb { mejor_adap_bpb = a; mejor_adap = b; }
    }
    println!("\n  mejor β fijo:       {mejor_fijo:.1}  ({mejor_fijo_bpb:.4} bpb en dev)");
    println!("  mejor β adaptativo: {mejor_adap:.1}  ({mejor_adap_bpb:.4} bpb en dev)");

    // EL NÚMERO: validación, tocada una sola vez, con los β ya elegidos.
    let (base_val, ..) = bpb(&pos_val, 0.0, false);
    let (fijo_val, ..) = bpb(&pos_val, mejor_fijo, false);
    let (adap_val, deleg_val, segura_val, n_deleg, n_segura) = bpb(&pos_val, mejor_adap, true);

    println!("\n=== EL NÚMERO: bits/byte en VALIDACIÓN (tocada una sola vez) ===");
    println!("  modelo solo (β=0):                {base_val:.4}");
    println!("  λ FIJO, estilo kNN-LM (β={mejor_fijo:.1}):    {fijo_val:.4}");
    println!("  λ ADAPTATIVO vía p[argmax] (β={mejor_adap:.1}): {adap_val:.4}");

    // Mismo desglose por zona pero con el MODELO SOLO, para comparar
    // manzana con manzana (no el número aislado, el delta contra el
    // baseline en la MISMA zona) -- es lo que pidió Dante explícito.
    let (_, base_deleg, base_segura, ..) = bpb(&pos_val, 0.0, false);

    println!("\n=== bonus: dónde viene la ganancia (adaptativo vs modelo solo, por zona) ===");
    println!("  zona            n       modelo solo   adaptativo   Δ");
    println!("  delegación   {n_deleg:>6}   {base_deleg:>10.4}   {deleg_val:>10.4}   {:+.4}",
        deleg_val - base_deleg);
    println!("  segura       {n_segura:>6}   {base_segura:>10.4}   {segura_val:>10.4}   {:+.4}",
        segura_val - base_segura);
    println!("  (la zona segura tiene que dar ~0.0000 de diferencia -- el piso fuerza λ=0");
    println!("  ahí, así que es control de que la implementación no toca lo que no debe.)");

    println!("\n=== VEREDICTO ===");
    if adap_val < fijo_val - 0.001 {
        println!("  El adaptativo le gana al fijo ({adap_val:.4} < {fijo_val:.4}).");
    } else if fijo_val < adap_val - 0.001 {
        println!("  El fijo le gana al adaptativo ({fijo_val:.4} < {adap_val:.4}) -- la");
        println!("  adaptación por p[argmax] no aportó sobre el λ global simple.");
    } else {
        println!("  Empatan dentro del ruido -- el gate no cambia nada medible acá.");
    }
    if fijo_val < base_val - 0.001 || adap_val < base_val - 0.001 {
        println!("  Y ambos, o al menos uno, le ganan al modelo solo -- la combinación");
        println!("  log-lineal aporta sobre no usar la tabla en absoluto.");
    } else {
        println!("  Ninguno le gana claro al modelo solo -- la combinación log-lineal no");
        println!("  aporta a esta escala, publicada o no.");
    }
}
