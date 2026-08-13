//! Ronda 3, tarea de Dante: "confiado y errado" -- el lado inverso de "que
//! apueste". La zona segura (p[argmax] >= 0.8) no se toca con el gate y da
//! 0.0000 de diferencia -- pero tiene errores (0.79 bpb no es cero) y nadie
//! midió DÓNDE caen.
//!
//! Hipótesis de Dante: los errores confiados se concentran donde el
//! contexto es RARO en la tabla (count bajo) -- confianza que excede la
//! cobertura. Si la celda "confianza alta × contexto raro" tiene la MISMA
//! precisión que "confianza alta × contexto común", no hay overreach y el
//! punto cierra. Si sobresale, nace el "gate curioso".
//!
//! Una sola pasada: precisión del MODELO (no de la tabla) cruzada por
//! decil de p[argmax] × banda de count del contexto (miss / c=2 / 3-9 /
//! 10-49 / 50+). NO toca `src/`: sólo `forward_hidden`, `classify_row`,
//! `Recall::lookup_detail`, todo público, cero op nueva.

use eva_llm_v0::bet::classify_row;
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::save::load_model;

const PISO_CONFIANZA: f32 = 0.8; // mismo piso que lambda_adaptativo.rs

struct Pos {
    p_argmax: f32,
    correcto: bool,
    banda: &'static str,
}

const BANDAS: [&str; 5] = ["miss", "c=2", "3-9", "10-49", "50+"];

fn banda_count(hit: Option<(Vec<f32>, u32)>) -> &'static str {
    match hit {
        None => "miss",
        Some((_, 2)) => "c=2",
        Some((_, c)) if c <= 9 => "3-9",
        Some((_, c)) if c <= 49 => "10-49",
        Some(_) => "50+",
    }
}

fn recolectar(
    model: &eva_llm_v0::model::EvaModel,
    ds: &TextDataset,
    tabla: &Recall,
    ventanas: &[usize],
    seq: usize,
) -> Vec<Pos> {
    let vocab = model.cfg.vocab;
    let mut out = Vec::new();
    for &wi in ventanas {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_hidden(&input);
        for t in 0..input.len() {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let (p_argmax, _, correcto, _) = classify_row(row, target[t]);

            // mismo indexado que Paso 0 / sonda_tabla, ya corregido una vez.
            let pos_global = wi * seq + t + 1;
            let desde = pos_global.saturating_sub(8);
            let ctx = &ds.ids[desde..pos_global];
            let hit = if ctx.is_empty() { None } else { tabla.lookup_detail(ctx, vocab) };
            let banda = banda_count(hit);

            out.push(Pos { p_argmax, correcto, banda });
        }
    }
    out
}

/// decil 0..=9 para p en [0,1]; 1.0 cae en el decil 9.
fn decil(p: f32) -> usize {
    ((p * 10.0).floor() as usize).min(9)
}

fn crosstab(posiciones: &[Pos]) -> std::collections::HashMap<(usize, &'static str), (usize, usize)> {
    let mut m: std::collections::HashMap<(usize, &'static str), (usize, usize)> = std::collections::HashMap::new();
    for p in posiciones {
        let e = m.entry((decil(p.p_argmax), p.banda)).or_insert((0, 0));
        e.1 += 1;
        if p.correcto {
            e.0 += 1;
        }
    }
    m
}

fn imprimir_tabla(nombre: &str, posiciones: &[Pos]) {
    let m = crosstab(posiciones);
    println!("\n=== {nombre}: precisión del modelo, decil p[argmax] × banda de count ===");
    print!("  decil    ");
    for b in BANDAS {
        print!("{b:>14}");
    }
    println!();
    for d in 0..10 {
        print!("  [{:.1},{:.1})", d as f32 / 10.0, (d + 1) as f32 / 10.0);
        for b in BANDAS {
            match m.get(&(d, b)) {
                Some(&(correctos, n)) if n > 0 => {
                    print!("  {:5.1}%(n={n:<5})", 100.0 * correctos as f32 / n as f32);
                }
                _ => print!("  {:>13}", "-"),
            }
        }
        println!();
    }
}

/// El número que la mata: dentro de zona segura (p[argmax]>=0.8, deciles
/// 8-9), compara precisión en contexto raro (c=2) vs común (10-49, 50+).
fn numero_que_mata(nombre: &str, posiciones: &[Pos]) {
    let seguros: Vec<&Pos> = posiciones.iter().filter(|p| p.p_argmax >= PISO_CONFIANZA).collect();
    let n_total = seguros.len();
    let prec = |banda: &str| -> Option<(f32, usize)> {
        let sub: Vec<&&Pos> = seguros.iter().filter(|p| p.banda == banda).collect();
        if sub.is_empty() {
            return None;
        }
        let c = sub.iter().filter(|p| p.correcto).count();
        Some((c as f32 / sub.len() as f32, sub.len()))
    };
    println!("\n=== {nombre}: EL NÚMERO -- zona segura (p[argmax]>={PISO_CONFIANZA}), n={n_total} ===");
    for b in BANDAS {
        match prec(b) {
            Some((p, n)) => println!("  banda {b:<6}  precisión {:5.1}%  (n={n})", 100.0 * p),
            None => println!("  banda {b:<6}  sin datos"),
        }
    }
    if let (Some((p_raro, n_raro)), Some((p_comun, n_comun))) = (prec("c=2"), prec("50+")) {
        let delta = p_comun - p_raro;
        println!("\n  Δ precisión (50+ menos c=2): {:+.1} puntos  (n_raro={n_raro}, n_comun={n_comun})", 100.0 * delta);
        if n_raro < 30 || n_comun < 30 {
            println!("  ATENCIÓN: al menos una celda tiene n<30 -- no alcanza para decidir solo.");
        }
        if delta > 0.03 {
            println!("  El contexto común tiene precisión claramente mayor en zona segura --");
            println!("  hay overreach: confianza alta con contexto raro es MENOS confiable.");
            println!("  Vale construir el gate curioso.");
        } else if delta < -0.03 {
            println!("  El contexto RARO tiene mayor precisión (sorpresa, contra la hipótesis) --");
            println!("  no hay overreach en la dirección esperada. Revisar antes de actuar.");
        } else {
            println!("  Sin diferencia clara -- no hay overreach medible. La zona segura ya es");
            println!("  casi perfecta pareja, y el tema cierra con este resultado.");
        }
    } else {
        println!("\n  Falta al menos una de las bandas c=2 / 50+ dentro de zona segura -- no se");
        println!("  puede calcular el delta con este corte.");
    }
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

    // Mismo corte que el resto de la ronda: mitad de val para mirar (dev),
    // la otra mitad se toca una sola vez para el número final.
    let mitad = n_train + (n_val / 2);
    let ventanas_dev: Vec<usize> = (n_train..mitad).collect();
    let ventanas_val: Vec<usize> = (mitad..n).collect();
    println!("eva confiado_errado: {} ventanas dev | {} ventanas val", ventanas_dev.len(), ventanas_val.len());

    let pos_dev = recolectar(&model, &ds, &tabla, &ventanas_dev, seq);
    imprimir_tabla("DEV (mirar, no decide)", &pos_dev);
    numero_que_mata("DEV", &pos_dev);

    let pos_val = recolectar(&model, &ds, &tabla, &ventanas_val, seq);
    imprimir_tabla("VAL (tocada una sola vez)", &pos_val);
    numero_que_mata("VAL", &pos_val);
}
