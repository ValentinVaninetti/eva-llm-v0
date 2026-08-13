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
//!
//! Extensión (tarea de Dante, "entropía-por-count"): teoría de Claudio para
//! explicar la reversión de arriba con una sola causa -- count alto podría
//! no significar "contexto común y confiable" sino "contexto de baja
//! información" (fragmento genérico con más continuaciones válidas). Se
//! mide directo: H(q) de Shannon sobre la distribución que ya devuelve
//! `lookup_detail` (re-agregación, no op nueva), por banda de count. El
//! número que la mata: si H no crece con el count, la teoría cae acá.
//!
//! Segunda extensión (cierre del hilo "cabeza de Valentín" / deliberación):
//! H(p), la entropía de Shannon de la distribución de salida del MODELO
//! (no de la tabla -- eso es H(q), arriba). Es la aproximación libre, de
//! una sola pasada, a "si me hicieran generar de nuevo, ¿cuánto discreparía
//! conmigo mismo" -- el costo de un muestreo múltiple real (self-
//! consistency/debate), sin pagarlo. El número que la mata: si H(p) no
//! separa nada que `p[argmax]` no separe ya, la línea barata de
//! "deliberación" cierra. Re-agregación sobre logits ya calculados, cero
//! op nueva.

use eva_llm_v0::bet::classify_row;
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::save::load_model;

const PISO_CONFIANZA: f32 = 0.8; // mismo piso que lambda_adaptativo.rs

struct Pos {
    p_argmax: f32,
    correcto: bool,
    banda: &'static str,
    /// H(q) en bits sobre la distribución de continuaciones de la tabla en
    /// esta posición. None si no hubo hit (banda "miss": no hay q).
    entropia: Option<f32>,
    /// H(p) en bits sobre la distribución de salida del MODELO. Siempre
    /// disponible -- no depende de que la tabla haya contestado.
    h_p: f32,
}

/// Entropía de Shannon en bits (log2), 0*log2(0) := 0 por convención.
fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

/// H(p) directo desde logits crudos: softmax + Shannon, sin op nueva.
fn entropia_logits_bits(row: &[f32]) -> f32 {
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&z| (z - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    shannon_bits(&probs)
}

/// AUROC por rangos (Mann-Whitney), mismo método que `sonda_tabla.rs`.
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
            let h_p = entropia_logits_bits(row);

            // mismo indexado que Paso 0 / sonda_tabla, ya corregido una vez.
            let pos_global = wi * seq + t + 1;
            let desde = pos_global.saturating_sub(8);
            let ctx = &ds.ids[desde..pos_global];
            let hit = if ctx.is_empty() { None } else { tabla.lookup_detail(ctx, vocab) };
            let entropia = hit.as_ref().map(|(q, _)| shannon_bits(q));
            let banda = banda_count(hit);

            out.push(Pos { p_argmax, correcto, banda, entropia, h_p });
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

/// La teoría de Claudio, medida directo: H(q) media por banda de count.
/// Si crece con el count, "count alto" es contexto de baja información
/// (más continuaciones válidas), lo cual explicaría por qué la vara de la
/// tabla y la precisión del modelo bajan las dos con count alto. Si no
/// crece, la teoría cae acá.
fn entropia_por_banda(nombre: &str, posiciones: &[Pos]) {
    let vocab_bits = 8.0f32; // techo teórico: log2(256)
    println!("\n=== {nombre}: H(q) de Shannon (bits) por banda de count, techo={vocab_bits} bits ===");
    let mut medias = Vec::new();
    for b in ["c=2", "3-9", "10-49", "50+"] {
        let vals: Vec<f32> = posiciones.iter()
            .filter(|p| p.banda == b)
            .filter_map(|p| p.entropia)
            .collect();
        if vals.is_empty() {
            println!("  banda {b:<6}  sin datos");
            continue;
        }
        let n = vals.len();
        let media = vals.iter().sum::<f32>() / n as f32;
        let mut orden = vals.clone();
        orden.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let mediana = orden[n / 2];
        println!("  banda {b:<6}  H media {media:.3} bits  |  H mediana {mediana:.3}  |  n={n}");
        medias.push((b, media));
    }

    println!("\n=== {nombre}: EL NÚMERO -- ¿crece H con el count? ===");
    if let (Some(&(_, h_raro)), Some(&(_, h_comun))) =
        (medias.iter().find(|(b, _)| *b == "c=2"), medias.iter().find(|(b, _)| *b == "50+"))
    {
        let delta = h_comun - h_raro;
        println!("  Δ H (50+ menos c=2): {delta:+.3} bits");
        let monotona = medias.windows(2).all(|w| w[1].1 >= w[0].1 - 0.02);
        println!("  monótona no-decreciente c=2→3-9→10-49→50+: {monotona}");
        if delta > 0.1 && monotona {
            println!("  H crece con el count, monótona -- confirma la teoría: count alto es");
            println!("  contexto de baja información (más continuaciones válidas), explica las");
            println!("  dos anomalías (vara de tabla y precisión del modelo bajando con count).");
        } else if delta > 0.1 {
            println!("  H crece de punta a punta pero NO monótona -- señal parcial, no la teoría");
            println!("  limpia. Anotar, no cerrar como confirmada.");
        } else {
            println!("  H NO crece claramente con el count -- la teoría CAE acá. La reversión de");
            println!("  \"confiado y errado\" queda sin explicación, hay que buscar otra causa.");
        }
    } else {
        println!("  Falta alguna banda para calcular el delta.");
    }
}

/// Cierre del hilo "deliberación barata": ¿H(p) predice acierto del modelo
/// mejor, o al menos ADEMÁS, de lo que ya predice p[argmax] gratis? Si no,
/// la aproximación de una sola pasada a "cuánto discreparía conmigo mismo"
/// no aporta nada que la confianza ya no diga, y la línea barata cierra.
fn hp_vs_pargmax(nombre: &str, posiciones: &[Pos]) {
    let pool_pargmax: Vec<(f64, bool)> = posiciones.iter().map(|p| (p.p_argmax as f64, p.correcto)).collect();
    // score = -H(p): menos entropía tiene que correlacionar con acierto,
    // igual dirección que p[argmax].
    let pool_hp: Vec<(f64, bool)> = posiciones.iter().map(|p| (-p.h_p as f64, p.correcto)).collect();

    let auc_pargmax = auroc(&pool_pargmax);
    let auc_hp = auroc(&pool_hp);

    println!("\n=== {nombre}: EL NÚMERO -- H(p) vs p[argmax], n={} ===", posiciones.len());
    println!("  AUROC p[argmax] (gratis, ya publicado hoy):        {auc_pargmax:.3}");
    println!("  AUROC -H(p) (aprox. libre de \"discreparía conmigo\"): {auc_hp:.3}");
    println!("  (el AUROC global de las dos casi no separa -- están pegadas por");
    println!("  construcción, una distribución picuda tiene alto p[argmax] Y bajo H(p)");
    println!("  a la vez. La pregunta que importa es la condicional, abajo.)");

    // Residual: dentro de zona segura (donde importan los errores caros),
    // separa por mediana de H(p) -- si p[argmax] ya se comió toda la señal,
    // las dos mitades tienen que salir con precisión pareja. ESTE, no el
    // AUROC global de arriba, es el número que decide el veredicto.
    let seguros: Vec<&Pos> = posiciones.iter().filter(|p| p.p_argmax >= PISO_CONFIANZA).collect();
    let mut delta_condicional: Option<(f32, usize, usize)> = None;
    if seguros.len() >= 20 {
        let mut hs: Vec<f32> = seguros.iter().map(|p| p.h_p).collect();
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mediana_h = hs[hs.len() / 2];
        let (baja, alta): (Vec<&&Pos>, Vec<&&Pos>) = seguros.iter().partition(|p| p.h_p <= mediana_h);
        let prec = |g: &[&&Pos]| -> f32 {
            if g.is_empty() { return f32::NAN; }
            g.iter().filter(|p| p.correcto).count() as f32 / g.len() as f32
        };
        let (p_baja, p_alta) = (prec(&baja), prec(&alta));
        println!("\n  dentro de zona segura (p[argmax]>={PISO_CONFIANZA}), split por mediana H(p)={mediana_h:.3}:");
        println!("    H(p) baja (más determinística)  precisión {:5.1}%  (n={})", 100.0 * p_baja, baja.len());
        println!("    H(p) alta (menos determinística) precisión {:5.1}%  (n={})", 100.0 * p_alta, alta.len());
        println!("    Δ: {:+.1} puntos", 100.0 * (p_alta - p_baja));
        delta_condicional = Some((p_baja - p_alta, baja.len(), alta.len()));
    }

    println!("\n=== {nombre}: VEREDICTO (sobre el delta CONDICIONAL, no el AUROC global) ===");
    match delta_condicional {
        Some((delta, n_baja, n_alta)) if delta > 0.03 && n_baja >= 30 && n_alta >= 30 => {
            println!("  A confianza igual, H(p) separa {:.1} puntos de precisión (n={n_baja}/{n_alta}).", 100.0 * delta);
            println!("  H(p) le agrega algo a p[argmax] que p[argmax] solo no ve. Primera");
            println!("  justificación barata real para pensar en la versión cara (muestreo");
            println!("  múltiple de verdad). No cierra la línea de deliberación, la abre.");
        }
        _ => {
            println!("  Sin diferencia condicional clara (o n insuficiente) -- H(p) no agrega");
            println!("  nada que p[argmax] no tuviera ya. Cierra la línea barata de deliberación.");
        }
    }
}

/// Gate 2D, aprobado por Dante ("la pelota que dejaste"): zona segura más
/// angosta que la de `numero_que_mata` (que sólo usa p[argmax]) -- exige
/// TAMBIÉN que la forma del resto de la distribución sea determinística
/// (H(p) bajo su propia mediana dentro de zona segura). Formaliza como
/// medición propia lo que en `hp_vs_pargmax` salía como sub-producto.
fn gate_2d(nombre: &str, posiciones: &[Pos]) {
    let seguros: Vec<&Pos> = posiciones.iter().filter(|p| p.p_argmax >= PISO_CONFIANZA).collect();
    if seguros.len() < 20 {
        println!("\n=== {nombre}: gate 2D -- n insuficiente en zona segura ===");
        return;
    }
    let mut hs: Vec<f32> = seguros.iter().map(|p| p.h_p).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mediana_h = hs[hs.len() / 2];

    let prec_de = |g: &[&Pos]| -> f32 {
        if g.is_empty() { return f32::NAN; }
        g.iter().filter(|p| p.correcto).count() as f32 / g.len() as f32
    };
    let precision_1d = prec_de(&seguros);
    let angosta: Vec<&Pos> = seguros.iter().filter(|p| p.h_p <= mediana_h).copied().collect();
    let precision_2d = prec_de(&angosta);

    println!("\n=== {nombre}: gate 2D -- p[argmax]≥{PISO_CONFIANZA} Y H(p)≤{mediana_h:.3} ===");
    println!("  zona segura 1D (sólo p[argmax]):        precisión {:5.1}%  (n={})", 100.0 * precision_1d, seguros.len());
    println!("  zona segura 2D (+H(p)≤mediana):         precisión {:5.1}%  (n={}, {:.0}% de la 1D)",
        100.0 * precision_2d, angosta.len(), 100.0 * angosta.len() as f32 / seguros.len() as f32);
    println!("  ganancia de precisión por angostar: {:+.1} puntos, a costo de {:.0}% menos cobertura",
        100.0 * (precision_2d - precision_1d), 100.0 * (1.0 - angosta.len() as f32 / seguros.len() as f32));
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
    entropia_por_banda("DEV (mirar, no decide)", &pos_dev);
    hp_vs_pargmax("DEV (mirar, no decide)", &pos_dev);
    gate_2d("DEV (mirar, no decide)", &pos_dev);

    let pos_val = recolectar(&model, &ds, &tabla, &ventanas_val, seq);
    imprimir_tabla("VAL (tocada una sola vez)", &pos_val);
    numero_que_mata("VAL", &pos_val);
    entropia_por_banda("VAL (tocada una sola vez)", &pos_val);
    hp_vs_pargmax("VAL (tocada una sola vez)", &pos_val);
    gate_2d("VAL (tocada una sola vez)", &pos_val);
}
