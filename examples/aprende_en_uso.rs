//! Ronda 4, próxima pieza anotada por Dante: el pibe no aprende de sus
//! errores EN USO. La tabla se construye una vez con train y después es
//! sólo lectura -- cuando el sistema final erra y la respuesta real llega
//! (acá: el byte real del corpus, simulando streaming/transcripción/
//! corrección), ese error se pierde. "El pibe mejora mientras trabaja."
//!
//! Diseño del maestro (Dante), tal cual:
//! - Marca el PAR EXACTO (contexto -> byte real), no un concepto.
//! - Marca sólo donde el sistema FINAL erró Y la memoria estaba ignorante
//!   (miss, o hit con H(q) alta). Si la tabla ya sabía, el hit ya lo
//!   corrige -- marcar ahí es redundante.
//! - Entra como +1 de conteo con la MISMA disciplina de `build`
//!   (MIN_COUNT=2 en `find`), no como bandera "ésta es LA respuesta". El
//!   maestro que remarca, no el que grita.
//!
//! Único cambio de `src/`: `Recall::observe` (write path, ya commiteado por
//! separado). El resto es re-agregación: el mismo forward del modelo
//! (frozen, sin entrenar acá) para control y tratamiento, sólo difiere la
//! tabla.
//!
//! Control: misma segunda mitad del stream, memoria SIEMPRE congelada
//! (nunca se llama `observe`). El número que decide: segunda mitad
//! TRATAMIENTO vs segunda mitad CONTROL -- no "primera vs segunda mitad
//! dentro del tratamiento", que se confundiría con que el texto se vuelva
//! más fácil más adelante por razones que no tienen nada que ver con
//! aprender.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;
use eva_llm_v0::save::load_model;

const CAP: f32 = 7.7; // mismo cap de Séneca de siempre
const EPS: f32 = 1e-9;
const LAMBDA: f32 = 0.2; // λ fijo publicado esta semana (kNN-LM style)
const H_HIGH: f32 = 2.0; // mismo umbral de "memoria ignorante" del regularizador

fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

/// El sistema FINAL: modelo + tabla combinados log-lineal (misma fórmula
/// que `lambda_adaptativo.rs`, λ fijo). Devuelve (nats de esta posición,
/// argmax combinado, Some(H(q)) si hubo hit).
fn combinar(logits_m: &[f32], target: usize, tabla: &Recall, ctx: &[usize], vocab: usize) -> (f32, usize, Option<f32>) {
    let hit = if ctx.is_empty() { None } else { tabla.lookup_detail(ctx, vocab) };
    let mx_m = logits_m.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bias: Option<Vec<f32>> = hit.as_ref().map(|(q, _)| {
        q.iter().map(|&qi| (qi + EPS).ln().max(-CAP)).collect()
    });
    let h_q = hit.as_ref().map(|(q, _)| shannon_bits(q));

    let mut sum = 0.0f32;
    let mut z_tgt = 0.0f32;
    let (mut argmax, mut mejor) = (0usize, f32::NEG_INFINITY);
    for (k, &zm) in logits_m.iter().enumerate() {
        let zc = zm - mx_m + LAMBDA * bias.as_ref().map(|b| b[k]).unwrap_or(0.0);
        let e = zc.exp();
        sum += e;
        if k == target {
            z_tgt = zc;
        }
        if zc > mejor {
            mejor = zc;
            argmax = k;
        }
    }
    let nats = -(z_tgt - sum.ln());
    (nats, argmax, h_q)
}

struct Resultado {
    n: usize,
    nats: f64,
    correctos: usize,
    ensenanzas: usize,
}

impl Resultado {
    fn nuevo() -> Self {
        Resultado { n: 0, nats: 0.0, correctos: 0, ensenanzas: 0 }
    }
    fn bpb(&self) -> f64 {
        self.nats / self.n.max(1) as f64 / std::f64::consts::LN_2
    }
    fn precision(&self) -> f32 {
        100.0 * self.correctos as f32 / self.n.max(1) as f32
    }
}

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("no pude cargar el checkpoint");
    let vocab = model.cfg.vocab;
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();

    let tabla_control = Recall::build(&bytes_train);
    let mut tabla_tratamiento = Recall::build(&bytes_train);

    // Stream ordenado, contiguo -- mismo orden que el corpus real. El
    // forward del modelo es idéntico para las dos condiciones (frozen, no
    // se entrena acá); sólo la tabla difiere entre control y tratamiento.
    let ventanas: Vec<usize> = (n_train..n).collect();
    println!("eva aprende_en_uso: {} ventanas de stream ({} posiciones)", ventanas.len(), ventanas.len() * seq);

    let mut mitad1_control = Resultado::nuevo();
    let mut mitad2_control = Resultado::nuevo();
    let mut mitad1_trat = Resultado::nuevo();
    let mut mitad2_trat = Resultado::nuevo();

    let total_posiciones = ventanas.len() * seq;
    let mut i_stream = 0usize;

    for &wi in &ventanas {
        let (input, target) = ds.window(wi);
        let (logits, _) = model.forward_hidden(&input);
        for t in 0..input.len() {
            let row = &logits.data[t * vocab..(t + 1) * vocab];
            let pos_global = wi * seq + t + 1;
            let desde = pos_global.saturating_sub(8);
            let ctx = &ds.ids[desde..pos_global];
            let tgt = target[t];

            let primera_mitad = i_stream < total_posiciones / 2;
            i_stream += 1;

            // CONTROL: memoria siempre congelada.
            let (nats_c, argmax_c, _) = combinar(row, tgt, &tabla_control, ctx, vocab);
            let r_c = if primera_mitad { &mut mitad1_control } else { &mut mitad2_control };
            r_c.n += 1;
            r_c.nats += nats_c as f64;
            if argmax_c == tgt { r_c.correctos += 1; }

            // TRATAMIENTO: predice con el estado actual, después enseña si
            // corresponde -- la enseñanza de ESTA posición nunca puede
            // filtrarse a la predicción de ESTA misma posición.
            let (nats_t, argmax_t, h_q) = combinar(row, tgt, &tabla_tratamiento, ctx, vocab);
            let r_t = if primera_mitad { &mut mitad1_trat } else { &mut mitad2_trat };
            r_t.n += 1;
            r_t.nats += nats_t as f64;
            let acerto = argmax_t == tgt;
            if acerto { r_t.correctos += 1; }

            let memoria_ignorante = h_q.map(|h| h > H_HIGH).unwrap_or(true); // None = miss = ignorante
            if !acerto && memoria_ignorante {
                tabla_tratamiento.observe(ctx, tgt as u8);
                r_t.ensenanzas += 1;
            }
        }
    }

    println!("\n=== EL NÚMERO: bpb y precisión, primera vs segunda mitad del stream ===");
    println!("  {:<28} {:>10} {:>10} {:>8}", "condición", "bpb", "precisión", "n");
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "control  1ª mitad", mitad1_control.bpb(), mitad1_control.precision(), mitad1_control.n);
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "control  2ª mitad", mitad2_control.bpb(), mitad2_control.precision(), mitad2_control.n);
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "tratamiento 1ª mitad", mitad1_trat.bpb(), mitad1_trat.precision(), mitad1_trat.n);
    println!("  {:<28} {:>10.4} {:>10.1}% {:>8}", "tratamiento 2ª mitad", mitad2_trat.bpb(), mitad2_trat.precision(), mitad2_trat.n);
    println!("\n  enseñanzas (erró Y memoria ignorante) en 1ª mitad: {}", mitad1_trat.ensenanzas);
    println!("  enseñanzas (erró Y memoria ignorante) en 2ª mitad: {}", mitad2_trat.ensenanzas);

    println!("\n=== VEREDICTO -- comparación que decide: 2ª mitad tratamiento vs 2ª mitad control ===");
    let delta_bpb = mitad2_trat.bpb() - mitad2_control.bpb();
    let delta_prec = mitad2_trat.precision() - mitad2_control.precision();
    println!("  Δ bpb (tratamiento - control), 2ª mitad: {delta_bpb:+.4}");
    println!("  Δ precisión (tratamiento - control), 2ª mitad: {delta_prec:+.1} puntos");
    // Control metodológico: si la 2ª mitad YA es más fácil que la 1ª en el
    // propio control (memoria congelada), el texto se volvió más fácil por
    // razones ajenas al aprendizaje -- hay que descontarlo antes de
    // festejar cualquier mejora del tratamiento.
    let deriva_texto = mitad2_control.bpb() - mitad1_control.bpb();
    println!("  (deriva del texto, control 2ª-1ª mitad: {deriva_texto:+.4} bpb -- si no es ~0,");
    println!("   el corpus solo ya cambia de dificultad, hay que leer el delta de arriba con eso en cuenta)");
    if delta_bpb < -0.005 {
        println!("  El tratamiento mejora sobre el control en la 2ª mitad -- aprender en uso");
        println!("  aporta algo medible más allá de la deriva del propio texto.");
    } else if delta_bpb > 0.005 {
        println!("  El tratamiento empeora sobre el control -- las enseñanzas metieron ruido");
        println!("  neto (votos ambiguos pesando mal) en vez de ayudar.");
    } else {
        println!("  Sin diferencia medible -- a esta escala de stream (n={} por mitad) no hay", mitad2_control.n);
        println!("  suficiente repetición dentro de un solo pase de val para que aprender en");
        println!("  uso se note. No refuta el mecanismo, dice que hace falta más stream/repetición.");
    }
}
