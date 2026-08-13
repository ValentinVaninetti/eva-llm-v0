//! Tarea 1 de TRABAJO-2026-08-12.md: ¿la bisección por mitades localiza el
//! primer error dentro de un tramo malo, o hay algo mejor con lo que ya
//! tenemos?
//!
//! CERO forward passes nuevos: re-agrega la traza que `bet::scan` ya
//! devuelve. Para cada tramo (largo 16) con `bien < 0.5` sobre la
//! validación, define primer_error = primera posición con `correct ==
//! false`, y compara tres formas de estimarlo:
//!
//! (a) vara del tramo entero -- no localiza nada, hay que re-tirar el tramo
//!     completo. Es la referencia de "no localizar".
//! (b) bisección: parte el tramo a la mitad, se queda con la mitad de menor
//!     confianza media, recursa hasta un solo token. Es la candidata de
//!     BitVMX-por-mitades tal como se propuso.
//! (c) change-point por token: el token de menor confianza declarada
//!     (`conf`) dentro del tramo -- no bisecta nada, mira la traza entera.
//!
//! Métricas: AUROC agrupado ("¿este token es el primer error?", sobre todos
//! los tramos malos juntos), distancia media |predicho - real|, y el
//! payoff -- si se re-tira sólo desde el punto predicho hasta el final del
//! tramo, cuánto del tramo se ahorra y qué fracción de las veces el punto
//! predicho cae ANTES o EN el error real (así el re-tiro lo cubre).

use eva_llm_v0::bet::{scan, TokenObs};
use eva_llm_v0::data::TextDataset;
use eva_llm_v0::save::load_model;

const SPAN: usize = 16;
const UMBRAL_BIEN: f64 = 0.5;

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa250.txt".into());
    let val: f32 = 0.1;

    let model = load_model(&weights).expect("no pude cargar el checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    println!("eva bisector: {} params | {} ventanas val (mismo corte que eva bet)",
        model.param_count(), n_val);

    let obs = scan(&model, &ds, n_train, n);
    println!("eva bisector: {} observaciones por token, una sola pasada", obs.len());

    // Tramos de SPAN, sin cruzar ventanas -- misma regla que span_analysis.
    let mut malos: Vec<&[TokenObs]> = Vec::new();
    let mut i = 0;
    while i < obs.len() {
        let hasta = (i / seq + 1) * seq;
        let mut t = i;
        while t + SPAN <= hasta {
            let tramo = &obs[t..t + SPAN];
            let bien = tramo.iter().filter(|o| o.correct).count() as f64 / SPAN as f64;
            if bien < UMBRAL_BIEN {
                malos.push(tramo);
            }
            t += SPAN;
        }
        i = hasta;
    }
    println!("eva bisector: {} tramos con bien < {:.1} de un total de {} tramos posibles\n",
        malos.len(), UMBRAL_BIEN, obs.len() / SPAN);

    if malos.is_empty() {
        println!("no hay tramos malos con este umbral -- no hay nada que localizar, listo.");
        return;
    }

    // Primer error real por tramo.
    let primer_error: Vec<usize> = malos.iter()
        .map(|t| t.iter().position(|o| !o.correct).unwrap_or(0))
        .collect();

    // (b) bisección recursiva por mitades, y un score por-token = cuántas
    // veces sobrevivió como "la mitad peor".
    let mut b_pred = Vec::new();
    let mut b_scores_pool: Vec<(f64, bool)> = Vec::new(); // (score, es_primer_error)
    for (tramo, &real) in malos.iter().zip(&primer_error) {
        let mut score = vec![0u32; SPAN];
        let (mut lo, mut hi) = (0usize, SPAN);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let media = |a: usize, b: usize| -> f64 {
                tramo[a..b].iter().map(|o| o.conf as f64).sum::<f64>() / (b - a) as f64
            };
            let (izq, der) = (media(lo, mid), media(mid, hi));
            if izq <= der { hi = mid } else { lo = mid }
            for s in score.iter_mut().take(hi).skip(lo) { *s += 1; }
        }
        b_pred.push(lo);
        for (k, &s) in score.iter().enumerate() {
            b_scores_pool.push((s as f64, k == real));
        }
    }

    // (c) change-point: el token de menor `conf` en el tramo. Score = 1-conf.
    let mut c_pred = Vec::new();
    let mut c_scores_pool: Vec<(f64, bool)> = Vec::new();
    for (tramo, &real) in malos.iter().zip(&primer_error) {
        let (mut peor_i, mut peor_v) = (0usize, f32::INFINITY);
        for (k, o) in tramo.iter().enumerate() {
            if o.conf < peor_v { peor_v = o.conf; peor_i = k; }
            c_scores_pool.push(((1.0 - o.conf) as f64, k == real));
        }
        c_pred.push(peor_i);
    }

    let auroc_b = auroc(&b_scores_pool);
    let auroc_c = auroc(&c_scores_pool);
    let dist_b = media_dist(&b_pred, &primer_error);
    let dist_c = media_dist(&c_pred, &primer_error);
    let dist_random = SPAN as f64 / 3.0; // valor esperado de |U(0,15) - U(0,15)| aprox

    println!("=== localización del primer error, {} tramos malos ===", malos.len());
    println!("  estrategia         AUROC   dist. media   dist. random (piso)");
    println!("  (b) bisección       {:.3}      {:5.2}            {:5.2}", auroc_b, dist_b, dist_random);
    println!("  (c) change-point    {:.3}      {:5.2}            {:5.2}", auroc_c, dist_c, dist_random);

    // Payoff: re-tirar desde el punto predicho hasta el final del tramo.
    let payoff = |preds: &[usize]| -> (f64, f64) {
        let frac: f64 = preds.iter().map(|&p| (SPAN - p) as f64 / SPAN as f64).sum::<f64>() / preds.len() as f64;
        let recall: f64 = preds.iter().zip(&primer_error)
            .filter(|(&p, &r)| p <= r).count() as f64 / preds.len() as f64;
        (frac, recall)
    };
    let (frac_b, rec_b) = payoff(&b_pred);
    let (frac_c, rec_c) = payoff(&c_pred);

    println!("\n=== payoff: re-tirar sólo desde el punto predicho hasta el final ===");
    println!("  (a) tramo entero     ahorra   0.0% del tramo | cubre 100.0% de los errores (trivial)");
    println!("  (b) bisección        ahorra {:5.1}% del tramo | cubre {:5.1}% de los errores", 100.0 * frac_b, 100.0 * rec_b);
    println!("  (c) change-point     ahorra {:5.1}% del tramo | cubre {:5.1}% de los errores", 100.0 * frac_c, 100.0 * rec_c);

    println!("\n=== VEREDICTO ===");
    if auroc_c > auroc_b + 0.03 {
        println!("  change-point por token gana claro -- la bisección por mitades es el objeto");
        println!("  equivocado, como predijo la literatura. No se construye el bisector.");
    } else if auroc_b > auroc_c + 0.03 {
        println!("  la bisección por mitades gana -- vale la pena seguirla, contra la predicción");
        println!("  de la literatura. Conviene remedir con otra semilla antes de construir.");
    } else {
        println!("  empatan dentro del ruido -- ninguna localiza mucho mejor que la otra.");
    }
    if auroc_b.max(auroc_c) < 0.6 {
        println!("  y ninguna de las dos despega mucho de 0.5 (azar): localizar el PRIMER token");
        println!("  del error puede no ser un problema barato, aunque detectar el tramo sí lo era.");
    }
}

fn media_dist(pred: &[usize], real: &[usize]) -> f64 {
    pred.iter().zip(real).map(|(&p, &r)| (p as f64 - r as f64).abs()).sum::<f64>() / pred.len() as f64
}

/// AUROC por rangos (Mann-Whitney), sobre pares (score, es_positivo).
fn auroc(pares: &[(f64, bool)]) -> f64 {
    let mut v: Vec<(f64, bool)> = pares.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = v.len() as f64;
    let n_pos = v.iter().filter(|(_, p)| *p).count() as f64;
    let n_neg = n - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    // Rango promedio para empates.
    let mut suma_rangos_pos = 0.0f64;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 { j += 1; }
        let rango_medio = (i + 1 + j) as f64 / 2.0; // rangos 1-based, promedio del bloque
        for k in i..j {
            if v[k].1 { suma_rangos_pos += rango_medio; }
        }
        i = j;
    }
    (suma_rangos_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}
