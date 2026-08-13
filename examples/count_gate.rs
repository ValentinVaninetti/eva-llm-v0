//! Distribución de counts del corpus sintético del gate, y el λ efectivo que
//! produce `mixed_ce_count` con c0 = 2. Responde a: en el corpus donde el gate
//! se probó, ¿cuántas posiciones conservan el gradiente entero (piso, c = 2)
//! y cuántas quedan atenuadas como boilerplate (c alto)?
//!
//! Usa sólo API pública (`TextDataset`, `Recall::lookup_detail`): no toca
//! `src/`, no carga modelo, no usa GPU.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;

fn main() {
    let data = std::env::args().nth(1).unwrap_or_else(|| "data/sintetico-gate.txt".into());
    let seq = 64usize;
    let lambda = 0.35f32;
    let c0 = 2.0f32;

    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_train = n - (((n as f32) * 0.2).round() as usize);
    let train: Vec<usize> = ds.ids[..n_train * seq].to_vec();
    let tabla = Recall::build(&train);

    let mut piso = 0u32;   // c = 2  → λ efectivo 0 (gradiente entero)
    let mut bajo = 0u32;   // c en 3..=9
    let mut medio = 0u32;  // c en 10..=49
    let mut alto = 0u32;   // c >= 50
    let mut miss = 0u32;
    for i in 0..train.len() {
        let ctx = &train[i.saturating_sub(8)..i];
        if ctx.is_empty() {
            continue;
        }
        match tabla.lookup_detail(ctx, 256) {
            None => miss += 1,
            Some((_, c)) if c == 2 => piso += 1,
            Some((_, c)) if c <= 9 => bajo += 1,
            Some((_, c)) if c <= 49 => medio += 1,
            Some(_) => alto += 1,
        }
    }
    let tot = (piso + bajo + medio + alto + miss) as f32;
    let pct = |x: u32| 100.0 * x as f32 / tot;
    println!("eva count_gate: {n_train} ventanas de train, {tot:.0} posiciones");
    println!("  count 2  (piso, λ_i=0, gradiente ENTERO):   {:6.1}%", pct(piso));
    println!("  count 3-9 (λ_i={:.2}-{:.2})             :   {:6.1}%",
        lambda * (1.0 - c0 / 9.0), lambda * (1.0 - c0 / 3.0), pct(bajo));
    println!("  count 10-49 (λ_i={:.2}-{:.2})          :   {:6.1}%",
        lambda * (1.0 - c0 / 49.0), lambda * (1.0 - c0 / 10.0), pct(medio));
    println!("  count >=50 (λ_i≈{:.3})                 :   {:6.1}%",
        lambda * (1.0 - c0 / 50.0), pct(alto));
    println!("  sin respuesta (miss)                      :   {:6.1}%", pct(miss));
    let atenuadas = 100.0 * (alto + medio + bajo) as f32 / tot;
    println!("\n  posiciones ATENUADAS por el gate: {atenuadas:.1}% -- el gate sólo");
    println!("  deja el gradiente entero en el {:.1}% que es count 2 (los hechos raros).",
        pct(piso));
}
