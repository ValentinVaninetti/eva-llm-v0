//! ¿De dónde sale el ~99% de cobertura que encontró Dante en el mix de
//! entrenamiento? Sospecha: si la mayoría de los aciertos vienen de órdenes
//! BAJOS (2-3, bigramas/trigramas -- estadística genérica del idioma) y no de
//! órdenes ALTOS (6-8, contexto específico), entonces un λ constante está
//! atenuando el gradiente casi siempre por razones equivocadas: el modelo
//! aprendería lo mismo con o sin la tabla en esas posiciones, y ahí no había
//! nada que "liberar". El λ debería depender de QUÉ ORDEN contestó, no ser
//! un número fijo.
//!
//! Sólo usa la tabla (CPU, sin cargar ningún modelo, sin GPU) -- cero
//! colisión con la corrida de Dante. Y no toca `recall.rs`: todo lo que usa
//! ya es público (`build`, `lookup`, `hit_profile`).

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::recall::Recall;

/// PASO 0 de la propuesta de la señal auxiliar (TRABAJO-2026-08-12.md): la
/// vara que gobierna las dos candidatas (cabecita de acuerdo / suma de
/// logits) no es azar, es "la tabla ya tiene razón casi siempre". Si
/// argmax(tabla) == verdad ya es ~97% donde el count es alto, cualquier
/// mecanismo nuevo tiene que ganarle a ESO, no a 50%.
///
/// Sobre VALIDACIÓN (no train): es la condición real en la que tendría que
/// funcionar cualquiera de las dos candidatas -- generalizar a texto que la
/// tabla no memorizó, igual que mide todo lo demás en este proyecto.
fn paso_0_vara_por_count(tabla: &Recall, ds: &TextDataset, n_train: usize, seq: usize) {
    let umbrales = [2u32, 5, 10, 50];
    let mut aciertos = vec![0usize; umbrales.len()];
    let mut totales = vec![0usize; umbrales.len()];

    for wi in n_train..ds.num_windows() {
        let (input, target) = ds.window(wi);
        for t in 0..input.len() {
            // Contexto = los bytes reales anteriores en el corpus completo
            // (puede venir de la cola de train para las primeras posiciones
            // de val -- es texto real, no fuga: la tabla no se construyó con
            // ningún byte de acá).
            //
            // OJO: target[t] = ids[wi*seq + t + 1] (TextDataset::window corre
            // el target uno respecto del input). La posición global del byte
            // que se predice es wi*seq+t+1, NO wi*seq+t -- me equivoqué acá
            // primero y el contexto le faltaba el byte más reciente. Con el
            // bug, la vara daba ~5%; corregido, se ve el número real abajo.
            let pos_global = wi * seq + t + 1;
            let desde = pos_global.saturating_sub(8);
            let ctx = &ds.ids[desde..pos_global];
            if ctx.is_empty() {
                continue;
            }
            let Some((q, count)) = tabla.lookup_detail(ctx, 256) else { continue };
            let argmax = q.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
                .0;
            let acierto = argmax == target[t];
            for (u, &c0) in umbrales.iter().enumerate() {
                if count >= c0 {
                    totales[u] += 1;
                    if acierto { aciertos[u] += 1; }
                }
            }
        }
    }

    println!("\n=== PASO 0: vara de argmax(tabla) == verdad, sobre VALIDACIÓN ===");
    println!("  count ≥ c0   n        acierto");
    for (u, &c0) in umbrales.iter().enumerate() {
        let pct = 100.0 * aciertos[u] as f32 / totales[u].max(1) as f32;
        println!("  {c0:>8}   {:>6}   {pct:6.1}%", totales[u]);
    }
    println!("\n  VEREDICTO: esta es la vara real para las dos candidatas (cabecita de");
    println!("  acuerdo / suma de logits). Si el mecanismo nuevo no le gana a estos");
    println!("  números en su propio umbral de count, no agrega nada -- la tabla sola");
    println!("  ya acierta eso, sin gastar un parámetro ni una línea de gradiente.");
}

fn main() {
    let data = std::env::args().nth(1).unwrap_or_else(|| "data/prosa250.txt".into());
    let seq = 64usize;
    let val = 0.1f32;

    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;
    let bytes_train: Vec<usize> = ds.ids[..n_train * seq].to_vec();

    let tabla = Recall::build(&bytes_train);
    println!("eva orden_cobertura: tabla construida sobre {} bytes de train ({} ventanas)",
        bytes_train.len(), n_train);

    // Recorre el propio texto de entrenamiento con una ventana de contexto
    // creciente hasta 8, igual que vería el modelo en cada posición.
    let mut cubiertas = 0usize;
    let mut total = 0usize;
    for i in 0..bytes_train.len() {
        let desde = i.saturating_sub(8);
        let ctx = &bytes_train[desde..i];
        if ctx.is_empty() {
            continue;
        }
        total += 1;
        if tabla.lookup(ctx, 256).is_some() {
            cubiertas += 1;
        }
    }

    println!("\n=== cobertura sobre el propio texto de entrenamiento ===");
    println!("  {cubiertas} / {total} posiciones ({:.1}%) -- este es el número que atenúa Dante", 100.0 * cubiertas as f32 / total.max(1) as f32);

    println!("\n=== de qué ORDEN viene cada acierto ===");
    println!("  orden   % de los aciertos");
    let mut bajo = 0.0f32; // 2-3
    let mut alto = 0.0f32; // 6-8
    for (k, pct) in tabla.hit_profile() {
        println!("  {k:>5}   {pct:5.1}%");
        if k <= 3 { bajo += pct } else if k >= 6 { alto += pct }
    }

    println!("\n=== VEREDICTO ===");
    println!("  órdenes bajos (2-3, genéricos):  {bajo:5.1}%");
    println!("  órdenes altos (6-8, específicos): {alto:5.1}%");
    if bajo > alto * 2.0 {
        println!("  La cobertura la dominan los órdenes BAJOS -- son bigramas/trigramas, no");
        println!("  hechos memorizados. Un λ constante atenúa el gradiente ahí por igual que en");
        println!("  un hit de orden 8, y ahí no había nada que 'liberar': el modelo aprendería");
        println!("  esa estadística igual, con o sin tabla. Candidata real: λ dependiente del");
        println!("  orden que contestó (o del count/entropía de la distribución q), no un número fijo.");
    } else {
        println!("  La cobertura no está dominada por órdenes bajos -- la hipótesis de 'λ ciego");
        println!("  al orden' no explica el negativo por sí sola, hay que mirar otra cosa.");
    }

    paso_0_vara_por_count(&tabla, &ds, n_train, seq);
}
