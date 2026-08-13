//! Ronda 4, tarea de Dante ("Número 1, gratis"): ¿el cuello de la tabla es
//! CONTEXTO (orden 8 corta información que sí desambigua más atrás) o DATA
//! (aunque se mire más lejos, el corpus tiene de verdad más de una
//! continuación válida)? Se separa re-agregando la tabla con órdenes más
//! largos (16 y 32) y mirando si la vara sube / H(q) baja sobre la MISMA
//! población de consultas que la cadena de siempre (8,6,4,3,2).
//!
//! NO toca `src/recall.rs`: `ORDERS` está topeado en 8 ahí a propósito
//! (empaqueta en un u64 sin colisión). Para 16 y 32 hace falta una clave más
//! larga -- acá se usa `Vec<u8>` exacto como clave de `HashMap` (sin pérdida
//! de información, sin necesidad de empaquetar en un entero: dos contextos
//! distintos nunca son la misma clave). Reimplementación mínima y local del
//! mismo backoff que `Recall`, sólo para esta medición. Sin modelo: "vara" y
//! H(q) son propiedades de la tabla sola.

use eva_llm_v0::data::TextDataset;
use std::collections::HashMap;

const MIN_COUNT: u32 = 2; // mismo umbral que src/recall.rs (privado ahí)
const SEQ: usize = 64; // mismo seq que toda la ronda de hoy, para el mismo corte

fn pack(ctx: &[usize]) -> Vec<u8> {
    ctx.iter().map(|&b| b as u8).collect()
}

fn build_table(train: &[usize], orders: &[usize]) -> Vec<HashMap<Vec<u8>, Vec<(u8, u32)>>> {
    orders.iter().map(|&k| {
        let mut t: HashMap<Vec<u8>, Vec<(u8, u32)>> = HashMap::new();
        if train.len() > k {
            for i in 0..train.len() - k {
                let key = pack(&train[i..i + k]);
                let next = train[i + k] as u8;
                let e = t.entry(key).or_default();
                match e.iter_mut().find(|(b, _)| *b == next) {
                    Some((_, c)) => *c += 1,
                    None => e.push((next, 1)),
                }
            }
        }
        t
    }).collect()
}

/// Igual que `Recall::find`: baja de orden hasta encontrar uno con
/// suficientes apariciones. Devuelve también qué orden contestó, para el
/// reparto de cobertura.
fn lookup<'a>(
    tables: &'a [HashMap<Vec<u8>, Vec<(u8, u32)>>],
    orders: &[usize],
    ctx: &[usize],
) -> Option<(&'a Vec<(u8, u32)>, usize)> {
    for (ti, &k) in orders.iter().enumerate() {
        if ctx.len() < k {
            continue;
        }
        let key = pack(&ctx[ctx.len() - k..]);
        let Some(hits) = tables[ti].get(&key) else { continue };
        let total: u32 = hits.iter().map(|(_, c)| *c).sum();
        if total < MIN_COUNT {
            continue;
        }
        return Some((hits, k));
    }
    None
}

fn distribution(hits: &[(u8, u32)], vocab: usize) -> Vec<f32> {
    let total: u32 = hits.iter().map(|(_, c)| *c).sum();
    let mut p = vec![0.0f32; vocab];
    let inv = 1.0 / total as f32;
    for &(b, c) in hits {
        p[b as usize] = c as f32 * inv;
    }
    p
}

fn shannon_bits(q: &[f32]) -> f32 {
    let nats: f32 = q.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    nats / std::f32::consts::LN_2
}

fn evaluar(nombre: &str, orders: &[usize], tables: &[HashMap<Vec<u8>, Vec<(u8, u32)>>], ids: &[usize], consultas: &[usize], vocab: usize) {
    let mut n_hit = 0usize;
    let mut n_correcto = 0usize;
    let mut suma_h = 0.0f64;
    let mut cobertura: HashMap<usize, usize> = HashMap::new();

    let max_orden = *orders.iter().max().unwrap();
    for &i in consultas {
        let ctx = &ids[i - max_orden..i];
        match lookup(tables, orders, ctx) {
            Some((hits, orden_usado)) => {
                n_hit += 1;
                *cobertura.entry(orden_usado).or_insert(0) += 1;
                let dist = distribution(hits, vocab);
                let argmax = dist.iter().enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (k, &v)| if v > bv { (k, v) } else { (bi, bv) }).0;
                if argmax == ids[i] {
                    n_correcto += 1;
                }
                suma_h += shannon_bits(&dist) as f64;
            }
            None => {
                *cobertura.entry(0).or_insert(0) += 1; // 0 = miss
            }
        }
    }

    let vara = 100.0 * n_correcto as f32 / n_hit.max(1) as f32;
    let h_media = suma_h / n_hit.max(1) as f64;
    println!("\n=== {nombre} (órdenes {orders:?}) ===");
    println!("  cobertura: {n_hit}/{} ({:.1}%) -- resto miss", consultas.len(), 100.0 * n_hit as f32 / consultas.len() as f32);
    println!("  vara (argmax==verdad, sobre hits): {vara:.1}%");
    println!("  H(q) media (bits, sobre hits): {h_media:.3}");
    print!("  reparto de aciertos por orden:");
    let mut ords: Vec<usize> = orders.to_vec();
    ords.push(0);
    for &o in &ords {
        if let Some(&n) = cobertura.get(&o) {
            let etiqueta = if o == 0 { "miss".to_string() } else { o.to_string() };
            print!("  {etiqueta}={n}({:.1}%)", 100.0 * n as f32 / consultas.len() as f32);
        }
    }
    println!();
}

fn main() {
    let data = std::env::args().nth(1).unwrap_or_else(|| "data/prosa250.txt".into());
    let val = 0.1f32;
    let vocab = 256usize;

    let ds = TextDataset::from_file(&data, SEQ).expect("no pude leer el corpus");
    let n_windows = ds.num_windows();
    let n_val = (((n_windows as f32) * val).round() as usize).clamp(1, n_windows / 2);
    let n_train = n_windows - n_val;
    let corte = n_train * SEQ;
    let train: Vec<usize> = ds.ids[..corte].to_vec();

    // Consultas: mismo set fijo para las tres tablas, apto para el máximo
    // orden (32) -- así la comparación es pareja, no cambia la población.
    let max_orden = 32;
    let consultas: Vec<usize> = (corte.max(max_orden)..ds.ids.len()).collect();
    println!("eva orden_largo: {} bytes train | {} consultas de validación (todas con >=32 bytes de contexto)",
        train.len(), consultas.len());

    let base = [8usize, 6, 4, 3, 2];
    let ext16 = [16usize, 8, 6, 4, 3, 2];
    let ext32 = [32usize, 16, 8, 6, 4, 3, 2];

    let t_base = build_table(&train, &base);
    let t_16 = build_table(&train, &ext16);
    let t_32 = build_table(&train, &ext32);

    evaluar("BASE (orden 8, la cadena de siempre)", &base, &t_base, &ds.ids, &consultas, vocab);
    evaluar("EXTENDIDA orden 16", &ext16, &t_16, &ds.ids, &consultas, vocab);
    evaluar("EXTENDIDA orden 32", &ext32, &t_32, &ds.ids, &consultas, vocab);

    println!("\n=== VEREDICTO -- ¿cuello de contexto o de data? ===");
    println!("  Si orden 16/32 casi nunca contesta (cobertura ~0%) y vara/H(q)");
    println!("  globales no se mueven: el corpus es demasiado chico para que un");
    println!("  contexto más largo se repita -- no hay con qué desambiguar más,");
    println!("  aunque en principio existiera la información. Eso es un límite de");
    println!("  DATA (tamaño de corpus), no necesariamente de que 'más contexto no");
    println!("  ayudaría' en un corpus más grande. Si en cambio orden 16/32 SÍ");
    println!("  contesta con cobertura no trivial y ahí la vara sube / H(q) baja");
    println!("  claro respecto a lo que contestaba orden 8 en esas mismas");
    println!("  posiciones, el cuello era de contexto: la tabla estaba tirando");
    println!("  información que sí desambiguaba.");
}
