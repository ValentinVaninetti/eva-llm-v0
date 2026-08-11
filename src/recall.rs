//! Conocimiento en una tabla, no en los pesos.
//!
//! ESTO EXISTE PARA DECIDIR LA HIPÓTESIS CENTRAL DEL PROYECTO:
//!
//! > ¿Se puede separar lo que necesita **generalizar** de lo que sólo necesita
//! > **recordarse**, sin perder la capacidad de razonar sobre lo recordado?
//!
//! Si la respuesta es sí, el costo de las LLM se derrumba: los hechos son
//! muchísima información y no necesitan representación distribuida; la
//! estructura del lenguaje es poca y sí la necesita. Hoy se paga precio de
//! generalización para guardar cosas que sólo hay que recordar.
//!
//! LA FORMA MÁS PURA DE PROBARLO es la más tonta: una tabla de k-gramas
//! exactos del texto de entrenamiento. Cero parámetros, cero entrenamiento, es
//! literalmente una tabla de consulta. Si **la mitad del modelo más la tabla**
//! alcanza al modelo entero, la hipótesis se sostiene. Si no, el conocimiento
//! en los pesos estaba haciendo algo que una tabla no reemplaza -- y eso
//! también hay que saberlo.
//!
//! NO SE AJUSTA NADA CONTRA VALIDACIÓN. El peso de mezcla se elige sobre una
//! porción de desarrollo separada del entrenamiento; validación se toca una
//! sola vez, al final. Ajustar la mezcla mirando el examen daría un número
//! lindo y falso.

use std::collections::HashMap;

/// Órdenes de k-grama, del más específico al más general.
///
/// Tope de 8 porque así el contexto entra exacto en un `u64` y **no hay
/// colisiones de hash**: la clave *es* el contexto. Con un hash de por medio,
/// dos contextos distintos podrían compartir entrada y el experimento mediría
/// las colisiones en vez de la hipótesis.
const ORDERS: [usize; 5] = [8, 6, 4, 3, 2];

/// Cuántas veces tiene que haberse visto un contexto para creerle.
///
/// Con una sola aparición la "distribución" es un único byte con probabilidad
/// 1, que es memorización pura y engaña: sube en entrenamiento y no generaliza.
const MIN_COUNT: u32 = 2;

pub struct Recall {
    /// Una tabla por orden: contexto empaquetado -> (byte siguiente, veces).
    tables: Vec<HashMap<u64, Vec<(u8, u32)>>>,
    /// Orden mínimo aceptado. Con 2 la tabla contesta casi siempre, pero un
    /// bigrama NO es conocimiento: es estadística genérica del idioma. Subirlo
    /// deja sólo los aciertos específicos, y sirve para separar "recuperar
    /// algo puntual" de "suavizar con n-gramas", que son cosas distintas y dan
    /// el mismo número si no se miran por separado.
    min_order: usize,
    /// Cuántas veces contestó cada orden, para poder mirar de dónde viene la
    /// mejora en vez de suponerlo.
    pub hits: std::cell::RefCell<[usize; ORDERS.len()]>,
}

fn pack(ctx: &[usize]) -> u64 {
    ctx.iter().fold(0u64, |acc, &b| (acc << 8) | (b as u64 & 0xff))
}

impl Recall {
    /// Construye la tabla **sólo con los bytes de entrenamiento**. Que acá
    /// entre un byte de validación invalida todo el experimento.
    pub fn build(train: &[usize]) -> Self {
        let mut tables = Vec::with_capacity(ORDERS.len());
        for &k in &ORDERS {
            let mut t: HashMap<u64, Vec<(u8, u32)>> = HashMap::new();
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
            tables.push(t);
        }
        let min_order = std::env::var("EVA_MIN_ORDER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        Recall { tables, min_order, hits: std::cell::RefCell::new([0; ORDERS.len()]) }
    }

    /// Reparto de aciertos por orden de k-grama, en porcentaje.
    pub fn hit_profile(&self) -> Vec<(usize, f32)> {
        let h = self.hits.borrow();
        let total: usize = h.iter().sum();
        ORDERS
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, 100.0 * h[i] as f32 / total.max(1) as f32))
            .collect()
    }

    /// Distribución del byte siguiente según la tabla, o `None` si no vio este
    /// contexto lo suficiente.
    ///
    /// Baja de orden hasta encontrar algo: primero pregunta por los últimos 8
    /// bytes, después 6, y así. Contexto más largo es más específico y más
    /// confiable; el retroceso es lo que evita quedarse mudo casi siempre.
    pub fn lookup(&self, ctx: &[usize], vocab: usize) -> Option<Vec<f32>> {
        for (ti, &k) in ORDERS.iter().enumerate() {
            if k < self.min_order || ctx.len() < k {
                continue;
            }
            let key = pack(&ctx[ctx.len() - k..]);
            let Some(hits) = self.tables[ti].get(&key) else { continue };
            let total: u32 = hits.iter().map(|(_, c)| *c).sum();
            if total < MIN_COUNT {
                continue;
            }
            self.hits.borrow_mut()[ti] += 1;
            let mut p = vec![0.0f32; vocab];
            let inv = 1.0 / total as f32;
            for &(b, c) in hits {
                p[b as usize] = c as f32 * inv;
            }
            return Some(p);
        }
        None
    }

    /// Cuántas entradas tiene, para poder contar lo que "pesa" la tabla contra
    /// lo que pesan los parámetros que reemplaza.
    pub fn entries(&self) -> usize {
        self.tables.iter().map(|t| t.len()).sum()
    }

    /// Bytes aproximados: clave de 8 + cada continuación de 5.
    pub fn bytes(&self) -> usize {
        self.tables
            .iter()
            .map(|t| t.iter().map(|(_, v)| 8 + v.len() * 5).sum::<usize>())
            .sum()
    }
}

/// Mezcla la predicción del modelo con la de la tabla y devuelve la pérdida
/// media en nats.
///
/// `lambda = 0` es el modelo solo, que es la línea base contra la que se
/// compara todo lo demás.
pub fn mixed_loss(
    logits: &[f32],
    vocab: usize,
    ctx_before: &[usize],
    targets: &[usize],
    table: &Recall,
    lambda: f32,
    cobertura: &mut (usize, usize),
) -> f32 {
    let mut total = 0.0;
    for (t, &tgt) in targets.iter().enumerate() {
        let row = &logits[t * vocab..(t + 1) * vocab];
        // softmax estable
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        let mut p = vec![0.0f32; vocab];
        for (i, &v) in row.iter().enumerate() {
            let e = (v - mx).exp();
            p[i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for v in p.iter_mut() {
            *v *= inv;
        }

        // El contexto de la tabla son los bytes REALES anteriores a esta
        // posición, que es lo que tendría un generador en ese punto.
        let hasta = ctx_before.len() + t;
        let ctx: Vec<usize> = if t == 0 {
            ctx_before.to_vec()
        } else {
            let mut c = ctx_before.to_vec();
            c.extend_from_slice(&targets[..t]);
            c
        };
        let _ = hasta;

        // Cobertura: cuántas veces la tabla tuvo algo que decir. Si es casi
        // siempre, el corpus de prueba se parece demasiado al guardado y el
        // resultado no se sostendría con texto nuevo.
        cobertura.1 += 1;
        if let Some(q) = table.lookup(&ctx, vocab) {
            cobertura.0 += 1;
            if lambda > 0.0 {
                for (pi, qi) in p.iter_mut().zip(&q) {
                    *pi = (1.0 - lambda) * *pi + lambda * qi;
                }
            }
        }
        // Piso para que un cero de la mezcla no dé infinito.
        total -= p[tgt].max(1e-9).ln();
    }
    total / targets.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_remembers_what_it_saw() {
        // "abcabcabc...": después de "abc" siempre viene 'a'.
        let train: Vec<usize> = "abcabcabcabcabcabcabcabc".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let ctx: Vec<usize> = "abcabcab".bytes().map(|b| b as usize).collect();
        let p = r.lookup(&ctx, 256).expect("debería reconocer el contexto");
        assert!(p['c' as usize] > 0.9, "esperaba 'c' y dio {:?}", p['c' as usize]);
    }

    #[test]
    fn it_stays_quiet_about_what_it_never_saw() {
        // Una tabla que inventa es peor que una que calla: la mezcla se lleva
        // la mentira a la predicción final.
        let train: Vec<usize> = "aaaaaaaaaaaaaaaa".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let ctx: Vec<usize> = "zzzzzzzz".bytes().map(|b| b as usize).collect();
        assert!(r.lookup(&ctx, 256).is_none());
    }

    #[test]
    fn one_sighting_is_not_knowledge() {
        // Un contexto visto una sola vez daría probabilidad 1 a un byte: es
        // memorización, no distribución. MIN_COUNT lo filtra.
        let train: Vec<usize> = "qwertyuiopasdfgh".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let ctx: Vec<usize> = "qwertyui".bytes().map(|b| b as usize).collect();
        assert!(r.lookup(&ctx, 256).is_none(), "le creyó a una sola aparición");
    }

    #[test]
    fn lambda_zero_is_exactly_the_model_alone() {
        // La línea base tiene que salir del MISMO código que la mezcla, o se
        // estarían comparando dos implementaciones distintas de softmax.
        let train: Vec<usize> = "abcabcabcabc".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let vocab = 4;
        let logits = vec![0.1, 2.0, -1.0, 0.5, 1.0, 0.0, 0.0, 0.0];
        let targets = vec![1usize, 0];
        let mut cob = (0, 0);
        let con = mixed_loss(&logits, vocab, &[], &targets, &r, 0.0, &mut cob);

        // A mano: -log softmax en la posición del objetivo.
        let mut esperado = 0.0;
        for (t, &tgt) in targets.iter().enumerate() {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            esperado -= ((row[tgt] - mx).exp() / s).ln();
        }
        esperado /= targets.len() as f32;
        assert!((con - esperado).abs() < 1e-5, "{con} != {esperado}");
    }
}
