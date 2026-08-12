//! Cuánta memoria sostiene el grafo de autograd, medido y no estimado.
//!
//! POR QUÉ EXISTE: para retropropagar hay que sostener lo que produjo la
//! pasada hacia adelante, y ESA es la razón por la que entrenar necesita
//! hardware caro -- no el cómputo, la memoria. Antes de intentar bajarla hay
//! que saber cuánta es, separada de los parámetros y del estado del
//! optimizador, que se pagan igual.
//!
//! El pico del proceso (`VmHWM`) no sirve para esto: mezcla todo y además es
//! marca de agua del asignador, así que cuenta fragmentación. Acá se cuentan
//! los bytes que el grafo tiene VIVOS en cada instante.
//!
//! Y UN DETALLE DE NUESTRO CASO que cambia el tamaño del premio: `finalize`
//! **clona las entradas** de cada operación en `saved_v`. O sea que no
//! guardamos activaciones: guardamos **copias** de activaciones. El desglose
//! por operación de acá abajo es lo que dice cuáles clonan de más.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;

static VIVOS: AtomicUsize = AtomicUsize::new(0);
static PICO: AtomicUsize = AtomicUsize::new(0);
static NODOS: AtomicUsize = AtomicUsize::new(0);
static NODOS_PICO: AtomicUsize = AtomicUsize::new(0);

/// Bytes acumulados por operación. No es lo vivo sino el total que pasó por
/// ahí: dice quién es el que más guarda a lo largo del entrenamiento.
static POR_OP: Mutex<Vec<(&'static str, usize, usize)>> = Mutex::new(Vec::new());

/// Un nodo entró al grafo.
pub fn entra(op: &'static str, bytes: usize) {
    let ahora = VIVOS.fetch_add(bytes, Relaxed) + bytes;
    PICO.fetch_max(ahora, Relaxed);
    let n = NODOS.fetch_add(1, Relaxed) + 1;
    NODOS_PICO.fetch_max(n, Relaxed);

    // Se acumula bajo lock porque son pocas operaciones distintas y sólo
    // interesa el total. Si algún día pesa, se hace por hilo.
    let mut t = POR_OP.lock().unwrap();
    match t.iter_mut().find(|(o, _, _)| *o == op) {
        Some((_, b, c)) => {
            *b += bytes;
            *c += 1;
        }
        None => t.push((op, bytes, 1)),
    }
}

/// Un nodo se liberó.
pub fn sale(bytes: usize) {
    VIVOS.fetch_sub(bytes, Relaxed);
    NODOS.fetch_sub(1, Relaxed);
}

pub fn pico_mb() -> f64 {
    PICO.load(Relaxed) as f64 / 1_048_576.0
}

pub fn vivos_mb() -> f64 {
    VIVOS.load(Relaxed) as f64 / 1_048_576.0
}

pub fn nodos_pico() -> usize {
    NODOS_PICO.load(Relaxed)
}

/// Informe: cuánto sostiene el grafo y quién lo sostiene.
static GRADS_PICO: AtomicUsize = AtomicUsize::new(0);

/// Cuánto ocupa el mapa de gradientes en su punto más alto.
///
/// `backward` crea un `Vec` nuevo por CADA tensor que recibe gradiente --no
/// sólo por parámetro, también por cada activación intermedia-- y lo tira al
/// terminar el paso. Si esto es grande, el cuello no es el grafo sino el mapa.
pub fn grads(bytes: usize) {
    GRADS_PICO.fetch_max(bytes, Relaxed);
}

pub fn informe(params: usize) {
    let mut t = POR_OP.lock().unwrap().clone();
    t.sort_by_key(|(_, b, _)| std::cmp::Reverse(*b));
    let total: usize = t.iter().map(|(_, b, _)| *b).sum();

    // El punto de comparación: los parámetros y el estado de AdamW se pagan
    // sí o sí. Lo que el grafo sostiene es lo que se PODRÍA no pagar.
    let pesos = params * 4;
    let adam = params * 8;
    println!("\n── qué sostiene la memoria de entrenamiento ──");
    println!("  parámetros            {:8.1} MB   (inevitable)", pesos as f64 / 1_048_576.0);
    println!("  estado de AdamW       {:8.1} MB   (inevitable hoy)", adam as f64 / 1_048_576.0);
    println!("  GRAFO, pico vivo      {:8.1} MB   <- lo que se puede atacar", pico_mb());
    println!("  nodos vivos, pico     {:8}", nodos_pico());
    println!("  grafo vivo ahora      {:8.1} MB   (0 si se liberó todo)", vivos_mb());
    println!("  MAPA DE GRADIENTES    {:8.1} MB   <- un Vec nuevo por tensor, cada paso",
        GRADS_PICO.load(Relaxed) as f64 / 1_048_576.0);

    println!("\n── quién guarda, acumulado sobre todo el entrenamiento ──");
    for (op, b, c) in t.iter().take(8) {
        println!("  {op:<16} {:9.1} MB en {c:>7} nodos   {:4.0}%",
            *b as f64 / 1_048_576.0, 100.0 * *b as f64 / total.max(1) as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_goes_in_comes_out() {
        // Si esto se descompensa, el contador miente y toda la medición
        // posterior no vale nada.
        let antes = VIVOS.load(Relaxed);
        entra("prueba", 1000);
        assert_eq!(antes + 1000, VIVOS.load(Relaxed));
        sale(1000);
        assert_eq!(antes, VIVOS.load(Relaxed));
    }

    #[test]
    fn the_peak_does_not_go_down() {
        entra("prueba-pico", 5_000_000);
        let p = PICO.load(Relaxed);
        sale(5_000_000);
        assert_eq!(p, PICO.load(Relaxed), "el pico bajó al liberar");
    }
}
