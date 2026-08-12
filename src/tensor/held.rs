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

/// El mecanismo, separado de la instancia global.
///
/// POR QUÉ SEPARADO: la primera versión tenía los contadores como estáticos
/// sueltos y el test afirmaba sobre ellos. Los tests de Rust corren en
/// paralelo y casi todos crean grafos, así que cualquier otro test movía el
/// contador entre la lectura y la aserción. **Flaky por construcción**, y
/// apareció una vez en la suite completa antes de que nadie lo buscara.
/// Con el mecanismo aparte, el test usa su propia instancia y no hay carrera.
pub struct Contador {
    vivos: AtomicUsize,
    pico: AtomicUsize,
    nodos: AtomicUsize,
    nodos_pico: AtomicUsize,
    grads_pico: AtomicUsize,
    /// Bytes acumulados por operación: no es lo vivo sino el total que pasó,
    /// para saber quién guarda más a lo largo del entrenamiento.
    por_op: Mutex<Vec<(&'static str, usize, usize)>>,
}

impl Contador {
    pub const fn new() -> Self {
        Contador {
            vivos: AtomicUsize::new(0),
            pico: AtomicUsize::new(0),
            nodos: AtomicUsize::new(0),
            nodos_pico: AtomicUsize::new(0),
            grads_pico: AtomicUsize::new(0),
            por_op: Mutex::new(Vec::new()),
        }
    }

    pub fn entra(&self, op: &'static str, bytes: usize) {
        let ahora = self.vivos.fetch_add(bytes, Relaxed) + bytes;
        self.pico.fetch_max(ahora, Relaxed);
        let n = self.nodos.fetch_add(1, Relaxed) + 1;
        self.nodos_pico.fetch_max(n, Relaxed);

        let mut t = self.por_op.lock().unwrap();
        match t.iter_mut().find(|(o, _, _)| *o == op) {
            Some((_, b, c)) => {
                *b += bytes;
                *c += 1;
            }
            None => t.push((op, bytes, 1)),
        }
    }

    pub fn sale(&self, bytes: usize) {
        self.vivos.fetch_sub(bytes, Relaxed);
        self.nodos.fetch_sub(1, Relaxed);
    }

    pub fn vivos(&self) -> usize {
        self.vivos.load(Relaxed)
    }

    pub fn pico(&self) -> usize {
        self.pico.load(Relaxed)
    }
}

static G: Contador = Contador::new();

/// Un nodo entró al grafo.
pub fn entra(op: &'static str, bytes: usize) {
    G.entra(op, bytes);
}

/// Un nodo se liberó.
pub fn sale(bytes: usize) {
    G.sale(bytes);
}

pub fn pico_mb() -> f64 {
    G.pico() as f64 / 1_048_576.0
}

pub fn vivos_mb() -> f64 {
    G.vivos() as f64 / 1_048_576.0
}

pub fn nodos_pico() -> usize {
    G.nodos_pico.load(Relaxed)
}

/// Cuánto ocupa el mapa de gradientes en su punto más alto.
///
/// `backward` crea un `Vec` nuevo por CADA tensor que recibe gradiente --no
/// sólo por parámetro-- y lo tira al terminar el paso.
pub fn grads(bytes: usize) {
    G.grads_pico.fetch_max(bytes, Relaxed);
}

/// Informe: cuánto sostiene el grafo y quién lo sostiene.
pub fn informe(params: usize) {
    let mut t = G.por_op.lock().unwrap().clone();
    t.sort_by_key(|(_, b, _)| std::cmp::Reverse(*b));
    let total: usize = t.iter().map(|(_, b, _)| *b).sum();

    let pesos = params * 4;
    let adam = params * 8;
    println!("\n── qué sostiene la memoria de entrenamiento ──");
    println!("  parámetros            {:8.1} MB   (inevitable)", pesos as f64 / 1_048_576.0);
    println!("  estado de AdamW       {:8.1} MB   (inevitable hoy)", adam as f64 / 1_048_576.0);
    println!("  GRAFO, pico propio    {:8.1} MB", pico_mb());
    println!("  nodos vivos, pico     {:8}", nodos_pico());
    println!("  grafo vivo ahora      {:8.1} MB   (0 si se liberó todo)", vivos_mb());
    println!("  MAPA DE GRADIENTES    {:8.1} MB   <- un Vec nuevo por tensor, cada paso",
        G.grads_pico.load(Relaxed) as f64 / 1_048_576.0);

    println!("\n── quién guarda, acumulado sobre todo el entrenamiento ──");
    for (op, b, c) in t.iter().take(8) {
        println!("  {op:<16} {:9.1} MB en {c:>7} nodos   {:4.0}%",
            *b as f64 / 1_048_576.0, 100.0 * *b as f64 / total.max(1) as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sobre una instancia PROPIA, no sobre la global: la versión anterior
    /// afirmaba sobre el contador compartido mientras el resto de la suite
    /// creaba grafos en paralelo. Era flaky por construcción y se vio una vez.
    #[test]
    fn what_goes_in_comes_out() {
        let c = Contador::new();
        c.entra("prueba", 1000);
        assert_eq!(1000, c.vivos());
        c.sale(1000);
        assert_eq!(0, c.vivos(), "el contador quedó descompensado");
    }

    #[test]
    fn the_peak_does_not_go_down() {
        let c = Contador::new();
        c.entra("prueba", 5_000_000);
        c.sale(5_000_000);
        assert_eq!(5_000_000, c.pico(), "el pico bajó al liberar");
        assert_eq!(0, c.vivos());
    }

    /// Y la propiedad que el test viejo NO probaba: que aguante concurrencia.
    /// El contador de producción lo tocan varios hilos si algún día se crean
    /// nodos en paralelo, y una suma perdida arruinaría toda la medición.
    #[test]
    fn it_survives_many_threads() {
        let c = std::sync::Arc::new(Contador::new());
        let mut hilos = Vec::new();
        for _ in 0..8 {
            let c = c.clone();
            hilos.push(std::thread::spawn(move || {
                for _ in 0..2000 {
                    c.entra("concurrente", 64);
                    c.sale(64);
                }
            }));
        }
        for h in hilos {
            h.join().unwrap();
        }
        assert_eq!(0, c.vivos(), "se perdieron sumas o restas entre hilos");
        assert!(c.pico() >= 64);
    }
}
