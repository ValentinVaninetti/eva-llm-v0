//! Un cronómetro por etiqueta, para saber en qué se va el tiempo.
//!
//! POR QUÉ EXISTE: dos veces en el mismo día la intuición eligió mal el cuello.
//! Primero "hay que optimizar el shader" cuando eran los flags de memoria (el
//! tiling dio 1.13x y el flag 2.5x), después "el cuello es ClockMem" a partir
//! de evidencia indirecta. Adivinar sale caro y medir sale barato.
//!
//! No hay `perf` en esta máquina, así que esto: contadores atómicos por
//! etiqueta, cero costo cuando está apagado, y un informe al final.
//!
//! ```text
//! EVA_PROFILE=1 cargo run --release -- train --data ...
//! ```
//!
//! Mide TIEMPO DE PARED acumulado, así que si algo corre en varios hilos suma
//! más que el reloj. Es a propósito: lo que se busca es en qué se va el
//! esfuerzo, no cuánto tardó la corrida.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Las etiquetas son fijas: un `HashMap` con lock en el camino caliente mide
/// sobre todo el lock.
#[derive(Clone, Copy)]
pub enum P {
    Matmul,
    ClockFwd,
    ClockBwd,
    Backward,
    Optim,
}

impl P {
    const ALL: [(P, &'static str); 5] = [
        (P::Matmul, "matmul"),
        (P::ClockFwd, "clockmem fwd"),
        (P::ClockBwd, "clockmem bwd"),
        (P::Backward, "backward (todo)"),
        (P::Optim, "optimizador"),
    ];
}

static NANOS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static HITS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("EVA_PROFILE").is_ok())
}

/// Mide lo que dure `body`. Con el perfil apagado es una llamada y nada más.
pub fn time<R>(p: P, body: impl FnOnce() -> R) -> R {
    if !on() {
        return body();
    }
    let t0 = Instant::now();
    let r = body();
    let i = p as usize;
    NANOS[i].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    HITS[i].fetch_add(1, Ordering::Relaxed);
    r
}

/// Informe final. `wall` es lo que tardó de verdad, para poner los % en escala.
pub fn report(wall: std::time::Duration) {
    if !on() {
        return;
    }
    let total = wall.as_secs_f64();
    eprintln!("\n── en qué se fue el tiempo ──");
    for (p, name) in P::ALL {
        let i = p as usize;
        let s = NANOS[i].load(Ordering::Relaxed) as f64 / 1e9;
        let n = HITS[i].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        eprintln!(
            "  {name:<16} {s:8.2} s  {:5.1}%  ({n} llamadas, {:.1} µs c/u)",
            100.0 * s / total,
            s * 1e6 / n as f64,
        );
    }
    eprintln!("  {:<16} {total:8.2} s", "reloj de pared");
}
