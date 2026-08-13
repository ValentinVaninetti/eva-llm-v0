//! ¿El count del contexto se pierde en `lookup()`? Lo que dejó anotado Claude
//! (recall.rs:117): `hits.iter()` suma todo antes de normalizar, y el caller
//! sólo recibe la distribución normalizada -- sin el count ni el orden que
//! contestó.
//!
//! Prueba con datos: dos contextos sintéticos, uno visto 500 veces y otro
//! exactamente 2 (el piso de MIN_COUNT), ambos con una única continuación.
//! Si el count se descartara, el caller vería la MISMA distribución (certeza
//! 1.0) y no podría distinguir la memoria sólida de la conjetura de 2 muestras.
//!
//! Sólo usa la API pública (`build`, `lookup`): no toca `src/`, no carga
//! modelo, no usa GPU -- cero colisión con el barrido de Dante.

use eva_llm_v0::recall::Recall;

fn main() {
    let mut bytes: Vec<usize> = Vec::new();
    let a: [usize; 4] = [1, 2, 3, 4];
    let b: [usize; 4] = [7, 8, 9, 10];
    for _ in 0..500 {
        bytes.extend_from_slice(&a);
        bytes.push(9);
    }
    for _ in 0..2 {
        bytes.extend_from_slice(&b);
        bytes.push(11);
    }

    let tabla = Recall::build(&bytes);
    println!("eva count_probe: tabla construida sobre {} bytes", bytes.len());

    let q_a = tabla.lookup(&a, 256).expect("A debería ser un hit (500 apariciones)");
    let q_b = tabla.lookup(&b, 256).expect("B debería ser un hit (2 apariciones, el piso)");

    println!("contexto A (visto 500 veces):   p(9)  = {:.3}", q_a[9]);
    println!("contexto B (visto 2 veces):     p(11) = {:.3}", q_b[11]);

    if (q_a[9] - q_b[11]).abs() < 1e-6 {
        println!("\n=== VERIFICADO ===");
        println!("Misma certeza (1.0) para 500 muestras y 2 muestras. El count se");
        println!("pierde en lookup(): el caller no puede distinguir memoria sólida de");
        println!("una conjetura de 2 muestras -- y un λ por count necesita el count.");
    } else {
        println!("\n=== NO se perdió: los counts se distinguen ===");
    }
}
