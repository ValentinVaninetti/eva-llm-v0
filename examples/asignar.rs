//! ¿Cuánto cuesta REALMENTE asignar el mapa de gradientes en cada paso?
//!
//! `backward` crea un `Vec` nuevo por cada tensor que recibe gradiente y lo
//! tira al terminar: 40,7 MB por paso. La hipótesis es que reusar esos buffers
//! ahorra tiempo. Pero `vec![0.0; n]` puede estar pidiéndole al sistema páginas
//! YA EN CERO, y entonces asignar sale casi gratis y reusar no ahorra nada.
//!
//! Esto lo decide antes de construir nada: replica exactamente los tamaños que
//! el modelo real pide y los cronometra.
use eva_llm_v0::model::{Arch, EvaConfig, EvaModel};

fn main() {
    let cfg = EvaConfig {
        vocab: 256, dim: 256, ffn_dim: 512, blocks: 16,
        conv_kernel: 5, eps: 1e-5, seq_len: 256, arch: Arch::Clock,
    };
    let m = EvaModel::new(cfg);
    let tamanos: Vec<usize> = m.parameters().iter().map(|p| p.data.len()).collect();
    let total: usize = tamanos.iter().sum();
    println!("  {} buffers, {:.1} MB en total", tamanos.len(), total as f64 * 4.0 / 1_048_576.0);

    // Asignar y tirar, como hace backward en cada paso.
    let mut mejor = f64::INFINITY;
    for _ in 0..50 {
        let t0 = std::time::Instant::now();
        let mapa: Vec<Vec<f32>> = tamanos.iter().map(|&n| vec![0.0f32; n]).collect();
        std::hint::black_box(&mapa);
        drop(mapa);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms < mejor { mejor = ms; }
    }
    println!("  asignar + tirar todo:      {mejor:8.3} ms   (mejor de 50)");

    // Reusar: los buffers ya existen, sólo hay que ponerlos en cero.
    let mut reuso: Vec<Vec<f32>> = tamanos.iter().map(|&n| vec![0.0f32; n]).collect();
    let mut mejor_r = f64::INFINITY;
    for _ in 0..50 {
        let t0 = std::time::Instant::now();
        for v in reuso.iter_mut() {
            v.fill(0.0);
        }
        std::hint::black_box(&reuso);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms < mejor_r { mejor_r = ms; }
    }
    println!("  reusar (sólo poner en cero): {mejor_r:8.3} ms   (mejor de 50)");
    println!("\n  diferencia por paso: {:.3} ms", mejor - mejor_r);
    println!("  el paso completo mide ~350 ms, así que esto es el {:.2}%",
        100.0 * (mejor - mejor_r) / 350.0);
}
