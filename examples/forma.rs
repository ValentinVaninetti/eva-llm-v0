//! Cuánto trabajo se ahorra cuando la estructura ya decidió.
//!
//! Genera la MISMA salida con y sin restricción y compara. El molde imita la
//! forma que tiene la salida útil de un sistema real: un envoltorio fijo con
//! huecos donde el modelo decide de verdad.
use eva_llm_v0::constrain::{Free, Skeleton};
use eva_llm_v0::gen::generate_shaped;
use eva_llm_v0::rng::Rng;
use eva_llm_v0::save::load_model;

fn main() {
    let ruta = std::env::args().nth(1).expect("uso: forma <pesos>");
    let model = load_model(&ruta).expect("cargar el modelo");
    let prompt: Vec<usize> = "La ".bytes().map(|b| b as usize).collect();
    // El largo del molde, no 300: si se generan cientos de bytes más allá del
    // molde, el resto va libre y la proporción medida no dice nada de lo que
    // pasa cuando la salida ES la cosa estructurada.
    let n = 100;

    let corrida = |forma: &mut dyn eva_llm_v0::constrain::Constraint, etq: &str| {
        // CALENTAR ANTES DE MEDIR. Sin esto la primera condición paga la
        // memoria fría y la segunda cobra la ventaja: la primera medición dio
        // 18% de ahorro donde el mecanismo sólo explica 0,2%, y ese
        // desajuste era el error, no el resultado.
        let mut rng = Rng::new(7);
        let _ = generate_shaped(&model, &prompt, n, 0.8, 40, &mut rng, forma);
        // MEJOR DE 20. Con una sola corrida esto dio +18% y después -75% para
        // el mismo mecanismo: la varianza de la máquina es mayor que el efecto.
        let mut mejor = f64::INFINITY;
        let mut c = eva_llm_v0::gen::Cuentas::default();
        for _ in 0..20 {
            let mut rng = Rng::new(7);
            let t0 = std::time::Instant::now();
            let (_, cc) = generate_shaped(&model, &prompt, n, 0.8, 40, &mut rng, forma);
            let t = t0.elapsed().as_secs_f64() * 1e3;
            if t < mejor { mejor = t; c = cc; }
        }
        let ms = mejor;
        println!(
            "  {etq:<12} {ms:7.1} ms | forzadas {:3} parciales {:3} libres {:3} \
             | columnas {} de {} ({:.0}% menos)",
            c.forzadas, c.parciales, c.libres, c.columnas, c.columnas_sin_restriccion,
            100.0 * (1.0 - c.columnas as f64 / c.columnas_sin_restriccion as f64),
        );
        ms
    };

    println!("generando {n} bytes con el mismo modelo:");
    let libre = corrida(&mut Free, "libre");

    // Un molde tipo respuesta estructurada: envoltorio fijo, dos huecos.
    // Huecos chicos, como los de verdad: lo que el modelo elige es poco y el
    // envoltorio es mucho.
    let mut molde = Skeleton::new(
        &[r#"{"decir":""#, r#"","citando":["#, "]}"],
        &[(70, b'"'), (3, b']')],
    );
    let con = corrida(&mut molde, "con molde");

    println!("\n  diferencia de tiempo: {:.1}%", 100.0 * (1.0 - con / libre));

    // Cuánto pesa la cabeza DE VERDAD en este modelo, medido y no supuesto.
    // Sin esto, cualquier extrapolación a vocabularios grandes es aire.
    let d = model.cfg.dim;
    let v = model.cfg.vocab;
    let x = vec![0.5f32; d];
    let mut out = vec![0.0f32; v];
    let mut cabeza = f64::INFINITY;
    for _ in 0..2000 {
        let t0 = std::time::Instant::now();
        eva_llm_v0::math::matmul(&x, &model.head_w.data, 1, d, v, &mut out);
        let t = t0.elapsed().as_secs_f64() * 1e3;
        if t < cabeza { cabeza = t; }
    }
    let por_token = libre / n as f64;
    println!("\n  cabeza sola: {:.4} ms | paso completo: {:.4} ms | la cabeza es el {:.1}%",
        cabeza, por_token, 100.0 * cabeza / por_token);
    println!("  con vocabulario de 32000 la cabeza costaría {:.1}x más y pasaría a dominar",
        32000.0 / v as f64);
}
