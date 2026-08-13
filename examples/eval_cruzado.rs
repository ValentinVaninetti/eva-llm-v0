//! Corrección de un error propio: `prosa250.txt` es EXACTO el prefijo de
//! los primeros 250.000 bytes de `prosa.txt` (mismo md5). Comparar
//! "VALIDACIÓN FINAL" de una corrida sobre prosa250 contra una corrida
//! sobre prosa.txt completo NO es una comparación válida: cada una define
//! su val como el ÚLTIMO 10% de SU PROPIO archivo, así que caen en tramos
//! de texto distintos -- y peor, el val de prosa250 (bytes ~225k-250k) cae
//! DENTRO del train de la corrida de prosa.txt completo (que sólo reserva
//! val desde ~370k). No hay forma de comparar limpio con los dos
//! checkpoints tal como están sin evaluar los dos sobre el MISMO tramo.
//!
//! Este script evalúa un checkpoint dado sobre el ÚLTIMO val_frac de UN
//! corpus dado -- exactamente lo mismo que hace `train.rs::eval_loss`
//! puertas adentro, pero expuesto para cualquier par checkpoint/corpus.
//! Uso: correr el checkpoint chico (250 KB) sobre el val de `prosa.txt`
//! completo (que le es 100% desconocido, nunca lo entrenó) y comparar ESE
//! número contra el 2.699 bpb que ya reportó la corrida grande sobre su
//! propio val -- ahí sí es el mismo tramo de texto para los dos.

use eva_llm_v0::data::TextDataset;
use eva_llm_v0::save::load_model;
use eva_llm_v0::tensor::ops;

fn main() {
    let weights = std::env::args().nth(1).unwrap_or_else(|| "16m5b_seed7.weights".into());
    let data = std::env::args().nth(2).unwrap_or_else(|| "data/prosa.txt".into());
    let val = 0.1f32;

    let model = load_model(&weights).expect("no pude cargar el checkpoint");
    let seq = model.cfg.seq_len;
    let ds = TextDataset::from_file(&data, seq).expect("no pude leer el corpus");
    let n = ds.num_windows();
    let n_val = (((n as f32) * val).round() as usize).clamp(1, n / 2);
    let n_train = n - n_val;

    let mut suma = 0.0f32;
    for wi in n_train..n {
        let (input, target) = ds.window(wi);
        let logits = model.forward(&input);
        suma += ops::cross_entropy(&logits, &target).data[0];
    }
    let loss = suma / n_val as f32;
    let bpb = loss / std::f32::consts::LN_2;
    println!("eva eval_cruzado: {weights} sobre el val de {data} ({n_val} ventanas)");
    println!("  loss {loss:.4} | {bpb:.3} bits/byte");
}
