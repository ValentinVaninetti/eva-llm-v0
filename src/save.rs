use std::collections::HashMap;
use std::io::{Read, Write};

use crate::model::{EvaConfig, EvaModel};

/// v1 y no v0 por un bug de formato, no por un campo nuevo: en v0 `eps` se
/// leía SIN avanzar el cursor, así que `seq_len` releía esos mismos cuatro
/// bytes y volvía 925353388 --el patrón de bits de 1e-5--. Como seq_len sólo
/// recorta la ventana, fallaba en silencio: se generaba muy por fuera del
/// largo con el que se entrenó y nadie se enteraba. Mantener compatibilidad
/// con un formato que se lee mal es peor que romperla.
const MAGIC: &[u8; 6] = b"EVAV1\0";
const MAGIC_V0: &[u8; 6] = b"EVAV0\0";

pub fn save_model(path: &str, model: &EvaModel) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    let cfg = &model.cfg;
    push_u32(&mut buf, cfg.vocab as u32);
    push_u32(&mut buf, cfg.dim as u32);
    push_u32(&mut buf, cfg.ffn_dim as u32);
    push_u32(&mut buf, cfg.blocks as u32);
    push_u32(&mut buf, cfg.conv_kernel as u32);
    buf.extend_from_slice(&cfg.eps.to_le_bytes());
    push_u32(&mut buf, cfg.seq_len as u32);
    push_u32(&mut buf, cfg.arch.as_u32());

    let named = model.named_parameters();
    push_u32(&mut buf, named.len() as u32);
    for (name, t) in named {
        push_str(&mut buf, &name);
        push_u32(&mut buf, t.shape.len() as u32);
        for &d in &t.shape {
            push_u32(&mut buf, d as u32);
        }
        for &x in t.data.iter() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    let mut f = std::fs::File::create(path)?;
    f.write_all(&buf)
}

pub fn load_model(path: &str) -> std::io::Result<EvaModel> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    if buf.len() >= 6 && &buf[0..6] == MAGIC_V0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "checkpoint EVAV0: su seq_len está mal guardado (ver MAGIC). Reentrenar.",
        ));
    }
    if buf.len() < 6 || &buf[0..6] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "archivo no es un modelo eva",
        ));
    }
    let mut pos = 6;
    let cfg = EvaConfig {
        vocab: read_u32(&buf, &mut pos) as usize,
        dim: read_u32(&buf, &mut pos) as usize,
        ffn_dim: read_u32(&buf, &mut pos) as usize,
        blocks: read_u32(&buf, &mut pos) as usize,
        conv_kernel: read_u32(&buf, &mut pos) as usize,
        eps: read_f32(&buf, &mut pos),
        seq_len: read_u32(&buf, &mut pos) as usize,
        arch: crate::model::Arch::from_u32(read_u32(&buf, &mut pos)),
    };

    let n = read_u32(&buf, &mut pos) as usize;
    let mut weights: HashMap<String, Vec<f32>> = HashMap::with_capacity(n);
    for _ in 0..n {
        let name = read_str(&buf, &mut pos);
        let ndim = read_u32(&buf, &mut pos) as usize;
        let mut shape = Vec::with_capacity(ndim);
        let mut numel = 1;
        for _ in 0..ndim {
            let d = read_u32(&buf, &mut pos) as usize;
            shape.push(d);
            numel *= d;
        }
        let mut data = vec![0.0f32; numel];
        for x in data.iter_mut() {
            let b = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
            *x = f32::from_le_bytes(b);
            pos += 4;
        }
        let _ = shape;
        weights.insert(name, data);
    }

    let mut model = EvaModel::new(cfg);
    for (name, t) in model.named_parameters_mut() {
        if let Some(w) = weights.remove(&name) {
            debug_assert_eq!(w.len(), t.data.len(), "shape mismatch en {}", name);
            t.data = std::sync::Arc::new(w);
        }
    }
    Ok(model)
}

pub fn model_info(path: &str) -> std::io::Result<EvaModel> {
    load_model(path)
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    push_u32(buf, b.len() as u32);
    buf.extend_from_slice(b);
}

/// Lee y AVANZA. La versión que no avanzaba fue el bug de v0.
fn read_f32(buf: &[u8], pos: &mut usize) -> f32 {
    let v = f32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos += 4;
    v
}

fn read_u32(buf: &[u8], pos: &mut usize) -> u32 {
    let v = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos += 4;
    v
}

fn read_str(buf: &[u8], pos: &mut usize) -> String {
    let len = read_u32(buf, pos) as usize;
    let s = String::from_utf8_lossy(&buf[*pos..*pos + len]).into_owned();
    *pos += len;
    s
}

pub fn describe(model: &EvaModel) -> String {
    let mut total = 0usize;
    let mut lines = Vec::new();
    for (name, t) in model.named_parameters() {
        let n = t.numel();
        total += n;
        lines.push(format!("  {:40} {:>10} params  shape={:?}", name, n, t.shape));
    }
    format!("modelo EVA v0\n  config: {:?}\n  total: {} params\n{}\n", model.cfg, total, lines.join("\n"))
}
