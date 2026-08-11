use std::collections::HashMap;

use crate::tensor::Tensor;

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    t: usize,
    state: HashMap<usize, (Vec<f32>, Vec<f32>)>,
}

impl AdamW {
    pub fn new(lr: f32, wd: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd,
            t: 0,
            state: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &HashMap<usize, Vec<f32>>) {
        self.t += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let t = self.t;
        let mhat_correction = 1.0 - b1.powi(t as i32);
        let vhat_correction = 1.0 - b2.powi(t as i32);
        for p in params.iter_mut() {
            let g = match grads.get(&p.id) {
                Some(g) => g,
                None => continue,
            };
            debug_assert_eq!(g.len(), p.data.len(), "grad shape mismatch for param {}", p.id);
            let (m, v) = self
                .state
                .entry(p.id)
                .or_insert_with(|| (vec![0.0; p.data.len()], vec![0.0; p.data.len()]));
            for i in 0..p.data.len() {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            }
            for i in 0..p.data.len() {
                let mhat = m[i] / mhat_correction;
                let vhat = v[i] / vhat_correction;
                let mut update = mhat / (vhat.sqrt() + self.eps);
                update += self.wd * p.data[i];
                p.data[i] -= self.lr * update;
            }
        }
    }
}
