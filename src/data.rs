use std::fs;

use crate::rng::Rng;

pub struct TextDataset {
    pub ids: Vec<usize>,
    pub seq: usize,
}

impl TextDataset {
    pub fn from_file(path: &str, seq: usize) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        let ids = bytes.iter().map(|&b| b as usize).collect();
        Ok(TextDataset { ids, seq })
    }

    pub fn from_str(text: &str, seq: usize) -> Self {
        let ids = text.as_bytes().iter().map(|&b| b as usize).collect();
        TextDataset { ids, seq }
    }

    pub fn num_windows(&self) -> usize {
        if self.ids.len() < self.seq + 1 {
            0
        } else {
            (self.ids.len() - self.seq - 1) / self.seq + 1
        }
    }

    pub fn window(&self, i: usize) -> (Vec<usize>, Vec<usize>) {
        let start = i * self.seq;
        let input: Vec<usize> = self.ids[start..start + self.seq].to_vec();
        let target: Vec<usize> = self.ids[start + 1..start + self.seq + 1].to_vec();
        (input, target)
    }

    /// Baraja SÓLO las primeras `n`: las de atrás son el examen.
    pub fn shuffled_train_indices(&self, n: usize, rng: &mut Rng) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut idx);
        idx
    }

    pub fn shuffled_indices(&self, rng: &mut Rng) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.num_windows()).collect();
        rng.shuffle(&mut idx);
        idx
    }
}
