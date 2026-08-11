use crate::model::EvaModel;
use crate::rng::Rng;

pub struct LogitsMask {
    allowed: Vec<bool>,
}

impl LogitsMask {
    pub fn all(vocab: usize) -> Self {
        LogitsMask { allowed: vec![true; vocab] }
    }

    pub fn disallow(&mut self, tok: usize) {
        if let Some(a) = self.allowed.get_mut(tok) {
            *a = false;
        }
    }

    pub fn allow_only(&mut self, toks: &[usize]) {
        self.allowed.fill(false);
        for &t in toks {
            if let Some(a) = self.allowed.get_mut(t) {
                *a = true;
            }
        }
    }

    pub fn is_allowed(&self, tok: usize) -> bool {
        self.allowed.get(tok).copied().unwrap_or(false)
    }
}

pub trait Constraint {
    fn allowed(&mut self, so_far: &[usize], vocab: usize) -> LogitsMask;
}

pub struct NoConstraint;

impl Constraint for NoConstraint {
    fn allowed(&mut self, _so_far: &[usize], vocab: usize) -> LogitsMask {
        LogitsMask::all(vocab)
    }
}

pub fn sample(logits: &[f32], temp: f32, top_k: usize, rng: &mut Rng) -> usize {
    sample_masked(logits, &LogitsMask::all(logits.len()), temp, top_k, rng)
}

pub fn sample_masked(
    logits: &[f32],
    mask: &LogitsMask,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
) -> usize {
    let v = logits.len();
    let mut cand: Vec<(f32, usize)> = (0..v)
        .filter(|&i| mask.is_allowed(i))
        .map(|i| (logits[i], i))
        .collect();
    if cand.is_empty() {
        cand = (0..v).map(|i| (logits[i], i)).collect();
    }
    let k = top_k.min(cand.len());
    let n = if k == cand.len() {
        cand.len()
    } else {
        cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        k
    };
    let mut max = f32::NEG_INFINITY;
    for &(val, _) in cand.iter().take(n) {
        if val > max {
            max = val;
        }
    }
    let mut sum = 0.0;
    let mut probs = Vec::with_capacity(n);
    for &(val, idx) in cand.iter().take(n) {
        let p = ((val - max) / temp).exp();
        probs.push((p, idx));
        sum += p;
    }
    let mut r = rng.next_f32() * sum;
    for (p, idx) in probs {
        r -= p;
        if r <= 0.0 {
            return idx;
        }
    }
    cand[0].1
}

pub fn generate_constrained(
    model: &EvaModel,
    prompt: &[usize],
    max_tokens: usize,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
    seq: usize,
    constraint: &mut dyn Constraint,
) -> Vec<usize> {
    let mut ctx: Vec<usize> = prompt.to_vec();
    let mut out: Vec<usize> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let start = ctx.len().saturating_sub(seq);
        let window = &ctx[start..];
        let logits = model.forward(window);
        let (s, v) = (logits.shape[0], logits.shape[1]);
        let mask = constraint.allowed(&ctx, v);
        let last = &logits.data[(s - 1) * v..s * v];
        let tok = sample_masked(last, &mask, temp, top_k, rng);
        ctx.push(tok);
        out.push(tok);
    }
    out
}

pub fn generate(
    model: &EvaModel,
    prompt: &[usize],
    max_tokens: usize,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
    seq: usize,
) -> Vec<usize> {
    let mut nc = NoConstraint;
    generate_constrained(model, prompt, max_tokens, temp, top_k, rng, seq, &mut nc)
}
