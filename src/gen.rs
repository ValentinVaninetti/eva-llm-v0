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
    constraint: &mut dyn Constraint,
) -> Vec<usize> {
    let mut ctx: Vec<usize> = prompt.to_vec();
    let mut out: Vec<usize> = Vec::with_capacity(max_tokens);
    if ctx.is_empty() {
        return out;
    }

    // This used to redo the whole pass over the entire window for EVERY
    // token and discard every row but the last: up to 64x of wasted work.
    // Now the state travels along and each token costs one token.
    let mut st = crate::stream::Streamer::new(model);
    let mut logits = Vec::new();
    for &id in &ctx {
        logits = st.next(id);
    }

    for _ in 0..max_tokens {
        let mask = constraint.allowed(&ctx, logits.len());
        let tok = sample_masked(&logits, &mask, temp, top_k, rng);
        ctx.push(tok);
        out.push(tok);
        logits = st.next(tok);
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
) -> Vec<usize> {
    let mut nc = NoConstraint;
    generate_constrained(model, prompt, max_tokens, temp, top_k, rng, &mut nc)
}

/// What happened at each position, to be able to say how much was saved
/// instead of assuming it.
#[derive(Default, Debug)]
pub struct Counts {
    /// The structure left only one option: the model wasn't consulted.
    pub forced: usize,
    /// Only some columns of the head were computed.
    pub partial: usize,
    /// The model decided freely, with the whole head.
    pub free: usize,
    /// Head columns computed, versus what would have been computed without
    /// the constraint. The honest measure of the savings in the output.
    pub columns: usize,
    pub columns_unrestricted: usize,
}

/// Generates while respecting a constraint **consulted before every step**.
///
/// The difference with `generate_constrained` isn't what it produces but
/// what it spends: there the mask was applied over logits already
/// computed, here where the structure doesn't leave the model a choice it
/// isn't consulted at all.
pub fn generate_shaped(
    model: &EvaModel,
    prompt: &[usize],
    max_tokens: usize,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
    shape: &mut dyn crate::constrain::Constraint,
) -> (Vec<usize>, Counts) {
    use crate::constrain::Allowed;
    let mut c = Counts::default();
    let mut ctx: Vec<usize> = prompt.to_vec();
    let mut out = Vec::with_capacity(max_tokens);
    if ctx.is_empty() {
        return (out, c);
    }

    let vocab = model.cfg.vocab;
    let mut st = crate::stream::Streamer::new(model);
    // The prompt is consumed without a head: nothing comes out of those
    // positions.
    for &id in &ctx[..ctx.len() - 1] {
        st.consume(id);
    }
    let mut last = ctx[ctx.len() - 1];

    for _ in 0..max_tokens {
        c.columns_unrestricted += vocab;
        let tok = match shape.allowed(&ctx) {
            Allowed::Only(v) if v.len() == 1 => {
                // Nothing to decide. Advance the state and that's it.
                st.consume(last);
                c.forced += 1;
                v[0]
            }
            Allowed::Only(v) => {
                let logits = st.next_among(last, &v);
                c.partial += 1;
                c.columns += v.len();
                let i = pick(&logits, temp, top_k, rng);
                v[i]
            }
            Allowed::Any => {
                let logits = st.next(last);
                c.free += 1;
                c.columns += vocab;
                let mask = LogitsMask::all(vocab);
                sample_masked(&logits, &mask, temp, top_k, rng)
            }
        };
        ctx.push(tok);
        out.push(tok);
        last = tok;
    }
    (out, c)
}

/// Picks an index within a handful of already-restricted logits.
fn pick(logits: &[f32], temp: f32, top_k: usize, rng: &mut Rng) -> usize {
    let mask = LogitsMask::all(logits.len());
    sample_masked(logits, &mask, temp, top_k, rng)
}
