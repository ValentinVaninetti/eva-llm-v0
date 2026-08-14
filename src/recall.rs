//! Knowledge in a table, not in the weights.
//!
//! THIS EXISTS TO DECIDE THE PROJECT'S CENTRAL HYPOTHESIS:
//!
//! > Can what needs to **generalize** be separated from what only needs to
//! > **be remembered**, without losing the ability to reason over what was
//! > remembered?
//!
//! If the answer is yes, the cost of LLMs collapses: facts are a huge
//! amount of information and don't need distributed representation; the
//! structure of language is little and does need it. Today the cost of
//! generalization is being paid to store things that only need remembering.
//!
//! THE PUREST WAY TO TEST IT is the dumbest one: a table of exact k-grams
//! from the training text. Zero parameters, zero training, it's literally a
//! lookup table. If **half the model plus the table** matches the full
//! model, the hypothesis holds. If not, the knowledge in the weights was
//! doing something a table doesn't replace -- and that's worth knowing too.
//!
//! NOTHING IS TUNED AGAINST VALIDATION. The mixing weight is chosen on a
//! development slice separate from training; validation is touched exactly
//! once, at the end. Tuning the mix while looking at the exam would give a
//! nice-looking, fake number.

use std::collections::HashMap;

/// K-gram orders, from most specific to most general.
///
/// Capped at 8 because that way the context fits exactly in a `u64` and
/// **there are no hash collisions**: the key *is* the context. With a hash
/// in between, two different contexts could share an entry and the
/// experiment would be measuring collisions instead of the hypothesis.
const ORDERS: [usize; 5] = [8, 6, 4, 3, 2];

/// How many times a context has to have been seen before it's trusted.
///
/// With a single occurrence the "distribution" is a single byte with
/// probability 1, which is pure memorization and is misleading: it goes up
/// during training and doesn't generalize.
const MIN_COUNT: u32 = 2;

pub struct Recall {
    /// One table per order: packed context -> (next byte, count).
    tables: Vec<HashMap<u64, Vec<(u8, u32)>>>,
    /// Minimum order accepted. At 2 the table answers almost always, but a
    /// bigram is NOT knowledge: it's generic language statistics. Raising it
    /// keeps only the specific hits, and lets us separate "retrieving
    /// something specific" from "smoothing with n-grams", which are
    /// different things that give the same number if not looked at
    /// separately.
    min_order: usize,
    /// How many times each order answered, to see where the improvement
    /// comes from instead of assuming it.
    pub hits: std::cell::RefCell<[usize; ORDERS.len()]>,
}

fn pack(ctx: &[usize]) -> u64 {
    ctx.iter().fold(0u64, |acc, &b| (acc << 8) | (b as u64 & 0xff))
}

/// Normalizes a context's continuations into a distribution.
fn distribution(hits: &[(u8, u32)], vocab: usize) -> Vec<f32> {
    let total: u32 = hits.iter().map(|(_, c)| *c).sum();
    let mut p = vec![0.0f32; vocab];
    let inv = 1.0 / total as f32;
    for &(b, c) in hits {
        p[b as usize] = c as f32 * inv;
    }
    p
}

impl Recall {
    /// Builds the table **only from the training bytes**. Letting a
    /// validation byte in here invalidates the whole experiment.
    pub fn build(train: &[usize]) -> Self {
        let mut tables = Vec::with_capacity(ORDERS.len());
        for &k in &ORDERS {
            let mut t: HashMap<u64, Vec<(u8, u32)>> = HashMap::new();
            if train.len() > k {
                for i in 0..train.len() - k {
                    let key = pack(&train[i..i + k]);
                    let next = train[i + k] as u8;
                    let e = t.entry(key).or_default();
                    match e.iter_mut().find(|(b, _)| *b == next) {
                        Some((_, c)) => *c += 1,
                        None => e.push((next, 1)),
                    }
                }
            }
            tables.push(t);
        }
        let min_order = std::env::var("EVA_MIN_ORDER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        Recall { tables, min_order, hits: std::cell::RefCell::new([0; ORDERS.len()]) }
    }

    /// Breakdown of hits by k-gram order, in percent.
    pub fn hit_profile(&self) -> Vec<(usize, f32)> {
        let h = self.hits.borrow();
        let total: usize = h.iter().sum();
        ORDERS
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, 100.0 * h[i] as f32 / total.max(1) as f32))
            .collect()
    }

    /// Distribution of the next byte according to the table, or `None` if
    /// this context wasn't seen enough.
    ///
    /// Falls back to lower orders until it finds something: first it asks
    /// about the last 8 bytes, then 6, and so on. A longer context is more
    /// specific and more reliable; the fallback is what keeps it from
    /// staying silent almost all the time.
    pub fn lookup(&self, ctx: &[usize], vocab: usize) -> Option<Vec<f32>> {
        let hits = self.find(ctx)?;
        Some(distribution(hits, vocab))
    }

    /// Same as `lookup`, but also exposes HOW MANY times the context that
    /// answered was seen (`total`). It's the variable `lookup` throws away:
    /// a context seen 2 times and one seen 500 times return the same
    /// distribution (certainty 1.0) and the caller can't tell them apart.
    /// A per-count lambda needs the count, and here it is.
    pub fn lookup_detail(&self, ctx: &[usize], vocab: usize) -> Option<(Vec<f32>, u32)> {
        let hits = self.find(ctx)?;
        let total: u32 = hits.iter().map(|(_, c)| *c).sum();
        Some((distribution(hits, vocab), total))
    }

    /// The context that answers: falls back to lower orders until it finds
    /// one with enough occurrences, and counts the hit in `hits` for the
    /// profile.
    fn find(&self, ctx: &[usize]) -> Option<&Vec<(u8, u32)>> {
        for (ti, &k) in ORDERS.iter().enumerate() {
            if k < self.min_order || ctx.len() < k {
                continue;
            }
            let key = pack(&ctx[ctx.len() - k..]);
            let Some(hits) = self.tables[ti].get(&key) else { continue };
            let total: u32 = hits.iter().map(|(_, c)| *c).sum();
            if total < MIN_COUNT {
                continue;
            }
            self.hits.borrow_mut()[ti] += 1;
            return Some(hits);
        }
        None
    }

    /// Adds ONE usage observation, with the same counting discipline as
    /// `build()`: it doesn't replace anything, doesn't mark "this is THE
    /// answer" -- it's one more vote for `real_byte` in the given context,
    /// across EVERY order that context reaches into (just like a real
    /// training position would have done). If the context is ambiguous,
    /// it's an honest vote among several; if the fact repeats, `MIN_COUNT`
    /// and the distribution consolidate it on their own -- the teacher who
    /// marks it, not the one who shouts.
    ///
    /// Precedent: online cache from kNN-LM / cache models (Grave et al.,
    /// "Improving Neural Language Models with a Continuous Cache") -- an
    /// external memory that extends with what it keeps seeing, without
    /// touching the weights.
    pub fn observe(&mut self, ctx: &[usize], real_byte: u8) {
        for (ti, &k) in ORDERS.iter().enumerate() {
            if ctx.len() < k {
                continue;
            }
            let key = pack(&ctx[ctx.len() - k..]);
            let e = self.tables[ti].entry(key).or_default();
            match e.iter_mut().find(|(b, _)| *b == real_byte) {
                Some((_, c)) => *c += 1,
                None => e.push((real_byte, 1)),
            }
        }
    }

    /// How many entries it has, to count what the table "weighs" against
    /// what the parameters it replaces weigh.
    pub fn entries(&self) -> usize {
        self.tables.iter().map(|t| t.len()).sum()
    }

    /// Approximate bytes: 8-byte key + 5 per continuation.
    pub fn bytes(&self) -> usize {
        self.tables
            .iter()
            .map(|t| t.iter().map(|(_, v)| 8 + v.len() * 5).sum::<usize>())
            .sum()
    }
}

/// Mixes the model's prediction with the table's and returns the mean loss
/// in nats.
///
/// `lambda = 0` is the model alone, which is the baseline everything else
/// gets compared against.
pub fn mixed_loss(
    logits: &[f32],
    vocab: usize,
    ctx_before: &[usize],
    targets: &[usize],
    table: &Recall,
    lambda: f32,
    coverage: &mut (usize, usize),
) -> f32 {
    let mut total = 0.0;
    for (t, &tgt) in targets.iter().enumerate() {
        let row = &logits[t * vocab..(t + 1) * vocab];
        // stable softmax
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        let mut p = vec![0.0f32; vocab];
        for (i, &v) in row.iter().enumerate() {
            let e = (v - mx).exp();
            p[i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for v in p.iter_mut() {
            *v *= inv;
        }

        // The table's context is the REAL bytes before this position, which
        // is what a generator would have at that point.
        let ctx: Vec<usize> = if t == 0 {
            ctx_before.to_vec()
        } else {
            let mut c = ctx_before.to_vec();
            c.extend_from_slice(&targets[..t]);
            c
        };

        // Coverage: how many times the table had something to say. If it's
        // almost always, the test corpus is too similar to what was stored
        // and the result wouldn't hold up on new text.
        coverage.1 += 1;
        if let Some(q) = table.lookup(&ctx, vocab) {
            coverage.0 += 1;
            if lambda > 0.0 {
                for (pi, qi) in p.iter_mut().zip(&q) {
                    *pi = (1.0 - lambda) * *pi + lambda * qi;
                }
            }
        }
        // Floor so a zero in the mix doesn't give infinity.
        total -= p[tgt].max(1e-9).ln();
    }
    total / targets.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_remembers_what_it_saw() {
        // "abcabcabc...": after "abc" it's always 'a'.
        let train: Vec<usize> = "abcabcabcabcabcabcabcabc".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let ctx: Vec<usize> = "abcabcab".bytes().map(|b| b as usize).collect();
        let p = r.lookup(&ctx, 256).expect("should recognize the context");
        assert!(p['c' as usize] > 0.9, "expected 'c' and got {:?}", p['c' as usize]);
    }

    #[test]
    fn it_stays_quiet_about_what_it_never_saw() {
        // A table that makes things up is worse than one that stays quiet:
        // the mix carries the lie into the final prediction.
        let train: Vec<usize> = "aaaaaaaaaaaaaaaa".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let ctx: Vec<usize> = "zzzzzzzz".bytes().map(|b| b as usize).collect();
        assert!(r.lookup(&ctx, 256).is_none());
    }

    #[test]
    fn one_sighting_is_not_knowledge() {
        // A context seen only once would give probability 1 to a byte:
        // that's memorization, not a distribution. MIN_COUNT filters it out.
        let train: Vec<usize> = "qwertyuiopasdfgh".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let ctx: Vec<usize> = "qwertyui".bytes().map(|b| b as usize).collect();
        assert!(r.lookup(&ctx, 256).is_none(), "trusted a single occurrence");
    }

    #[test]
    fn observe_is_a_vote_not_a_flag() {
        // A single `observe` isn't enough -- it stays quiet, same discipline
        // as MIN_COUNT in build(). Only with the second vote does it answer.
        let train: Vec<usize> = "xyzxyzxyzxyz".bytes().map(|b| b as usize).collect();
        let mut r = Recall::build(&train);
        let ctx: Vec<usize> = "qqqqqqqq".bytes().map(|b| b as usize).collect();
        assert!(r.lookup(&ctx, 256).is_none(), "shouldn't know anything about this context yet");
        r.observe(&ctx, b'!');
        assert!(r.lookup(&ctx, 256).is_none(), "a single vote isn't MIN_COUNT, still silent");
        r.observe(&ctx, b'!');
        let p = r.lookup(&ctx, 256).expect("with 2 votes it should answer now");
        assert!(p['!' as usize] > 0.9, "expected '!' and got {:?}", p['!' as usize]);
    }

    #[test]
    fn lambda_zero_is_exactly_the_model_alone() {
        // The baseline has to come out of the SAME code as the mix, or two
        // different softmax implementations would be getting compared.
        let train: Vec<usize> = "abcabcabcabc".bytes().map(|b| b as usize).collect();
        let r = Recall::build(&train);
        let vocab = 4;
        let logits = vec![0.1, 2.0, -1.0, 0.5, 1.0, 0.0, 0.0, 0.0];
        let targets = vec![1usize, 0];
        let mut cov = (0, 0);
        let with = mixed_loss(&logits, vocab, &[], &targets, &r, 0.0, &mut cov);

        // By hand: -log softmax at the target position.
        let mut expected = 0.0;
        for (t, &tgt) in targets.iter().enumerate() {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            expected -= ((row[tgt] - mx).exp() / s).ln();
        }
        expected /= targets.len() as f32;
        assert!((with - expected).abs() < 1e-5, "{with} != {expected}");
    }
}
