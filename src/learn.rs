//! **What the model learns from.** Interchangeable rules, not an `if` in the
//! loop.
//!
//! Today's convention is that every example deserves the same effort: for
//! every token there's a forward pass, a backward pass, and an optimizer
//! step, whatever it costs and whatever it contributes. But most of a
//! corpus is trivially predictable --spaces, the second half of a word, "of
//! the"-- and backward costs twice what forward does. We're paying full
//! price for what's already known.
//!
//! An animal doesn't work that way: it learns from what surprises it. Here
//! surprise has a cheap, operational definition -- **this example's loss
//! against what the model has been expecting** -- and whether it helps can
//! be measured.
//!
//! Each rule decides ONE thing: whether this example gets learned from or
//! skipped. What gets skipped is skipped entirely: backward and optimizer
//! both.

/// Decides whether it's worth learning from an example, knowing how much it
/// cost to predict it.
pub trait LearnGate {
    /// `loss` is already computed: the forward pass always happens, because
    /// without it there's nothing to decide with (and no prediction to
    /// give).
    fn should_learn(&mut self, loss: f32) -> bool;

    /// So the run can say which rule it was trained with.
    fn name(&self) -> String;
}

/// The current convention: learn from everything. It's the baseline
/// everything else gets measured against.
pub struct Always;

impl LearnGate for Always {
    fn should_learn(&mut self, _loss: f32) -> bool {
        true
    }
    fn name(&self) -> String {
        "all".into()
    }
}

/// Learn only from what's surprising.
///
/// The expectation is a moving average of the model's own recent loss, so
/// the threshold **moves on its own** as the model improves: there's no
/// magic number that ages badly. With `k = 1.0` whatever did better than
/// average gets skipped; with `k = 0.8`, only what's noticeably easier than
/// average.
///
/// THE KNOWN RISK, which needs watching and not hiding: if only what comes
/// out wrong gets learned, you end up learning **the corpus's noise**,
/// which is exactly what always comes out wrong. That's why the run reports
/// validation: if this happens, validation loss detaches from training loss
/// and it shows.
pub struct Surprise {
    k: f32,
    /// How much weight new data gets in the moving average.
    rate: f32,
    ema: Option<f32>,
    /// Examples at the start that get learned unconditionally. Without
    /// this, the first average is computed with a random model and means
    /// nothing.
    warmup: usize,
    seen: usize,
}

impl Surprise {
    pub fn new(k: f32, warmup: usize) -> Self {
        Surprise { k, rate: 0.02, ema: None, warmup, seen: 0 }
    }
}

impl LearnGate for Surprise {
    fn should_learn(&mut self, loss: f32) -> bool {
        self.seen += 1;
        let expected = self.ema.unwrap_or(loss);
        // The average is ALWAYS updated, whether it gets learned or not:
        // it's the expectation about the world, not about what was
        // studied. If it only updated on the hard examples, it would climb
        // on its own and end up skipping everything.
        self.ema = Some(match self.ema {
            None => loss,
            Some(e) => e + self.rate * (loss - e),
        });
        self.seen <= self.warmup || loss >= self.k * expected
    }

    fn name(&self) -> String {
        format!("surprise k={:.2}", self.k)
    }
}

/// Builds the rule from the command line. `k <= 0` is the baseline.
pub fn gate(k: f32, warmup: usize) -> Box<dyn LearnGate> {
    if k <= 0.0 {
        Box::new(Always)
    } else {
        Box::new(Surprise::new(k, warmup))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_learns_everything() {
        let mut g = Always;
        assert!(g.should_learn(0.0));
        assert!(g.should_learn(99.0));
    }

    #[test]
    fn surprise_skips_the_easy_and_keeps_the_hard() {
        let mut g = Surprise::new(1.0, 0);
        // Stabilizes around 2.0.
        for _ in 0..500 {
            g.should_learn(2.0);
        }
        assert!(!g.should_learn(1.0), "something much easier than average should be skipped");
        assert!(g.should_learn(4.0), "something much harder should be learned");
    }

    #[test]
    fn warmup_learns_unconditionally() {
        // The first average would be computed with a random model: it
        // means nothing and would decide badly right when it matters most.
        let mut g = Surprise::new(1.0, 10);
        for i in 0..10 {
            assert!(g.should_learn(0.0001), "warmup example {i} got skipped");
        }
    }

    #[test]
    fn the_expectation_tracks_everything_not_only_what_was_studied() {
        // If the average only updated on what got learned (the hard
        // examples), it would climb on its own until it skipped
        // everything. Here it has to come down when the world gets easy,
        // and go back to accepting what's hard.
        let mut g = Surprise::new(1.0, 0);
        for _ in 0..500 {
            g.should_learn(5.0);
        }
        for _ in 0..500 {
            g.should_learn(0.5);
        }
        assert!(g.should_learn(1.5), "the expectation did not follow the world downward");
    }
}
