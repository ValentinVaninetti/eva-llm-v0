//! What can come next. Interchangeable pieces.
//!
//! Two different things, and only the first existed here before:
//!   1. NOT CHOOSING WRONG. A mask over the logits already did that.
//!   2. NOT WASTING COMPUTE. The mask did not, because it applied *after*
//!      computing all 256 logits: the model did the full work, then the
//!      illegal part was thrown away.
//!
//! The difference is WHEN it is asked. Consulted *before* the step, the
//! constraint can say "only this can come next": with a single continuation
//! nothing is decided, it is emitted and the output projection is never
//! computed; with k continuations, k columns of the head are computed instead
//! of the whole vocabulary.
//!
//! In this byte-level model the head is `dim x 256`, ~2.5% of the per-token
//! work, so the saving barely shows. With a 32k vocabulary the head becomes
//! `dim x 32000` and dominates the per-token cost. There it is not a detail.
//!
//! A WARNING THAT CAME MEASURED from the other end of the project: a LOOSE
//! constraint came out worse than none -- with loose grammar, 8 garbage
//! citations appeared where there were zero without grammar. Half a constraint
//! is not half the protection: it is the protection off, plus the illusion of
//! having it. A constraint that cannot say with certainty what is legal must
//! return `Any`, not a half-built list.

/// What the structure allows at this position.
pub enum Allowed {
    /// Any token: the model decides freely.
    Any,
    /// Only these, in order. **With just one, the model isn't consulted.**
    Only(Vec<usize>),
}

impl Allowed {
    /// If there's a single option, there's no decision to make.
    pub fn forced(&self) -> Option<usize> {
        match self {
            Allowed::Only(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }
}

pub trait Constraint {
    /// Consulted BEFORE running the model. `so_far` is everything emitted
    /// so far.
    fn allowed(&mut self, so_far: &[usize]) -> Allowed;

    /// So the run can say under which constraint it generated.
    fn name(&self) -> String;
}

/// No constraint: the model decides everything. This is the baseline.
pub struct Free;

impl Constraint for Free {
    fn allowed(&mut self, _so_far: &[usize]) -> Allowed {
        Allowed::Any
    }
    fn name(&self) -> String {
        "free".into()
    }
}

/// A template with holes: fixed text and spans where the model decides.
///
/// This is the shape real, useful output takes in an actual system --
/// marco emits `{"say": "...", "citing": [2]}` and out of those ~30
/// characters the model only chooses the ones in the hole. Everything else
/// is syntax known in advance that today gets produced token by token,
/// paying the full model for every brace and every comma.
pub struct Skeleton {
    /// Alternating spans: fixed, hole, fixed, hole... A hole carries its
    /// max length and the byte that closes it.
    fixed: Vec<Vec<usize>>,
    slots: Vec<(usize, usize)>,
}

impl Skeleton {
    /// `parts` alternates fixed text and holes: the first and last spans
    /// are fixed. A hole is `(max length, byte that closes it)`.
    ///
    /// WATCH OUT FOR THE CLOSING BYTE: the hole emits it, not the next
    /// span. For `{"say":"hi"}` the template is `["{\"say\":\"", "}"]`
    /// with the hole closing on `"`. Also putting the quote at the start
    /// of the tail would ask for it twice -- happened to me writing the
    /// test.
    pub fn new(parts: &[&str], slots: &[(usize, u8)]) -> Self {
        Skeleton {
            fixed: parts.iter().map(|p| p.bytes().map(|b| b as usize).collect()).collect(),
            slots: slots.iter().map(|&(n, b)| (n, b as usize)).collect(),
        }
    }

    /// How many bytes of the template are fixed: the ceiling of what can be
    /// skipped.
    pub fn fixed_bytes(&self) -> usize {
        self.fixed.iter().map(|f| f.len()).sum()
    }
}

impl Constraint for Skeleton {
    fn allowed(&mut self, so_far: &[usize]) -> Allowed {
        // Reconstructs where it's looking based on what's already emitted.
        // No internal state that could drift out of sync with the text:
        // the template gets re-read every time.
        let mut i = 0usize;
        let mut fi = 0usize;
        let mut si = 0usize;
        loop {
            // Fixed span.
            if fi < self.fixed.len() {
                let f = &self.fixed[fi];
                let done = so_far.len() - i;
                if done < f.len() {
                    return Allowed::Only(vec![f[done]]);
                }
                i += f.len();
                fi += 1;
            }
            // Hole.
            if si < self.slots.len() {
                let (max, closes) = self.slots[si];
                let mut n = 0;
                while i + n < so_far.len() && so_far[i + n] != closes && n < max {
                    n += 1;
                }
                if i + n >= so_far.len() {
                    // Inside the hole: the model decides, unless it went
                    // past the max length, where the only legal thing is
                    // to close it.
                    return if n >= max { Allowed::Only(vec![closes]) } else { Allowed::Any };
                }
                i += n + 1; // the closing byte is already there
                si += 1;
            } else if fi >= self.fixed.len() {
                return Allowed::Any;
            }
        }
    }

    fn name(&self) -> String {
        format!("template ({} fixed spans, {} holes)", self.fixed.len(), self.slots.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(s: &str) -> Vec<usize> {
        s.bytes().map(|b| b as usize).collect()
    }

    #[test]
    fn the_fixed_part_leaves_nothing_to_decide() {
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(20, b'"')]);
        // Byte by byte of the prefix: always a single option.
        for (i, expected) in r#"{"di":""#.bytes().enumerate() {
            let so_far = bytes(&r#"{"di":""#[..i]);
            let a = k.allowed(&so_far);
            assert_eq!(Some(expected as usize), a.forced(), "at position {i}");
        }
    }

    #[test]
    fn inside_the_slot_the_model_decides() {
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(20, b'"')]);
        let a = k.allowed(&bytes(r#"{"di":"ho"#));
        assert!(a.forced().is_none(), "inside the hole the model has to decide");
    }

    #[test]
    fn a_slot_that_ran_out_can_only_close() {
        // The length cap is part of the constraint: without it, the hole
        // is a degree of freedom that fills with garbage -- measured in
        // the other project, an uncapped citation list cited everything.
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(3, b'"')]);
        let a = k.allowed(&bytes(r#"{"di":"abc"#));
        assert_eq!(Some(b'"' as usize), a.forced(), "past the cap it can only close");
    }

    #[test]
    fn after_the_slot_the_tail_is_forced_again() {
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(20, b'"')]);
        assert_eq!(Some(b'}' as usize), k.allowed(&bytes(r#"{"di":"hola""#)).forced());
    }

    #[test]
    fn free_decides_nothing() {
        assert!(Free.allowed(&[]).forced().is_none());
    }
}
