//! Non-fatal diagnostics (warnings) collected while reading and processing
//! structures.
//!
//! The library never prints (Spec FR-DIAG-1): warnings are aggregated here and
//! returned to the caller, which decides how to report them.

use std::fmt;

/// Category of a [`Warning`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum WarningKind {
    /// A non-essential field could not be parsed; a default value was used.
    InvalidField,
    /// The element was missing and was inferred from the atom name.
    InferredElement,
    /// The element could not be determined.
    UnknownElement,
    /// Alternate conformations were discarded by the alt-loc policy.
    AltLocResolved,
    /// No radius was found for an atom; the fallback radius was used.
    UnknownRadius,
    /// The chosen radius set does not cover some atoms well (for example
    /// explicit hydrogens with united-atom ProtOr radii).
    RadiusSetMismatch,
}

/// An aggregated warning: one entry per distinct `(kind, message)`, with an
/// occurrence count and a few examples.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Warning {
    /// Category.
    pub kind: WarningKind,
    /// Human-readable description shared by all occurrences.
    pub message: String,
    /// Number of occurrences.
    pub count: u64,
    /// Up to [`Diagnostics::MAX_EXAMPLES`] example locations or values.
    pub examples: Vec<String>,
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if self.count > 1 {
            write!(f, " ({} times)", self.count)?;
        }
        if !self.examples.is_empty() {
            write!(f, "; e.g. {}", self.examples.join(", "))?;
            if self.count > self.examples.len() as u64 {
                write!(f, ", ...")?;
            }
        }
        Ok(())
    }
}

/// Collection of aggregated warnings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Diagnostics {
    warnings: Vec<Warning>,
}

impl Diagnostics {
    /// Maximum number of examples kept per warning.
    pub const MAX_EXAMPLES: usize = 5;

    /// Creates an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one occurrence of a warning. `example` is only evaluated while
    /// fewer than [`Self::MAX_EXAMPLES`] examples are stored.
    pub fn warn(&mut self, kind: WarningKind, message: &str, example: impl FnOnce() -> String) {
        let i = self
            .warnings
            .iter()
            .position(|w| w.kind == kind && w.message == message)
            .unwrap_or_else(|| {
                self.warnings.push(Warning {
                    kind,
                    message: message.to_owned(),
                    count: 0,
                    examples: Vec::new(),
                });
                self.warnings.len() - 1
            });
        let entry = &mut self.warnings[i];
        entry.count += 1;
        if entry.examples.len() < Self::MAX_EXAMPLES {
            entry.examples.push(example());
        }
    }

    /// Adds all warnings of `other`, aggregating identical entries.
    pub fn merge(&mut self, other: Diagnostics) {
        for w in other.warnings {
            match self
                .warnings
                .iter_mut()
                .find(|x| x.kind == w.kind && x.message == w.message)
            {
                Some(x) => {
                    x.count += w.count;
                    let room = Self::MAX_EXAMPLES.saturating_sub(x.examples.len());
                    x.examples.extend(w.examples.into_iter().take(room));
                }
                None => self.warnings.push(w),
            }
        }
    }

    /// The aggregated warnings in first-occurrence order.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// `true` when no warning was recorded.
    pub fn is_empty(&self) -> bool {
        self.warnings.is_empty()
    }

    /// Total number of occurrences of warnings of `kind`.
    pub fn count(&self, kind: WarningKind) -> u64 {
        self.warnings
            .iter()
            .filter(|w| w.kind == kind)
            .map(|w| w.count)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_by_kind_and_message() {
        let mut d = Diagnostics::new();
        for i in 0..7 {
            d.warn(WarningKind::InvalidField, "bad serial", || {
                format!("line {i}")
            });
        }
        d.warn(WarningKind::UnknownElement, "unknown element", || {
            "X1".into()
        });
        assert_eq!(d.warnings().len(), 2);
        assert_eq!(d.warnings()[0].count, 7);
        assert_eq!(d.warnings()[0].examples.len(), Diagnostics::MAX_EXAMPLES);
        assert_eq!(d.count(WarningKind::UnknownElement), 1);
        let text = d.warnings()[0].to_string();
        assert!(
            text.starts_with("bad serial (7 times); e.g. line 0"),
            "{text}"
        );
        assert!(text.ends_with(", ..."));
    }

    #[test]
    fn merge_combines_counts() {
        let mut a = Diagnostics::new();
        a.warn(WarningKind::InvalidField, "m", || "a".into());
        let mut b = Diagnostics::new();
        b.warn(WarningKind::InvalidField, "m", || "b".into());
        b.warn(WarningKind::InferredElement, "n", || "c".into());
        a.merge(b);
        assert_eq!(a.count(WarningKind::InvalidField), 2);
        assert_eq!(a.warnings()[0].examples, ["a", "b"]);
        assert_eq!(a.warnings().len(), 2);
    }
}
