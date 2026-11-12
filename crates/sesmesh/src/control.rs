//! Cooperative cancellation and progress reporting (Spec FR-LIB-5,
//! Plan T5.6).

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};

/// A stage of [`compute_surface_with`](crate::compute_surface_with).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Stage {
    /// SAS arrangement: neighbours, circles, triple points.
    Geometry,
    /// Brick classification.
    Classify,
    /// Field sampling and per-brick meshing (reported per brick).
    Mesh,
    /// Global assembly.
    Assemble,
    /// Validation, components and measurements.
    Measure,
}

impl Stage {
    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Geometry => "geometry",
            Self::Classify => "classify",
            Self::Mesh => "mesh",
            Self::Assemble => "assemble",
            Self::Measure => "measure",
        }
    }
}

/// A progress report: `done` of `total` work units of `stage`.
///
/// Every stage reports `(0, total)` when it starts and `(total, total)` when
/// it ends; [`Stage::Mesh`] also reports about every percent of its bricks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Progress {
    /// Current stage.
    pub stage: Stage,
    /// Units done.
    pub done: u64,
    /// Units in the stage (bricks for [`Stage::Mesh`], 1 otherwise).
    pub total: u64,
}

/// Cancellation flag and progress callback for a computation.
///
/// The callback may run on any worker thread, concurrently with itself;
/// within [`Stage::Mesh`], `done` values can arrive slightly out of order.
/// Setting the flag makes the computation return
/// [`Error::Cancelled`] promptly (it is checked per sphere and per brick).
///
/// ```
/// use std::sync::atomic::AtomicBool;
/// use sesmesh::{compute_surface_with, Control, Error, Sphere, SurfaceParams};
///
/// let cancel = AtomicBool::new(true); // already cancelled
/// let control = Control::new().with_cancel(&cancel);
/// let r = compute_surface_with(&[Sphere::new([0.0; 3], 1.5)], &SurfaceParams::default(), &control);
/// assert!(matches!(r, Err(Error::Cancelled)));
/// ```
#[derive(Clone, Copy, Default)]
pub struct Control<'a> {
    cancel: Option<&'a AtomicBool>,
    progress: Option<&'a (dyn Fn(Progress) + Sync)>,
}

impl fmt::Debug for Control<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Control")
            .field("cancel", &self.cancel)
            .field("progress", &self.progress.map(|_| "Fn(Progress)"))
            .finish()
    }
}

impl<'a> Control<'a> {
    /// No cancellation, no progress.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancels the computation when `flag` becomes `true`.
    #[must_use]
    pub fn with_cancel(mut self, flag: &'a AtomicBool) -> Self {
        self.cancel = Some(flag);
        self
    }

    /// Calls `callback` with progress reports.
    #[must_use]
    pub fn with_progress(mut self, callback: &'a (dyn Fn(Progress) + Sync)) -> Self {
        self.progress = Some(callback);
        self
    }

    /// `true` once cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }

    /// `Err(Error::Cancelled)` once cancellation was requested.
    pub(crate) fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    pub(crate) fn report(&self, stage: Stage, done: u64, total: u64) {
        if let Some(p) = self.progress {
            p(Progress { stage, done, total });
        }
    }

    pub(crate) fn has_progress(&self) -> bool {
        self.progress.is_some()
    }
}
