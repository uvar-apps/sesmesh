// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compute back ends: where the mesh stages run (docs/05-gpu-hip.md §9).
//!
//! The CPU is always available and the default. GPU back ends are Cargo
//! features (`hip`); the [`Backend`] values are the same in every build, and
//! one that is not compiled in is an [`Error::Unsupported`].
//!
//! GPU back ends reproduce the CPU's output bit for bit (exact refinement,
//! docs/05-gpu-hip.md §3), so [`Backend::Auto`] never changes the mesh, only
//! the time.

use crate::diag::{Diagnostics, WarningKind};
use crate::error::{Error, Result};

/// Where the mesh stages run ([`crate::SurfaceParams::backend`]).
///
/// ```
/// use sesmesh::{Backend, SurfaceParams};
///
/// let params = SurfaceParams::ses(1.4).with_backend(Backend::Auto);
/// assert_eq!(params.backend.name(), "auto");
/// assert!(Backend::Cpu.is_compiled());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Backend {
    /// The CPU (the threads of the current rayon pool). Default.
    #[default]
    Cpu,
    /// The first GPU of this build's GPU back ends that opens and passes
    /// its self-test (device 0), else the CPU. When a GPU runtime is
    /// installed but its device cannot be used, the output carries a
    /// [`WarningKind::GpuFallback`] warning saying why.
    Auto,
    /// An AMD GPU through HIP (ROCm; feature `hip`). Fails if the device
    /// cannot be used.
    Hip(HipOptions),
}

/// Options of [`Backend::Hip`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct HipOptions {
    /// Device index (HIP ordinal, as `rocm-smi` and `HIP_VISIBLE_DEVICES`
    /// count). Default 0.
    pub device: usize,
}

impl Backend {
    /// [`Backend::Hip`] on device `index`.
    #[must_use]
    pub fn hip(index: usize) -> Self {
        Self::Hip(HipOptions { device: index })
    }

    /// Lower-case name: `cpu`, `auto` or `hip`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Hip(_) => "hip",
        }
    }

    /// Whether this build includes the back end (`Cpu` and `Auto` always;
    /// `Hip` with the feature `hip`).
    pub fn is_compiled(&self) -> bool {
        match self {
            Self::Cpu | Self::Auto => true,
            Self::Hip(_) => cfg!(feature = "hip"),
        }
    }
}

/// The back end a computation runs on, resolved from [`Backend`].
#[derive(Debug, Clone)]
pub(crate) enum Resolved {
    /// The CPU; with `Auto`, why no GPU was used (a warning) when a GPU
    /// runtime is installed.
    Cpu { fallback: Option<String> },
    /// An open, self-tested HIP device.
    #[cfg(feature = "hip")]
    Hip(std::sync::Arc<sesmesh_gpu::hip::Device>),
}

impl Resolved {
    /// Opens the device of `backend` (devices stay open for the process, so
    /// repeated calls are cheap).
    ///
    /// # Errors
    /// [`Error::Unsupported`] for a back end not in this build,
    /// [`Error::Backend`] for an explicit GPU back end whose device cannot be
    /// used.
    pub(crate) fn new(backend: &Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Ok(Self::Cpu { fallback: None }),
            Backend::Auto => Ok(Self::auto()),
            #[cfg(feature = "hip")]
            Backend::Hip(o) => hip::open(o.device)
                .map(Self::Hip)
                .map_err(|u| Error::Backend(format!("hip device {}: {}", o.device, u.message))),
            #[cfg(not(feature = "hip"))]
            Backend::Hip(_) => Err(Error::Unsupported(
                "the hip back end is not in this build (rebuild with the `hip` feature)".into(),
            )),
        }
    }

    #[cfg(feature = "hip")]
    fn auto() -> Self {
        match hip::open(0) {
            Ok(d) => Self::Hip(d),
            Err(u) => Self::Cpu {
                fallback: (!u.quiet).then(|| format!("hip device 0: {}", u.message)),
            },
        }
    }

    #[cfg(not(feature = "hip"))]
    fn auto() -> Self {
        Self::Cpu { fallback: None }
    }

    /// Name for [`crate::SurfaceStats::backend`].
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Cpu { .. } => "cpu",
            #[cfg(feature = "hip")]
            Self::Hip(_) => "hip",
        }
    }

    /// Description of the device for [`crate::SurfaceStats::device`].
    pub(crate) fn device(&self) -> Option<String> {
        match self {
            Self::Cpu { .. } => None,
            #[cfg(feature = "hip")]
            Self::Hip(d) => {
                let i = d.info();
                Some(format!("{} ({}, {})", i.name, i.arch, i.code_object))
            }
        }
    }

    /// Adds the fallback warning, if any, to `diagnostics`.
    pub(crate) fn report(&self, diagnostics: &mut Diagnostics) {
        if let Self::Cpu {
            fallback: Some(why),
        } = self
        {
            diagnostics.note(
                WarningKind::GpuFallback,
                &format!("no GPU used, computed on the CPU ({why})"),
            );
        }
    }
}

/// Process-wide HIP state: the runtime library is loaded once and devices
/// stay open (opening runs the self-test).
#[cfg(feature = "hip")]
mod hip {
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use sesmesh_gpu::Error as GpuError;
    use sesmesh_gpu::hip::{Device, Runtime};

    /// Why a device cannot be used.
    pub(super) struct Unusable {
        /// No GPU at all (no runtime or no device): `Auto` falls back
        /// without a warning.
        pub(super) quiet: bool,
        pub(super) message: String,
    }

    impl From<GpuError> for Unusable {
        fn from(e: GpuError) -> Self {
            Self {
                quiet: matches!(e, GpuError::NoRuntime { .. } | GpuError::NoDevice(_)),
                message: e.to_string(),
            }
        }
    }

    fn runtime() -> Result<Runtime, Unusable> {
        static RUNTIME: OnceLock<Result<Runtime, (bool, String)>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| {
                Runtime::load().map_err(|e| {
                    let u = Unusable::from(e);
                    (u.quiet, u.message)
                })
            })
            .clone()
            .map_err(|(quiet, message)| Unusable { quiet, message })
    }

    pub(super) fn open(index: usize) -> Result<Arc<Device>, Unusable> {
        static DEVICES: Mutex<Vec<(usize, Arc<Device>)>> = Mutex::new(Vec::new());
        let runtime = runtime()?;
        let mut devices = DEVICES.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, d)) = devices.iter().find(|(i, _)| *i == index) {
            return Ok(Arc::clone(d));
        }
        let device = Arc::new(runtime.open(index)?);
        devices.push((index, Arc::clone(&device)));
        Ok(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_defaults() {
        assert_eq!(Backend::default(), Backend::Cpu);
        assert_eq!(Backend::hip(1), Backend::Hip(HipOptions { device: 1 }));
        assert_eq!(HipOptions::default().device, 0);
        let names: Vec<_> = [Backend::Cpu, Backend::Auto, Backend::hip(0)]
            .iter()
            .map(Backend::name)
            .collect();
        assert_eq!(names, ["cpu", "auto", "hip"]);
        assert_eq!(Backend::hip(0).is_compiled(), cfg!(feature = "hip"));
    }

    #[test]
    fn cpu_resolves_to_cpu() {
        let r = Resolved::new(&Backend::Cpu).unwrap();
        assert_eq!(r.name(), "cpu");
        assert_eq!(r.device(), None);
        let mut d = Diagnostics::new();
        r.report(&mut d);
        assert!(d.is_empty());
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn hip_without_the_feature_is_unsupported() {
        let e = Resolved::new(&Backend::hip(0)).unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)), "{e}");
        assert!(e.to_string().contains("`hip` feature"), "{e}");
        assert_eq!(Resolved::new(&Backend::Auto).unwrap().name(), "cpu");
    }

    /// With the feature: `Auto` and `Hip` agree on whether a device is
    /// usable, and an absent device index is an error.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_and_auto_agree() {
        let auto = Resolved::new(&Backend::Auto).unwrap();
        match Resolved::new(&Backend::hip(0)) {
            Ok(r) => {
                assert_eq!((r.name(), auto.name()), ("hip", "hip"));
                assert!(r.device().is_some_and(|d| d.contains("gfx")));
            }
            Err(e) => {
                assert!(matches!(e, Error::Backend(_)), "{e}");
                assert!(
                    e.to_string()
                        .starts_with("GPU back end unavailable: hip device 0")
                );
                assert_eq!(auto.name(), "cpu");
            }
        }
        let e = Resolved::new(&Backend::hip(1 << 20)).unwrap_err();
        assert!(matches!(e, Error::Backend(_)), "{e}");
    }
}
