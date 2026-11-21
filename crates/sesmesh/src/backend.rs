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
    /// Device 0 of the first of this build's GPU back ends (HIP, then
    /// CUDA) that opens and passes its self-test, else the CPU. When a GPU
    /// runtime is installed but its device cannot be used, the output
    /// carries a [`WarningKind::GpuFallback`] warning saying why.
    Auto,
    /// An AMD GPU through HIP (ROCm; feature `hip`). Fails if the device
    /// cannot be used.
    Hip(HipOptions),
    /// An NVIDIA GPU through CUDA (feature `cuda`). Fails if the device
    /// cannot be used.
    Cuda(CudaOptions),
}

/// Options of [`Backend::Hip`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct HipOptions {
    /// Device index (HIP ordinal, as `rocm-smi` and `HIP_VISIBLE_DEVICES`
    /// count). Default 0.
    pub device: usize,
}

/// Options of [`Backend::Cuda`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct CudaOptions {
    /// Device index (CUDA ordinal, as `CUDA_VISIBLE_DEVICES` counts; with
    /// `CUDA_DEVICE_ORDER=PCI_BUS_ID` also as `nvidia-smi` does). Default 0.
    pub device: usize,
}

impl Backend {
    /// [`Backend::Hip`] on device `index`.
    #[must_use]
    pub fn hip(index: usize) -> Self {
        Self::Hip(HipOptions { device: index })
    }

    /// [`Backend::Cuda`] on device `index`.
    #[must_use]
    pub fn cuda(index: usize) -> Self {
        Self::Cuda(CudaOptions { device: index })
    }

    /// Lower-case name: `cpu`, `auto`, `hip` or `cuda`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Hip(_) => "hip",
            Self::Cuda(_) => "cuda",
        }
    }

    /// Whether this build includes the back end (`Cpu` and `Auto` always;
    /// `Hip` with the feature `hip`, `Cuda` with `cuda`).
    pub fn is_compiled(&self) -> bool {
        match self {
            Self::Cpu | Self::Auto => true,
            Self::Hip(_) => cfg!(feature = "hip"),
            Self::Cuda(_) => cfg!(feature = "cuda"),
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
    /// An open, self-tested CUDA device.
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<sesmesh_gpu::cuda::Device>),
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
            Backend::Hip(o) => open::hip(o.device)
                .map(Self::Hip)
                .map_err(|u| unusable("hip", o.device, &u.message)),
            #[cfg(feature = "cuda")]
            Backend::Cuda(o) => open::cuda(o.device)
                .map(Self::Cuda)
                .map_err(|u| unusable("cuda", o.device, &u.message)),
            #[allow(unreachable_patterns, reason = "with every GPU feature")]
            b => Err(Error::Unsupported(format!(
                "the {name} back end is not in this build (rebuild with the `{name}` feature)",
                name = b.name()
            ))),
        }
    }

    /// Device 0 of the first GPU back end that opens: HIP, then CUDA.
    fn auto() -> Self {
        #[allow(unused_mut, reason = "without GPU features")]
        let mut why: Vec<String> = Vec::new();
        #[cfg(feature = "hip")]
        match open::hip(0) {
            Ok(d) => return Self::Hip(d),
            Err(u) if u.quiet => {}
            Err(u) => why.push(format!("hip device 0: {}", u.message)),
        }
        #[cfg(feature = "cuda")]
        match open::cuda(0) {
            Ok(d) => return Self::Cuda(d),
            Err(u) if u.quiet => {}
            Err(u) => why.push(format!("cuda device 0: {}", u.message)),
        }
        Self::Cpu {
            fallback: (!why.is_empty()).then(|| why.join("; ")),
        }
    }

    /// The GPU device, if this is one.
    #[cfg(any(feature = "hip", feature = "cuda"))]
    pub(crate) fn gpu(&self) -> Option<crate::field::GpuDevice<'_>> {
        use crate::field::GpuDevice;
        match self {
            Self::Cpu { .. } => None,
            #[cfg(feature = "hip")]
            Self::Hip(d) => Some(GpuDevice::Hip(d)),
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => Some(GpuDevice::Cuda(d)),
        }
    }

    /// Name for [`crate::SurfaceStats::backend`].
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Cpu { .. } => "cpu",
            #[cfg(feature = "hip")]
            Self::Hip(_) => "hip",
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
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
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => {
                let i = d.info();
                Some(if i.code == i.arch {
                    format!("{} ({})", i.name, i.arch)
                } else {
                    format!("{} ({}, {})", i.name, i.arch, i.code)
                })
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

/// The error of an explicit GPU back end whose device cannot be used.
#[cfg(any(feature = "hip", feature = "cuda"))]
fn unusable(name: &str, index: usize, why: &str) -> Error {
    Error::Backend(format!("{name} device {index}: {why}"))
}

/// Process-wide GPU state: each runtime library is loaded once and devices
/// stay open (opening runs the self-test).
#[cfg(any(feature = "hip", feature = "cuda"))]
mod open {
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use sesmesh_gpu::Error as GpuError;

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

    /// One back end's runtime and open devices.
    struct Devices<R, D> {
        runtime: OnceLock<Result<R, (bool, String)>>,
        open: Mutex<Vec<(usize, Arc<D>)>>,
    }

    impl<R: Clone, D> Devices<R, D> {
        const fn new() -> Self {
            Self {
                runtime: OnceLock::new(),
                open: Mutex::new(Vec::new()),
            }
        }

        fn get(
            &self,
            index: usize,
            load: impl FnOnce() -> Result<R, GpuError>,
            open: impl FnOnce(&R, usize) -> Result<D, GpuError>,
        ) -> Result<Arc<D>, Unusable> {
            let runtime = self
                .runtime
                .get_or_init(|| {
                    load().map_err(|e| {
                        let u = Unusable::from(e);
                        (u.quiet, u.message)
                    })
                })
                .clone()
                .map_err(|(quiet, message)| Unusable { quiet, message })?;
            let mut devices = self.open.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some((_, d)) = devices.iter().find(|(i, _)| *i == index) {
                return Ok(Arc::clone(d));
            }
            let device = Arc::new(open(&runtime, index)?);
            devices.push((index, Arc::clone(&device)));
            Ok(device)
        }
    }

    #[cfg(feature = "hip")]
    pub(super) fn hip(index: usize) -> Result<Arc<sesmesh_gpu::hip::Device>, Unusable> {
        use sesmesh_gpu::hip::{Device, Runtime};
        static DEVICES: Devices<Runtime, Device> = Devices::new();
        DEVICES.get(index, Runtime::load, Runtime::open)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn cuda(index: usize) -> Result<Arc<sesmesh_gpu::cuda::Device>, Unusable> {
        use sesmesh_gpu::cuda::{Device, Runtime};
        static DEVICES: Devices<Runtime, Device> = Devices::new();
        DEVICES.get(index, Runtime::load, Runtime::open)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_defaults() {
        assert_eq!(Backend::default(), Backend::Cpu);
        assert_eq!(Backend::hip(1), Backend::Hip(HipOptions { device: 1 }));
        assert_eq!(Backend::cuda(2), Backend::Cuda(CudaOptions { device: 2 }));
        assert_eq!(HipOptions::default().device, 0);
        assert_eq!(CudaOptions::default().device, 0);
        let names: Vec<_> = [
            Backend::Cpu,
            Backend::Auto,
            Backend::hip(0),
            Backend::cuda(0),
        ]
        .iter()
        .map(Backend::name)
        .collect();
        assert_eq!(names, ["cpu", "auto", "hip", "cuda"]);
        assert_eq!(Backend::hip(0).is_compiled(), cfg!(feature = "hip"));
        assert_eq!(Backend::cuda(0).is_compiled(), cfg!(feature = "cuda"));
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
        #[cfg(not(feature = "cuda"))]
        assert_eq!(Resolved::new(&Backend::Auto).unwrap().name(), "cpu");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_the_feature_is_unsupported() {
        let e = Resolved::new(&Backend::cuda(0)).unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)), "{e}");
        assert!(e.to_string().contains("`cuda` feature"), "{e}");
    }

    /// With the feature: `Auto` uses HIP whenever `Hip` opens, and an
    /// absent device index is an error.
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
                assert_ne!(auto.name(), "hip");
            }
        }
        let e = Resolved::new(&Backend::hip(1 << 20)).unwrap_err();
        assert!(matches!(e, Error::Backend(_)), "{e}");
    }

    /// With the feature: `Auto` uses CUDA whenever `Cuda` opens and HIP does
    /// not, and an absent device index is an error.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_and_auto_agree() {
        let auto = Resolved::new(&Backend::Auto).unwrap();
        let hip = cfg!(feature = "hip") && Resolved::new(&Backend::hip(0)).is_ok();
        match Resolved::new(&Backend::cuda(0)) {
            Ok(r) => {
                assert_eq!(r.name(), "cuda");
                assert_eq!(auto.name(), if hip { "hip" } else { "cuda" });
                assert!(r.device().is_some_and(|d| d.contains("sm_")));
            }
            Err(e) => {
                assert!(matches!(e, Error::Backend(_)), "{e}");
                assert!(
                    e.to_string()
                        .starts_with("GPU back end unavailable: cuda device 0")
                );
                assert_ne!(auto.name(), "cuda");
            }
        }
        let e = Resolved::new(&Backend::cuda(1 << 20)).unwrap_err();
        assert!(matches!(e, Error::Backend(_)), "{e}");
    }
}
