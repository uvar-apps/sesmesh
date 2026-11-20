// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Errors of the GPU back ends.

/// An error of a GPU back end.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The vendor's runtime library is not installed (or not found).
    #[error("GPU runtime library not found ({tried})")]
    NoRuntime {
        /// `tried` and the library names and paths tried, comma separated,
        /// or why there is nothing to try on this platform.
        tried: String,
    },
    /// The runtime library loaded but lacks a function or has an unexpected
    /// layout (unsupported version).
    #[error("GPU runtime library {library} is unusable: {reason}")]
    BadRuntime {
        /// Path or name of the library.
        library: String,
        /// What is missing or wrong.
        reason: String,
    },
    /// The runtime reports no device, or the requested index does not exist.
    #[error("no usable GPU: {0}")]
    NoDevice(String),
    /// This build embeds no GPU code (built without the GPU compiler).
    #[error(
        "this build has no GPU kernels (built with SESMESH_GPU_NO_KERNELS or for docs.rs); \
         rebuild with the GPU compiler installed"
    )]
    NoKernels,
    /// No code object of this build runs on the device.
    #[error("no GPU code in this build runs on {arch} (built for {built})")]
    Unsupported {
        /// The device's architecture, e.g. `gfx1151`.
        arch: String,
        /// The targets this build has code for.
        built: String,
    },
    /// A runtime call failed.
    #[error("{call} failed: {message} (error {code})")]
    Call {
        /// The runtime function.
        call: &'static str,
        /// The runtime's error code.
        code: i32,
        /// The runtime's description of the error.
        message: String,
    },
    /// The device does not compute IEEE operations as the CPU does (it would
    /// not reproduce CPU results).
    #[error("GPU self-test failed: {0}")]
    SelfTest(String),
    /// A copy between buffers of different lengths.
    #[error("length mismatch: the buffer holds {buffer} elements, the slice {slice}")]
    Length {
        /// Elements in the device or host buffer.
        buffer: usize,
        /// Elements in the slice.
        slice: usize,
    },
}

/// Result with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
