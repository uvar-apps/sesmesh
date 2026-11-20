// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU back ends of [sesmesh](https://docs.rs/sesmesh): the implementation
//! crate.
//!
//! This crate is not meant to be used directly. Enable sesmesh's GPU
//! features and choose the back end through sesmesh's API or CLI. Its API
//! serves sesmesh and may change in minor releases.
//!
//! Each GPU vendor is a feature:
//!
//! * `hip`: AMD GPUs through HIP (ROCm), module [`hip`]. Building it needs
//!   ROCm's `hipcc` (or `SESMESH_GPU_NO_KERNELS=1` for code checks: such a
//!   build has no kernels). Running needs only the HIP runtime library,
//!   which is loaded when a device is opened, so a binary built with the
//!   feature still runs, on the CPU, where ROCm is missing.
//!
//! All the `unsafe` code of sesmesh lives here: `sesmesh` itself forbids
//! it. Device code follows the exactness rules of `docs/05-gpu-hip.md` §3,
//! so kernels reproduce the CPU's floating-point results bit for bit.

mod error;

pub use error::{Error, Result};

#[cfg(feature = "hip")]
#[allow(unsafe_code)]
pub mod hip;
