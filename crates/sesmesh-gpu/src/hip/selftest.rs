// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The arithmetic self-test (docs/05-gpu-hip.md §9): the device computes the
//! operations exact kernels rely on, and the host compares the bits.
//!
//! Device kernels use f32 and integer arithmetic only (no f64 on the
//! device). The test runs once on device buffers (copied in and out) and
//! once on pinned host buffers that the kernel reads and writes in place, so
//! opening a device also checks both kinds of memory.

use std::ffi::c_void;

use super::Device;
use crate::Result;
use crate::selftest::{OPS, Rng, SelfTestReport, check, input_f32};

const BLOCK: u32 = 256;

pub(crate) fn run(device: &Device, values: usize, seed: u64) -> Result<SelfTestReport> {
    let mut report = SelfTestReport {
        values,
        f32_mismatches: [0; OPS],
        first: None,
    };
    if values == 0 {
        return Ok(report);
    }
    let mut n = u32::try_from(values).expect("self-test size fits u32");
    let grid = n.div_ceil(BLOCK);
    let mut rng = Rng::new(seed);
    let a: Vec<f32> = (0..values).map(|i| input_f32(&mut rng, i)).collect();
    let b: Vec<f32> = (0..values).map(|i| input_f32(&mut rng, i + 1)).collect();

    // Device buffers.
    let (da, db) = (device.upload(&a)?, device.upload(&b)?);
    let out = device.alloc::<u32>(values * OPS)?;
    let (mut pa, mut pb, mut po) = (da.as_device_ptr(), db.as_device_ptr(), out.as_device_ptr());
    let mut args: [*mut c_void; 4] = [
        (&raw mut pa).cast(),
        (&raw mut pb).cast(),
        (&raw mut po).cast(),
        (&raw mut n).cast(),
    ];
    // SAFETY: `sesmesh_selftest_f32(const float*, const float*, unsigned*,
    // unsigned)`; the buffers hold `n` and `n * OPS` values and outlive the
    // synchronous launch.
    unsafe { device.launch_sync(device.kernels.selftest_f32, grid, BLOCK, &mut args)? };
    check(&mut report, &a, &b, &out.to_vec()?, "device memory");

    // Pinned host buffers, read and written by the kernel in place.
    let mut ha = device.alloc_host::<f32>(values)?;
    let mut hb = device.alloc_host::<f32>(values)?;
    let mut ho = device.alloc_host::<u32>(values * OPS)?;
    ha.copy_from_slice(&a);
    hb.copy_from_slice(&b);
    let (mut pa, mut pb, mut po) = (ha.as_device_ptr(), hb.as_device_ptr(), ho.as_device_ptr());
    let mut args: [*mut c_void; 4] = [
        (&raw mut pa).cast(),
        (&raw mut pb).cast(),
        (&raw mut po).cast(),
        (&raw mut n).cast(),
    ];
    // SAFETY: as above; pinned host memory is device-accessible (UVA), and
    // the host does not touch the buffers during the synchronous launch.
    unsafe { device.launch_sync(device.kernels.selftest_f32, grid, BLOCK, &mut args)? };
    check(&mut report, &a, &b, &ho, "pinned host memory");
    Ok(report)
}
