// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The arithmetic self-test (docs/05-gpu-hip.md §9): the device computes the
//! operations exact kernels rely on, and the host compares the bits.
//!
//! The f32 half runs on device buffers (copied in and out), the f64 half on
//! pinned host buffers that the kernel reads and writes in place, so opening
//! a device also checks both kinds of memory.

use std::ffi::c_void;
use std::fmt;

use super::Device;
use crate::Result;

/// Operations per value; `SELFTEST_OPS` in `kernels/hip/sesmesh.hip`.
pub(crate) const OPS: usize = 8;
const BLOCK: u32 = 256;

const NAMES_F32: [&str; OPS] = [
    "x + y",
    "x * y",
    "x / y",
    "sqrt(|x|)",
    "norm",
    "(d - y) / d",
    "min(x, y)",
    "|d - |y||",
];
const NAMES_F64: [&str; OPS] = [
    "x + y",
    "x * y",
    "x / y",
    "sqrt(|x|)",
    "norm",
    "(d - y) / d",
    "f32(x * 1e3)",
    "|d - |y||",
];

/// `sesmesh_selftest_f32`, same operations in the same order.
fn reference_f32(x: f32, y: f32) -> [f32; OPS] {
    let (dx, dy, dz) = (x - y, y * 0.5 - x, x * y - 1.0);
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    [
        x + y,
        x * y,
        x / y,
        x.abs().sqrt(),
        d,
        (d - y) / d,
        x.min(y),
        (d - y.abs()).abs(),
    ]
}

/// `sesmesh_selftest_f64`, same operations in the same order.
fn reference_f64(x: f64, y: f64) -> [f64; OPS] {
    let (dx, dy, dz) = (x - y, y * 0.5 - x, x * y - 1.0);
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    #[allow(clippy::cast_possible_truncation)]
    let rounded = f64::from((x * 1e3) as f32);
    [
        x + y,
        x * y,
        x / y,
        x.abs().sqrt(),
        d,
        (d - y) / d,
        rounded,
        (d - y.abs()).abs(),
    ]
}

/// Deterministic xorshift64* generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Random exponent in `lo..=hi`.
    fn exponent(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

/// Finite, non-zero inputs (no signed-zero or NaN-payload ambiguity): one in
/// 16 subnormal, one in 16 from the whole exponent range, the rest within
/// 2^±30 (coordinates and radii).
fn input_f32(rng: &mut Rng, i: usize) -> f32 {
    let r = rng.next();
    let sign = (r >> 63) << 31;
    let mantissa = (r >> 8) & 0x7F_FFFF;
    let exponent = match i % 16 {
        0 => 0,
        1 => rng.exponent(1, 254),
        _ => rng.exponent(127 - 30, 127 + 30),
    };
    let mantissa = if exponent == 0 {
        mantissa | 1
    } else {
        mantissa
    };
    #[allow(clippy::cast_possible_truncation)]
    f32::from_bits((sign | (exponent << 23) | mantissa) as u32)
}

/// As [`input_f32`] for f64.
fn input_f64(rng: &mut Rng, i: usize) -> f64 {
    let r = rng.next();
    let sign = (r >> 63) << 63;
    let mantissa = rng.next() & 0xF_FFFF_FFFF_FFFF;
    let exponent = match i % 16 {
        0 => 0,
        1 => rng.exponent(1, 2046),
        _ => rng.exponent(1023 - 30, 1023 + 30),
    };
    let mantissa = if exponent == 0 {
        mantissa | 1
    } else {
        mantissa
    };
    f64::from_bits(sign | (exponent << 52) | mantissa)
}

/// Result of [`Device::self_test`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SelfTestReport {
    /// Input pairs tested.
    pub values: usize,
    /// Results whose bits differ from the CPU's, per f32 operation.
    pub f32_mismatches: [usize; OPS],
    /// As `f32_mismatches`, per f64 operation.
    pub f64_mismatches: [usize; OPS],
    /// The first mismatch, described.
    pub first: Option<String>,
}

impl SelfTestReport {
    /// Whether every result agreed bit for bit (NaNs agree with NaNs).
    #[must_use]
    pub fn passed(&self) -> bool {
        self.f32_mismatches
            .iter()
            .chain(&self.f64_mismatches)
            .all(|&m| m == 0)
    }
}

impl fmt::Display for SelfTestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed() {
            return write!(f, "{} values: all results equal the CPU's", self.values);
        }
        write!(f, "{} values: results differ from the CPU's", self.values)?;
        for (names, counts, ty) in [
            (&NAMES_F32, &self.f32_mismatches, "f32"),
            (&NAMES_F64, &self.f64_mismatches, "f64"),
        ] {
            for (name, &n) in names.iter().zip(counts) {
                if n != 0 {
                    write!(f, "; {ty} {name}: {n}")?;
                }
            }
        }
        if let Some(first) = &self.first {
            write!(f, "; first: {first}")?;
        }
        Ok(())
    }
}

pub(crate) fn run(device: &Device, values: usize, seed: u64) -> Result<SelfTestReport> {
    let mut report = SelfTestReport {
        values,
        f32_mismatches: [0; OPS],
        f64_mismatches: [0; OPS],
        first: None,
    };
    if values == 0 {
        return Ok(report);
    }
    let mut n = u32::try_from(values).expect("self-test size fits u32");
    let grid = n.div_ceil(BLOCK);
    let mut rng = Rng::new(seed);

    // f32 on device buffers.
    let a: Vec<f32> = (0..values).map(|i| input_f32(&mut rng, i)).collect();
    let b: Vec<f32> = (0..values).map(|i| input_f32(&mut rng, i + 1)).collect();
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
    let got = out.to_vec()?;
    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
        for (k, want) in reference_f32(x, y).into_iter().enumerate() {
            let have = f32::from_bits(got[OPS * i + k]);
            if have.to_bits() != want.to_bits() && !(have.is_nan() && want.is_nan()) {
                report.f32_mismatches[k] += 1;
                report.first.get_or_insert_with(|| {
                    format!(
                        "f32 {} at x = {x:e}, y = {y:e}: device {have:e} ({:#010x}), CPU {want:e} \
                         ({:#010x})",
                        NAMES_F32[k],
                        have.to_bits(),
                        want.to_bits()
                    )
                });
            }
        }
    }

    // f64 on pinned host buffers, read and written by the kernel in place.
    let mut ha = device.alloc_host::<f64>(values)?;
    let mut hb = device.alloc_host::<f64>(values)?;
    let mut ho = device.alloc_host::<u64>(values * OPS)?;
    for i in 0..values {
        ha[i] = input_f64(&mut rng, i);
        hb[i] = input_f64(&mut rng, i + 1);
    }
    let (mut pa, mut pb, mut po) = (ha.as_device_ptr(), hb.as_device_ptr(), ho.as_device_ptr());
    let mut args: [*mut c_void; 4] = [
        (&raw mut pa).cast(),
        (&raw mut pb).cast(),
        (&raw mut po).cast(),
        (&raw mut n).cast(),
    ];
    // SAFETY: `sesmesh_selftest_f64(const double*, const double*, unsigned
    // long long*, unsigned)`; pinned host memory is device-accessible (UVA),
    // the buffers hold `n` and `n * OPS` values, are not touched by the host
    // during the synchronous launch, and outlive it.
    unsafe { device.launch_sync(device.kernels.selftest_f64, grid, BLOCK, &mut args)? };
    for i in 0..values {
        let (x, y) = (ha[i], hb[i]);
        for (k, want) in reference_f64(x, y).into_iter().enumerate() {
            let have = f64::from_bits(ho[OPS * i + k]);
            if have.to_bits() != want.to_bits() && !(have.is_nan() && want.is_nan()) {
                report.f64_mismatches[k] += 1;
                report.first.get_or_insert_with(|| {
                    format!(
                        "f64 {} at x = {x:e}, y = {y:e}: device {have:e} ({:#018x}), CPU {want:e} \
                         ({:#018x})",
                        NAMES_F64[k],
                        have.to_bits(),
                        want.to_bits()
                    )
                });
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_are_finite_and_non_zero() {
        let mut rng = Rng::new(7);
        let mut subnormal = (0, 0);
        for i in 0..10_000 {
            let (x, y) = (input_f32(&mut rng, i), input_f64(&mut rng, i));
            assert!(x.is_finite() && x != 0.0, "{x}");
            assert!(y.is_finite() && y != 0.0, "{y}");
            subnormal.0 += usize::from(x.is_subnormal());
            subnormal.1 += usize::from(y.is_subnormal());
        }
        assert_eq!(subnormal, (625, 625));
    }

    #[test]
    fn report_formats_mismatches() {
        let mut r = SelfTestReport {
            values: 4,
            f32_mismatches: [0; OPS],
            f64_mismatches: [0; OPS],
            first: None,
        };
        assert!(r.passed());
        r.f64_mismatches[2] = 1;
        r.first = Some("example".into());
        assert!(!r.passed());
        assert_eq!(
            r.to_string(),
            "4 values: results differ from the CPU's; f64 x / y: 1; first: example"
        );
    }
}
