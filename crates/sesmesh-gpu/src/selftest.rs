// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The arithmetic self-test shared by the back ends (docs/05-gpu-hip.md §9):
//! the device computes the operations exact kernels rely on
//! (`sesmesh_selftest_f32`), and the host compares the bits with its own
//! results. Device kernels use f32 and integer arithmetic only.

use std::fmt;

/// Operations per value; `SELFTEST_OPS` in `kernels/sesmesh.cuh`.
pub(crate) const OPS: usize = 8;
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

/// Deterministic xorshift64* generator.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
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
pub(crate) fn input_f32(rng: &mut Rng, i: usize) -> f32 {
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

/// Result of a device's `self_test`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SelfTestReport {
    /// Input pairs tested (each on device and on pinned host memory).
    pub values: usize,
    /// Results whose bits differ from the CPU's, per f32 operation.
    pub f32_mismatches: [usize; OPS],
    /// The first mismatch, described.
    pub first: Option<String>,
}

impl SelfTestReport {
    /// Whether every result agreed bit for bit (NaNs agree with NaNs).
    #[must_use]
    pub fn passed(&self) -> bool {
        self.f32_mismatches.iter().all(|&m| m == 0)
    }
}

impl fmt::Display for SelfTestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed() {
            return write!(f, "{} values: all results equal the CPU's", self.values);
        }
        write!(f, "{} values: results differ from the CPU's", self.values)?;
        for (name, &n) in NAMES_F32.iter().zip(&self.f32_mismatches) {
            if n != 0 {
                write!(f, "; f32 {name}: {n}")?;
            }
        }
        if let Some(first) = &self.first {
            write!(f, "; first: {first}")?;
        }
        Ok(())
    }
}

/// Compares the device's results for `x[i], y[i]` (`got`, bits) with the
/// CPU's.
pub(crate) fn check(report: &mut SelfTestReport, x: &[f32], y: &[f32], got: &[u32], memory: &str) {
    for (i, (&x, &y)) in x.iter().zip(y).enumerate() {
        for (k, want) in reference_f32(x, y).into_iter().enumerate() {
            let have = f32::from_bits(got[OPS * i + k]);
            if have.to_bits() != want.to_bits() && !(have.is_nan() && want.is_nan()) {
                report.f32_mismatches[k] += 1;
                report.first.get_or_insert_with(|| {
                    format!(
                        "f32 {} at x = {x:e}, y = {y:e} ({memory}): device {have:e} \
                         ({:#010x}), CPU {want:e} ({:#010x})",
                        NAMES_F32[k],
                        have.to_bits(),
                        want.to_bits()
                    )
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_are_finite_and_non_zero() {
        let mut rng = Rng::new(7);
        let mut subnormal = 0;
        for i in 0..10_000 {
            let x = input_f32(&mut rng, i);
            assert!(x.is_finite() && x != 0.0, "{x}");
            subnormal += usize::from(x.is_subnormal());
        }
        assert_eq!(subnormal, 625);
    }

    #[test]
    fn report_formats_mismatches() {
        let mut r = SelfTestReport {
            values: 4,
            f32_mismatches: [0; OPS],
            first: None,
        };
        assert!(r.passed());
        r.f32_mismatches[2] = 1;
        r.first = Some("example".into());
        assert!(!r.passed());
        assert_eq!(
            r.to_string(),
            "4 values: results differ from the CPU's; f32 x / y: 1; first: example"
        );
    }
}
