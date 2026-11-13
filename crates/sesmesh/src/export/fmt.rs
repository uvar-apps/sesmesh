// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast number formatting for text mesh formats (Design §10.1).

const POW10: [u64; 10] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
];

/// Largest precision supported by [`push_fixed`].
pub(crate) const MAX_PRECISION: usize = 9;

/// Appends `x` with `prec` decimals, byte-identical to
/// `format!("{x:.prec$}")`.
///
/// For an `f32`, `x · 10^prec` is exact in `f64` (24-bit mantissa times a
/// ≤ 21-bit odd factor), so rounding it half-to-even reproduces the exact
/// decimal rounding of the standard formatter, including the sign of
/// negative values that round to zero.
pub(crate) fn push_fixed(buf: &mut Vec<u8>, x: f32, prec: usize) {
    debug_assert!(prec <= MAX_PRECISION);
    let scaled = (f64::from(x) * POW10[prec] as f64).round_ties_even();
    if !scaled.is_finite() || scaled.abs() >= 1e18 {
        buf.extend_from_slice(format!("{x:.prec$}").as_bytes());
        return;
    }
    if x.is_sign_negative() {
        buf.push(b'-');
    }
    let m = scaled.abs() as u64;
    push_u64(buf, m / POW10[prec]);
    if prec > 0 {
        buf.push(b'.');
        let frac = m % POW10[prec];
        let start = buf.len();
        buf.resize(start + prec, b'0');
        let mut f = frac;
        for k in (0..prec).rev() {
            buf[start + k] = b'0' + (f % 10) as u8;
            f /= 10;
        }
    }
}

/// Appends the decimal digits of `n`.
pub(crate) fn push_u64(buf: &mut Vec<u8>, mut n: u64) {
    let mut tmp = [0u8; 20];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    buf.extend_from_slice(&tmp[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn fixed(x: f32, prec: usize) -> String {
        let mut b = Vec::new();
        push_fixed(&mut b, x, prec);
        String::from_utf8(b).unwrap()
    }

    #[test]
    fn edge_cases_match_std() {
        let values = [
            0.0f32,
            -0.0,
            0.5,
            1.5,
            2.5,
            -0.5,
            0.25,
            0.125,
            0.375,
            0.03125,
            -0.000_01,
            1e-9,
            123_456.79,
            -98_765.43,
            f32::MIN_POSITIVE,
            1e15,
            -3e30,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for x in values {
            for prec in 0..=MAX_PRECISION {
                assert_eq!(fixed(x, prec), format!("{x:.prec$}"), "{x:e} {prec}");
            }
        }
    }

    #[test]
    fn integers() {
        for n in [0u64, 7, 10, 99, 100, 4_294_967_295, u64::MAX] {
            let mut b = Vec::new();
            push_u64(&mut b, n);
            assert_eq!(String::from_utf8(b).unwrap(), n.to_string());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20_000))]
        #[test]
        fn random_bits_match_std(bits in any::<u32>(), prec in 0usize..=MAX_PRECISION) {
            let x = f32::from_bits(bits);
            prop_assert_eq!(fixed(x, prec), format!("{x:.prec$}"));
        }

        #[test]
        fn coordinates_match_std(x in -2000.0f32..2000.0, prec in 0usize..=MAX_PRECISION) {
            prop_assert_eq!(fixed(x, prec), format!("{x:.prec$}"));
        }
    }
}
