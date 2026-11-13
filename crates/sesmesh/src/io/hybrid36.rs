// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid-36 decoding of PDB atom serials and residue numbers beyond the
//! decimal field width (R. W. Grosse-Kunstleve, <https://cci.lbl.gov/hybrid_36/>).
//!
//! Width-`w` fields hold decimal numbers up to `10^w - 1`; larger values use
//! base-36 strings starting with an upper-case letter (`A000` = 10000 for
//! `w = 4`), then with a lower-case letter.

/// Decodes a hybrid-36 field of nominal `width`. Decimal values may be
/// space-padded; base-36 values must fill the width exactly.
pub(crate) fn decode(width: u32, field: &[u8]) -> Option<i64> {
    let first = *field.trim_ascii_start().first()?;
    if first.is_ascii_digit() || first == b'-' || first == b'+' {
        return super::num::parse_i64(field);
    }
    if field.len() != width as usize {
        return None;
    }
    let upper = first.is_ascii_uppercase();
    if !upper && !first.is_ascii_lowercase() {
        return None;
    }
    let mut value: i64 = 0;
    for &c in field {
        let digit = match c {
            b'0'..=b'9' => c - b'0',
            b'A'..=b'Z' if upper => c - b'A' + 10,
            b'a'..=b'z' if !upper => c - b'a' + 10,
            _ => return None,
        };
        value = value * 36 + i64::from(digit);
    }
    let block = 36_i64.pow(width - 1);
    let decimal_limit = 10_i64.pow(width);
    Some(if upper {
        value - 10 * block + decimal_limit
    } else {
        value + 16 * block + decimal_limit
    })
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn reference_values() {
        // Values from the cctbx hybrid_36 reference implementation.
        assert_eq!(decode(4, b"9999"), Some(9999));
        assert_eq!(decode(4, b"A000"), Some(10000));
        assert_eq!(decode(4, b"ZZZZ"), Some(1_223_055));
        assert_eq!(decode(4, b"a000"), Some(1_223_056));
        assert_eq!(decode(4, b"zzzz"), Some(2_436_111));
        assert_eq!(decode(5, b"99999"), Some(99999));
        assert_eq!(decode(5, b"A0000"), Some(100_000));
        assert_eq!(decode(5, b"A0001"), Some(100_001));
        assert_eq!(decode(5, b"zzzzz"), Some(87_440_031));
        assert_eq!(decode(4, b"  12"), Some(12));
        assert_eq!(decode(4, b" -12"), Some(-12));
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(decode(4, b"    "), None);
        assert_eq!(decode(4, b""), None);
        assert_eq!(decode(4, b"A00"), None);
        assert_eq!(decode(4, b"Aa00"), None);
        assert_eq!(decode(4, b"A0-0"), None);
        assert_eq!(decode(4, b"*000"), None);
    }
}
