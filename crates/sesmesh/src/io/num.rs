//! Number parsing on raw bytes, shared by all readers.

/// Parses a finite `f64`, ignoring surrounding ASCII whitespace.
pub(crate) fn parse_f64(bytes: &[u8]) -> Option<f64> {
    let t = bytes.trim_ascii();
    if t.is_empty() {
        return None;
    }
    fast_float2::parse::<f64, _>(t)
        .ok()
        .filter(|v| v.is_finite())
}

/// Parses a finite `f32`, ignoring surrounding ASCII whitespace.
pub(crate) fn parse_f32(bytes: &[u8]) -> Option<f32> {
    let t = bytes.trim_ascii();
    if t.is_empty() {
        return None;
    }
    fast_float2::parse::<f32, _>(t)
        .ok()
        .filter(|v| v.is_finite())
}

/// Parses a decimal integer with optional sign, ignoring surrounding ASCII
/// whitespace.
pub(crate) fn parse_i64(bytes: &[u8]) -> Option<i64> {
    let t = bytes.trim_ascii();
    let (negative, digits) = match t {
        [b'-', rest @ ..] => (true, rest),
        [b'+', rest @ ..] => (false, rest),
        _ => (false, t),
    };
    if digits.is_empty() {
        return None;
    }
    let mut value: i64 = 0;
    for &c in digits {
        if !c.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(i64::from(c - b'0'))?;
    }
    Some(if negative { -value } else { value })
}

/// [`parse_i64`] narrowed to `i32`.
pub(crate) fn parse_i32(bytes: &[u8]) -> Option<i32> {
    parse_i64(bytes).and_then(|v| i32::try_from(v).ok())
}

/// Lossy display of a field for error messages (trimmed, at most 32 bytes).
pub(crate) fn show(bytes: &[u8]) -> String {
    let t = bytes.trim_ascii();
    let t = &t[..t.len().min(32)];
    String::from_utf8_lossy(t).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats() {
        assert_eq!(parse_f64(b"  -12.345 "), Some(-12.345));
        assert_eq!(parse_f64(b"1e3"), Some(1000.0));
        assert_eq!(parse_f64(b""), None);
        assert_eq!(parse_f64(b"   "), None);
        assert_eq!(parse_f64(b"1.2.3"), None);
        assert_eq!(parse_f64(b"nan"), None);
        assert_eq!(parse_f64(b"inf"), None);
        assert_eq!(parse_f32(b" 0.50"), Some(0.5));
    }

    #[test]
    fn integers() {
        assert_eq!(parse_i64(b" 42 "), Some(42));
        assert_eq!(parse_i64(b"-7"), Some(-7));
        assert_eq!(parse_i64(b"+7"), Some(7));
        assert_eq!(parse_i64(b"-"), None);
        assert_eq!(parse_i64(b"4 2"), None);
        assert_eq!(parse_i64(b"99999999999999999999"), None);
        assert_eq!(parse_i32(b"3000000000"), None);
        assert_eq!(show(b"  abc  "), "abc");
    }
}
