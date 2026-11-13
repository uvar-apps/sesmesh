// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Line iteration with 1-based line numbers and CRLF handling.

/// Iterator over `(line number, line)` with `\n` / `\r\n` stripped.
pub(crate) struct Lines<'a> {
    data: &'a [u8],
    pos: usize,
    line: usize,
}

impl<'a> Lines<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            line: 0,
        }
    }

    /// Number of the last line returned (0 before the first).
    pub(crate) fn line(&self) -> usize {
        self.line
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = (usize, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let rest = &self.data[self.pos..];
        let end = memchr::memchr(b'\n', rest).unwrap_or(rest.len());
        let mut line = &rest[..end];
        if let [head @ .., b'\r'] = line {
            line = head;
        }
        self.pos += end + 1;
        self.line += 1;
        Some((self.line, line))
    }
}

/// Splits on ASCII whitespace, skipping empty tokens.
pub(crate) fn fields(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(u8::is_ascii_whitespace)
        .filter(|t| !t.is_empty())
}

/// 1-based byte column of `token` inside `line` (both from the same buffer).
pub(crate) fn column_of(line: &[u8], token: &[u8]) -> usize {
    (token.as_ptr() as usize).saturating_sub(line.as_ptr() as usize) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_and_fields() {
        let v: Vec<_> = Lines::new(b"a\r\nb\n\nc").collect();
        assert_eq!(v, [(1, &b"a"[..]), (2, b"b"), (3, b""), (4, b"c")]);
        let line = b"  x  yy\tz ";
        let f: Vec<_> = fields(line).collect();
        assert_eq!(f, [&b"x"[..], b"yy", b"z"]);
        assert_eq!(column_of(line, f[1]), 6);
    }
}
