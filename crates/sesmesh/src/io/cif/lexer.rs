//! Zero-copy CIF 1.1 tokenizer (Design §5.4).
//!
//! Recognizes data blocks, `loop_`, save frames, `global_`/`stop_`, tags,
//! unquoted values, single/double-quoted values (a closing quote must be
//! followed by whitespace), semicolon text fields, comments, and the special
//! values `?` (unknown) and `.` (inapplicable).

use crate::error::ParseError;

/// A CIF value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Value<'a> {
    /// Unquoted, quoted or text-field content (quotes removed).
    Text(&'a [u8]),
    /// `?`
    Unknown,
    /// `.`
    Inapplicable,
}

impl<'a> Value<'a> {
    /// The text, or `None` for `?` / `.`.
    pub(crate) fn text(self) -> Option<&'a [u8]> {
        match self {
            Value::Text(t) => Some(t),
            Value::Unknown | Value::Inapplicable => None,
        }
    }
}

/// A CIF token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Token<'a> {
    /// `data_NAME`
    DataBlock(&'a [u8]),
    /// `loop_`
    Loop,
    /// `save_NAME` (frame start) or `save_` (frame end, empty name).
    SaveFrame(&'a [u8]),
    /// `global_` or `stop_` (reserved words; ignored by the mmCIF reader).
    Reserved,
    /// `_category.item`
    Tag(&'a [u8]),
    /// A value.
    Value(Value<'a>),
}

/// A token with its 1-based position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Spanned<'a> {
    pub(crate) token: Token<'a>,
    pub(crate) line: usize,
    pub(crate) column: usize,
}

/// Iterator over the tokens of a CIF document.
pub(crate) struct Lexer<'a> {
    data: &'a [u8],
    pos: usize,
    line: usize,
    line_start: usize,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            line: 1,
            line_start: 0,
        }
    }

    fn column(&self, pos: usize) -> usize {
        pos - self.line_start + 1
    }

    fn newline_at(&mut self, pos: usize) {
        self.line += 1;
        self.line_start = pos + 1;
    }

    /// Skips whitespace and comments.
    fn skip_blank(&mut self) {
        while let Some(&c) = self.data.get(self.pos) {
            match c {
                b'\n' => {
                    self.newline_at(self.pos);
                    self.pos += 1;
                }
                b' ' | b'\t' | b'\r' => self.pos += 1,
                b'#' => {
                    self.pos = memchr::memchr(b'\n', &self.data[self.pos..])
                        .map_or(self.data.len(), |i| self.pos + i);
                }
                _ => break,
            }
        }
    }

    fn is_blank(c: u8) -> bool {
        matches!(c, b' ' | b'\t' | b'\r' | b'\n')
    }

    fn error(&self, pos: usize, message: impl Into<String>) -> ParseError {
        ParseError::new(self.line, self.column(pos), message)
    }

    fn quoted(&mut self, quote: u8) -> Result<&'a [u8], ParseError> {
        let open = self.pos;
        let mut i = open + 1;
        loop {
            match self.data.get(i) {
                None | Some(b'\n') => {
                    return Err(self.error(open, "unterminated quoted string"));
                }
                Some(&c) if c == quote => {
                    if self.data.get(i + 1).is_none_or(|&n| Self::is_blank(n)) {
                        self.pos = i + 1;
                        return Ok(&self.data[open + 1..i]);
                    }
                    i += 1;
                }
                Some(_) => i += 1,
            }
        }
    }

    fn text_field(&mut self) -> Result<&'a [u8], ParseError> {
        let open = self.pos;
        let (open_line, open_col) = (self.line, self.column(open));
        let mut i = open + 1;
        loop {
            let Some(nl) = memchr::memchr(b'\n', &self.data[i..]) else {
                return Err(ParseError::new(
                    open_line,
                    open_col,
                    "unterminated text field (no closing ';' line)",
                ));
            };
            let nl = i + nl;
            self.newline_at(nl);
            if self.data.get(nl + 1) == Some(&b';') {
                let mut content = &self.data[open + 1..nl];
                if let [rest @ .., b'\r'] = content {
                    content = rest;
                }
                self.pos = nl + 2;
                return Ok(content);
            }
            i = nl + 1;
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Spanned<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_blank();
        let start = self.pos;
        let &first = self.data.get(start)?;
        let (line, column) = (self.line, self.column(start));
        let token = match first {
            b'\'' | b'"' => match self.quoted(first) {
                Ok(t) => Token::Value(Value::Text(t)),
                Err(e) => return Some(Err(e)),
            },
            b';' if start == self.line_start => match self.text_field() {
                Ok(t) => Token::Value(Value::Text(t)),
                Err(e) => return Some(Err(e)),
            },
            _ => {
                let end = self.data[start..]
                    .iter()
                    .position(|&c| Self::is_blank(c))
                    .map_or(self.data.len(), |i| start + i);
                self.pos = end;
                classify(&self.data[start..end])
            }
        };
        Some(Ok(Spanned {
            token,
            line,
            column,
        }))
    }
}

fn classify(word: &[u8]) -> Token<'_> {
    if word[0] == b'_' {
        return Token::Tag(word);
    }
    let prefix = |p: &[u8]| word.len() >= p.len() && word[..p.len()].eq_ignore_ascii_case(p);
    if prefix(b"data_") {
        Token::DataBlock(&word[5..])
    } else if word.eq_ignore_ascii_case(b"loop_") {
        Token::Loop
    } else if prefix(b"save_") {
        Token::SaveFrame(&word[5..])
    } else if word.eq_ignore_ascii_case(b"global_") || word.eq_ignore_ascii_case(b"stop_") {
        Token::Reserved
    } else if word == b"?" {
        Token::Value(Value::Unknown)
    } else if word == b"." {
        Token::Value(Value::Inapplicable)
    } else {
        Token::Value(Value::Text(word))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(s: &str) -> Vec<Token<'_>> {
        Lexer::new(s.as_bytes()).map(|t| t.unwrap().token).collect()
    }

    #[test]
    fn token_kinds() {
        let doc = "data_1ABC\n# comment\nloop_\n_atom_site.id\n_atom_site.label_atom_id\n\
                   1 \"O5'\"\n2 'it''s ok' # trailing\n3 ?\n4 .\n;line one\nline two\n;\n\
                   save_frame\nsave_\nglobal_\n";
        let t = tokens(doc);
        assert_eq!(
            t,
            [
                Token::DataBlock(b"1ABC"),
                Token::Loop,
                Token::Tag(b"_atom_site.id"),
                Token::Tag(b"_atom_site.label_atom_id"),
                Token::Value(Value::Text(b"1")),
                Token::Value(Value::Text(b"O5'")),
                Token::Value(Value::Text(b"2")),
                Token::Value(Value::Text(b"it''s ok")),
                Token::Value(Value::Text(b"3")),
                Token::Value(Value::Unknown),
                Token::Value(Value::Text(b"4")),
                Token::Value(Value::Inapplicable),
                Token::Value(Value::Text(b"line one\nline two")),
                Token::SaveFrame(b"frame"),
                Token::SaveFrame(b""),
                Token::Reserved,
            ]
        );
    }

    #[test]
    fn quote_inside_word_and_crlf() {
        let t = tokens("a'b 'x'y' \"q\"\r\n;t\r\n;\r\nz");
        assert_eq!(
            t,
            [
                Token::Value(Value::Text(b"a'b")),
                Token::Value(Value::Text(b"x'y")),
                Token::Value(Value::Text(b"q")),
                Token::Value(Value::Text(b"t")),
                Token::Value(Value::Text(b"z")),
            ]
        );
        // A quoted '?' is a literal string, not "unknown".
        assert_eq!(tokens("'?'"), [Token::Value(Value::Text(b"?"))]);
    }

    #[test]
    fn positions() {
        let spans: Vec<(usize, usize)> = Lexer::new(b"data_x\n  loop_\n;a\n;\n  v")
            .map(|t| {
                let t = t.unwrap();
                (t.line, t.column)
            })
            .collect();
        assert_eq!(spans, [(1, 1), (2, 3), (3, 1), (5, 3)]);
    }

    #[test]
    fn errors() {
        let e = Lexer::new(b"x 'abc\n").find_map(Result::err).unwrap();
        assert_eq!((e.line, e.column), (1, 3));
        let e = Lexer::new(b"\n;never closed\n")
            .find_map(Result::err)
            .unwrap();
        assert_eq!((e.line, e.column), (2, 1));
    }
}
