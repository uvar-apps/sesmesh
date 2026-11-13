// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! XYZ and extended XYZ reader (Spec FR-IN-3/4, Design §5.5).
//!
//! Each frame is `N`, a comment line, then `N` lines `species x y z [...]`.
//! Frames become models numbered from 1. For extended XYZ the comment line's
//! `Properties=name:type:ncols:...` descriptor locates the species and
//! position columns, and optionally a per-atom radius column chosen by name.

use super::Parsed;
use super::lines::{Lines, column_of, fields};
use super::num::{parse_f32, parse_f64, parse_i64, show};
use crate::diag::{Diagnostics, WarningKind};
use crate::error::ParseError;
use crate::structure::{AtomRecord, Element, Model, StringPool};

/// Column layout of the atom lines.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Layout {
    species: usize,
    /// `true` if the species column holds atomic numbers (`Z:I:1`).
    atomic_numbers: bool,
    pos: usize,
    radius: Option<usize>,
    min_columns: usize,
}

impl Layout {
    const PLAIN: Layout = Layout {
        species: 0,
        atomic_numbers: false,
        pos: 1,
        radius: None,
        min_columns: 4,
    };

    fn from_comment(comment: &[u8], radius_column: Option<&str>) -> Result<Self, String> {
        let Some(descriptor) = properties_value(comment) else {
            return match radius_column {
                Some(name) => Err(format!(
                    "radius column '{name}' requested but the comment line has no \
                     extended-XYZ Properties= descriptor"
                )),
                None => Ok(Self::PLAIN),
            };
        };
        let parts: Vec<&str> = descriptor.split(':').collect();
        if !parts.len().is_multiple_of(3) {
            return Err(format!("malformed Properties descriptor '{descriptor}'"));
        }
        let (mut species, mut pos, mut radius) = (None, None, None);
        let mut column = 0usize;
        for triplet in parts.chunks(3) {
            let (name, kind) = (triplet[0], triplet[1]);
            let n: usize = triplet[2]
                .parse()
                .map_err(|_| format!("invalid column count in Properties entry '{name}'"))?;
            if (name.eq_ignore_ascii_case("species") && n == 1)
                || (name == "Z" && kind.eq_ignore_ascii_case("I") && n == 1)
            {
                species.get_or_insert((column, name == "Z"));
            } else if name.eq_ignore_ascii_case("pos") && n == 3 {
                pos.get_or_insert(column);
            }
            if radius_column.is_some_and(|r| r == name) {
                if n != 1 || !kind.eq_ignore_ascii_case("R") {
                    return Err(format!(
                        "radius column '{name}' must have type R and 1 column"
                    ));
                }
                radius = Some(column);
            }
            column += n;
        }
        let (species, atomic_numbers) =
            species.ok_or("Properties descriptor has no species (or Z) column")?;
        let pos = pos.ok_or("Properties descriptor has no pos:R:3 column")?;
        if let (Some(name), None) = (radius_column, radius) {
            return Err(format!(
                "Properties descriptor has no column named '{name}'"
            ));
        }
        Ok(Layout {
            species,
            atomic_numbers,
            pos,
            radius,
            min_columns: column,
        })
    }
}

/// Value of the `Properties=` key in an extended-XYZ comment line.
fn properties_value(comment: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(comment);
    let lower = text.to_ascii_lowercase();
    let mut search = 0;
    while let Some(found) = lower[search..].find("properties=") {
        let start = search + found;
        let at_word_start = start == 0 || text.as_bytes()[start - 1].is_ascii_whitespace();
        let value_start = start + "properties=".len();
        if at_word_start {
            let rest = &text[value_start..];
            let value = match rest.strip_prefix('"') {
                Some(quoted) => quoted.split('"').next().unwrap_or(""),
                None => rest.split(char::is_whitespace).next().unwrap_or(""),
            };
            return Some(value.to_owned());
        }
        search = value_start;
    }
    None
}

/// Element from an XYZ species token: a symbol optionally followed by a
/// label (`C`, `Ca`, `C12`), or an atomic number.
fn species_element(token: &[u8], atomic_number: bool) -> Option<Element> {
    if atomic_number || token.iter().all(u8::is_ascii_digit) {
        return parse_i64(token)
            .and_then(|z| u8::try_from(z).ok())
            .and_then(Element::from_atomic_number);
    }
    let letters: Vec<u8> = token
        .iter()
        .copied()
        .take_while(u8::is_ascii_alphabetic)
        .take(2)
        .collect();
    Element::from_symbol_bytes(&letters).or_else(|| Element::from_symbol_bytes(letters.get(..1)?))
}

pub(crate) fn parse(data: &[u8], radius_column: Option<&str>) -> Result<Parsed, ParseError> {
    let mut lines = Lines::new(data);
    let mut strings = StringPool::new();
    let mut diagnostics = Diagnostics::new();
    let mut models: Vec<Model> = Vec::new();
    let mut tokens: Vec<&[u8]> = Vec::new();

    while let Some((count_line, count)) = lines.by_ref().find(|(_, l)| !l.trim_ascii().is_empty()) {
        let n = parse_i64(count)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| {
                ParseError::new(
                    count_line,
                    1,
                    format!("expected the number of atoms, found '{}'", show(count)),
                )
            })?;
        let (comment_line, comment) = lines.next().ok_or_else(|| {
            ParseError::new(
                count_line + 1,
                1,
                "missing comment line after the atom count",
            )
        })?;
        let layout = Layout::from_comment(comment, radius_column)
            .map_err(|m| ParseError::new(comment_line, 1, m))?;
        let mut model = Model::new(i32::try_from(models.len() + 1).unwrap_or(i32::MAX));
        // `n` is untrusted: never reserve more than the input could hold
        // (an atom line has at least 8 bytes, e.g. "C 0 0 0\n").
        model.atoms.reserve(n.min(data.len() / 8 + 1));
        for k in 0..n {
            let (lineno, line) = lines.next().ok_or_else(|| {
                ParseError::new(
                    lines.line() + 1,
                    1,
                    format!("frame {} ends after {k} of {n} atoms", models.len() + 1),
                )
            })?;
            tokens.clear();
            tokens.extend(fields(line));
            if tokens.len() < layout.min_columns {
                return Err(ParseError::new(
                    lineno,
                    line.len() + 1,
                    format!(
                        "expected at least {} columns, found {}",
                        layout.min_columns,
                        tokens.len()
                    ),
                ));
            }
            let coord = |i: usize| {
                let t = tokens[layout.pos + i];
                parse_f64(t).ok_or_else(|| {
                    ParseError::new(
                        lineno,
                        column_of(line, t),
                        format!("invalid coordinate '{}'", show(t)),
                    )
                })
            };
            let mut atom = AtomRecord::new([coord(0)?, coord(1)?, coord(2)?]);
            atom.serial = u32::try_from(k + 1).unwrap_or(u32::MAX);
            let species = tokens[layout.species];
            atom.name = strings.intern_bytes(species);
            atom.element = species_element(species, layout.atomic_numbers);
            if atom.element.is_none() {
                diagnostics.warn(WarningKind::UnknownElement, "unrecognized species", || {
                    format!("line {lineno}: '{}'", show(species))
                });
            }
            if let Some(col) = layout.radius {
                let t = tokens[col];
                let r = parse_f32(t).filter(|r| *r >= 0.0).ok_or_else(|| {
                    ParseError::new(
                        lineno,
                        column_of(line, t),
                        format!("invalid radius '{}'", show(t)),
                    )
                })?;
                atom.radius = Some(r);
            }
            model.atoms.push(atom);
        }
        models.push(model);
    }
    if models.iter().all(|m| m.atoms.is_empty()) {
        return Err(ParseError::new(1, 1, "no atoms found"));
    }
    models.retain(|m| !m.atoms.is_empty());
    Ok(Parsed {
        models,
        strings,
        diagnostics,
        data_block: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_multi_frame() {
        let text = "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0 0\nH -0.24 0.93 0\n\n2\nsecond\nC1 1 2 3\n17 4 5 6 extra\n";
        let p = parse(text.as_bytes(), None).unwrap();
        assert_eq!(p.models.len(), 2);
        assert_eq!(p.models[0].atoms.len(), 3);
        assert_eq!(p.models[1].number, 2);
        let m = &p.models[1].atoms;
        assert_eq!(m[0].element, Some(Element::C));
        assert_eq!(p.strings.get(m[0].name), "C1");
        assert_eq!(m[1].element, Some(Element::Cl));
        assert_eq!(m[1].position, [4.0, 5.0, 6.0]);
        assert!(p.diagnostics.is_empty());
    }

    #[test]
    fn extended_xyz_with_radius() {
        let text = "2\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:rad:R:1 pbc=\"T T T\"\n\
                    Ca 0 0 0 1.5\nO 1 1 1 1.2\n";
        let p = parse(text.as_bytes(), Some("rad")).unwrap();
        let a = &p.models[0].atoms;
        assert_eq!(a[0].element, Some(Element::Ca));
        assert_eq!(a[0].radius, Some(1.5));
        assert_eq!(a[1].radius, Some(1.2));
        let e = parse(text.as_bytes(), Some("radius")).unwrap_err();
        assert_eq!(e.line, 2);
        let reordered = "1\nProperties=pos:R:3:Z:I:1\n1.0 2.0 3.0 8\n";
        let p = parse(reordered.as_bytes(), None).unwrap();
        assert_eq!(p.models[0].atoms[0].element, Some(Element::O));
        assert_eq!(p.models[0].atoms[0].position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn errors() {
        assert_eq!(parse(b"x\n", None).unwrap_err().line, 1);
        let e = parse(b"2\nc\nC 0 0 0\n", None).unwrap_err();
        assert_eq!(e.line, 4);
        let e = parse(b"1\nc\nC 0 zz 0\n", None).unwrap_err();
        assert_eq!((e.line, e.column), (3, 5));
        assert!(parse(b"1\nplain\nC 0 0 0\n", Some("r")).is_err());
        assert!(parse(b"\n\n", None).is_err());
        // Fuzzer regression: a huge atom count must fail fast, not allocate.
        let e = parse(b"377777773\nwater\nO 0 0 0\n", None).unwrap_err();
        assert!(
            e.message.contains("ends after 1 of 377777773"),
            "{}",
            e.message
        );
        let p = parse(b"1\nc\nXq 0 0 0\n", None).unwrap();
        assert_eq!(p.diagnostics.count(WarningKind::UnknownElement), 1);
    }
}
