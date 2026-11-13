// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PDB reader: wwPDB format v3.3 fixed-column `ATOM`/`HETATM` records plus
//! `MODEL`/`ENDMDL`/`END` (Spec FR-IN-1, Design §5.3).
//!
//! Large files are split into newline-aligned chunks parsed in parallel
//! (feature `parallel`); results are merged in file order, so output does not
//! depend on the number of threads.

use memchr::memchr;

use super::Parsed;
use super::hybrid36;
use super::model::ModelBuilder;
use super::num::{parse_f32, parse_f64, parse_i32, show};
use crate::diag::{Diagnostics, WarningKind};
use crate::error::ParseError;
use crate::structure::{AtomRecord, Element, StringPool, Sym};

/// Files at least this large are parsed in parallel chunks (feature
/// `parallel`).
const PARALLEL_MIN_BYTES: usize = 8 << 20;
/// Target chunk size for parallel parsing.
const CHUNK_BYTES: usize = 4 << 20;

enum Event {
    Atom(AtomRecord),
    Model(Option<i32>),
    EndModel,
}

struct Chunk {
    events: Vec<Event>,
    strings: StringPool,
    diagnostics: Diagnostics,
}

pub(crate) fn parse(data: &[u8]) -> Result<Parsed, ParseError> {
    parse_with(data, PARALLEL_MIN_BYTES, CHUNK_BYTES)
}

fn parse_with(data: &[u8], parallel_min: usize, chunk_bytes: usize) -> Result<Parsed, ParseError> {
    let mut strings = StringPool::new();
    let mut diagnostics = Diagnostics::new();
    let mut models = ModelBuilder::default();
    for chunk in parse_chunks(data, parallel_min, chunk_bytes)? {
        let remap: Vec<Sym> = chunk
            .strings
            .iter()
            .map(|(_, s)| strings.intern(s))
            .collect();
        diagnostics.merge(chunk.diagnostics);
        for event in chunk.events {
            match event {
                Event::Atom(mut atom) => {
                    atom.name = remap[atom.name.index()];
                    atom.res_name = remap[atom.res_name.index()];
                    atom.chain = remap[atom.chain.index()];
                    atom.label_chain = atom.chain;
                    models.push(atom);
                }
                Event::Model(number) => models.begin(number),
                Event::EndModel => models.end(),
            }
        }
    }
    let models = models.finish();
    if models.is_empty() {
        return Err(ParseError::new(1, 1, "no ATOM or HETATM records found"));
    }
    Ok(Parsed {
        models,
        strings,
        diagnostics,
        data_block: None,
    })
}

fn parse_chunks(
    data: &[u8],
    parallel_min: usize,
    chunk_bytes: usize,
) -> Result<Vec<Chunk>, ParseError> {
    #[cfg(not(feature = "parallel"))]
    let _ = (parallel_min, chunk_bytes);
    #[cfg(feature = "parallel")]
    if data.len() >= parallel_min {
        use rayon::prelude::*;
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < data.len() {
            let mut end = (start + chunk_bytes).min(data.len());
            if end < data.len() {
                end = memchr(b'\n', &data[end..]).map_or(data.len(), |i| end + i + 1);
            }
            ranges.push((start, end));
            start = end;
        }
        let mut jobs = Vec::with_capacity(ranges.len());
        let mut line = 1;
        for (s, e) in ranges {
            jobs.push((s, e, line));
            line += memchr::memchr_iter(b'\n', &data[s..e]).count();
        }
        let results: Vec<Result<Chunk, ParseError>> = jobs
            .par_iter()
            .map(|&(s, e, line)| parse_chunk(&data[s..e], line))
            .collect();
        // Report the first error in file order (deterministic).
        return results.into_iter().collect();
    }
    Ok(vec![parse_chunk(data, 1)?])
}

fn parse_chunk(data: &[u8], first_line: usize) -> Result<Chunk, ParseError> {
    let mut chunk = Chunk {
        events: Vec::new(),
        strings: StringPool::new(),
        diagnostics: Diagnostics::new(),
    };
    let mut lineno = first_line;
    let mut pos = 0;
    while pos < data.len() {
        let end = memchr(b'\n', &data[pos..]).map_or(data.len(), |i| pos + i);
        let mut line = &data[pos..end];
        if let [rest @ .., b'\r'] = line {
            line = rest;
        }
        process_line(line, lineno, &mut chunk)?;
        pos = end + 1;
        lineno += 1;
    }
    Ok(chunk)
}

fn process_line(line: &[u8], lineno: usize, chunk: &mut Chunk) -> Result<(), ParseError> {
    let atom_record =
        line.starts_with(b"ATOM") && line.get(4).is_none_or(|&c| c == b' ' || c.is_ascii_digit());
    if atom_record || line.starts_with(b"HETATM") {
        let atom = parse_atom(
            line,
            lineno,
            !atom_record,
            &mut chunk.strings,
            &mut chunk.diagnostics,
        )?;
        chunk.events.push(Event::Atom(atom));
    } else if line.starts_with(b"ENDMDL") || is_end_record(line) {
        chunk.events.push(Event::EndModel);
    } else if line.starts_with(b"MODEL") {
        let number = parse_i32(&line[5..]);
        if number.is_none() {
            chunk.diagnostics.warn(
                WarningKind::InvalidField,
                "invalid MODEL number; numbering sequentially",
                || format!("line {lineno}"),
            );
        }
        chunk.events.push(Event::Model(number));
    }
    Ok(())
}

fn is_end_record(line: &[u8]) -> bool {
    line.trim_ascii_end() == b"END" || line.starts_with(b"END ")
}

fn parse_atom(
    line: &[u8],
    lineno: usize,
    het: bool,
    strings: &mut StringPool,
    diags: &mut Diagnostics,
) -> Result<AtomRecord, ParseError> {
    let record = if het { "HETATM" } else { "ATOM" };
    if line.len() < 54 {
        return Err(ParseError::new(
            lineno,
            line.len() + 1,
            format!(
                "{record} record is {} columns long; coordinates need 54",
                line.len()
            ),
        ));
    }
    let col = |a: usize, b: usize| &line[a.min(line.len())..b.min(line.len())];
    let coord = |start: usize, axis: &str| {
        let field = &line[start..start + 8];
        parse_f64(field).ok_or_else(|| {
            ParseError::new(
                lineno,
                start + 1,
                format!("invalid {axis} coordinate '{}'", show(field)),
            )
        })
    };
    let mut atom = AtomRecord::new([coord(30, "x")?, coord(38, "y")?, coord(46, "z")?]);
    atom.het = het;

    let serial = col(6, 11);
    atom.serial = match hybrid36::decode(5, serial).map(u32::try_from) {
        Some(Ok(v)) => v,
        _ => {
            if !serial.trim_ascii().is_empty() {
                diags.warn(
                    WarningKind::InvalidField,
                    "invalid atom serial number; using 0",
                    || format!("line {lineno}: '{}'", show(serial)),
                );
            }
            0
        }
    };

    let name_field = col(12, 16);
    atom.name = strings.intern_bytes(name_field.trim_ascii());
    atom.alt_loc = char_field(line.get(16));
    let res_name = col(17, 21).trim_ascii();
    atom.res_name = strings.intern_bytes(res_name);
    atom.chain = strings.intern_bytes(col(21, 22).trim_ascii());
    atom.label_chain = atom.chain;

    let seq = col(22, 26);
    atom.res_seq = match hybrid36::decode(4, seq).map(i32::try_from) {
        Some(Ok(v)) => v,
        _ => {
            diags.warn(
                WarningKind::InvalidField,
                "missing or invalid residue number; using 0",
                || format!("line {lineno}: '{}'", show(seq)),
            );
            0
        }
    };
    atom.ins_code = char_field(line.get(26));

    let occupancy = col(54, 60);
    if !occupancy.trim_ascii().is_empty() {
        atom.occupancy = parse_f32(occupancy).unwrap_or_else(|| {
            diags.warn(
                WarningKind::InvalidField,
                "invalid occupancy; using 1.0",
                || format!("line {lineno}: '{}'", show(occupancy)),
            );
            1.0
        });
    }
    let b_factor = col(60, 66);
    if !b_factor.trim_ascii().is_empty() {
        atom.b_factor = parse_f32(b_factor).unwrap_or_else(|| {
            diags.warn(
                WarningKind::InvalidField,
                "invalid temperature factor; using 0.0",
                || format!("line {lineno}: '{}'", show(b_factor)),
            );
            0.0
        });
    }

    let element = col(76, 78).trim_ascii();
    atom.element = if element.is_empty() {
        None
    } else {
        let parsed = Element::from_symbol_bytes(element);
        if parsed.is_none() {
            diags.warn(
                WarningKind::UnknownElement,
                "unrecognized element symbol; inferring the element from the atom name",
                || format!("line {lineno}: '{}'", show(element)),
            );
        }
        parsed
    };
    if atom.element.is_none() {
        let mut padded = [b' '; 4];
        padded[..name_field.len()].copy_from_slice(name_field);
        let res = std::str::from_utf8(res_name).unwrap_or("");
        atom.element = Element::infer_from_pdb_name(&padded, res, het);
        match atom.element {
            Some(e) if element.is_empty() => diags.warn(
                WarningKind::InferredElement,
                "element columns 77-78 blank; element inferred from the atom name",
                || format!("line {lineno}: '{}' -> {e}", show(name_field)),
            ),
            Some(_) => {}
            None => diags.warn(
                WarningKind::UnknownElement,
                "element could not be determined",
                || format!("line {lineno}: atom '{}'", show(name_field)),
            ),
        }
    }

    let charge = col(78, 80);
    atom.formal_charge = parse_charge(charge).unwrap_or_else(|| {
        diags.warn(
            WarningKind::InvalidField,
            "invalid formal charge; using 0",
            || format!("line {lineno}: '{}'", show(charge)),
        );
        0
    });
    Ok(atom)
}

fn char_field(byte: Option<&u8>) -> Option<char> {
    match byte {
        Some(&c) if c.is_ascii_graphic() => Some(char::from(c)),
        _ => None,
    }
}

/// PDB charges are written `2+`/`1-` (also accepted: `+2`, `-1`, blank).
fn parse_charge(field: &[u8]) -> Option<i8> {
    let digit_sign = |d: u8, s: u8| -> Option<i8> {
        let v = i8::try_from(d.checked_sub(b'0').filter(|v| *v <= 9)?).ok()?;
        match s {
            b'+' => Some(v),
            b'-' => Some(-v),
            _ => None,
        }
    };
    match field.trim_ascii() {
        [] => Some(0),
        [d, s] if d.is_ascii_digit() => digit_sign(*d, *s),
        [s, d] if d.is_ascii_digit() => digit_sign(*d, *s),
        [d] if d.is_ascii_digit() => Some((*d - b'0') as i8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINES: &str = "\
HEADER    TEST
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA AALA A   1      11.639   6.071  -5.147  0.60 10.00           C
ATOM      3  CA BALA A   1      11.700   6.000  -5.100  0.40 10.00           C
HETATM    4 FE   HEM A 201       1.000   2.000   3.000  1.00 20.00          FE2+
TER
END
";

    fn parse_str(s: &str) -> Result<Parsed, ParseError> {
        parse(s.as_bytes())
    }

    #[test]
    fn parses_fields() {
        let p = parse_str(LINES).unwrap();
        assert_eq!(p.models.len(), 1);
        let atoms = &p.models[0].atoms;
        assert_eq!(atoms.len(), 4);
        let a = &atoms[1];
        assert_eq!(a.serial, 2);
        assert_eq!(p.strings.get(a.name), "CA");
        assert_eq!(a.alt_loc, Some('A'));
        assert_eq!(p.strings.get(a.res_name), "ALA");
        assert_eq!(p.strings.get(a.chain), "A");
        assert_eq!(a.res_seq, 1);
        assert_eq!(a.position, [11.639, 6.071, -5.147]);
        assert_eq!(a.occupancy, 0.6);
        assert_eq!(a.b_factor, 10.0);
        assert_eq!(a.element, Some(Element::C));
        assert!(!a.het);
        let fe = &atoms[3];
        assert!(fe.het);
        assert_eq!(fe.element, Some(Element::Fe));
        assert_eq!(fe.formal_charge, 2);
        assert_eq!(fe.res_seq, 201);
        assert!(p.diagnostics.is_empty(), "{:?}", p.diagnostics);
    }

    #[test]
    fn crlf_short_lines_models_and_hybrid36() {
        let text = "MODEL        1\r\n\
ATOM  A0000  CA  GLY BA000      -1.000   2.000   3.000\r\n\
ENDMDL\r\n\
MODEL        2\r\n\
ATOM      1  CA  GLY B   1      -1.500   2.000   3.000\r\n\
ENDMDL\r\n";
        let p = parse_str(text).unwrap();
        assert_eq!(p.models.len(), 2);
        assert_eq!(p.models[1].number, 2);
        let a = &p.models[0].atoms[0];
        assert_eq!(a.serial, 100_000);
        assert_eq!(a.res_seq, 10_000);
        assert_eq!(a.occupancy, 1.0);
        // Element inferred from " CA " -> carbon.
        assert_eq!(a.element, Some(Element::C));
        assert_eq!(p.diagnostics.count(WarningKind::InferredElement), 2);
    }

    #[test]
    fn end_separates_concatenated_frames() {
        let atom =
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00  0.00           C\n";
        let p = parse_str(&format!("{atom}END\n{atom}END\n")).unwrap();
        assert_eq!(p.models.len(), 2);
        assert_eq!(p.models[1].number, 2);
    }

    #[test]
    fn errors_have_positions() {
        let err = parse_str("ATOM      1  CA  GLY A   1       1.000   2.000\n").unwrap_err();
        assert_eq!((err.line, err.column), (1, 47));
        let err = parse_str("REMARK\nATOM      1  CA  GLY A   1       1.0x0   2.000   3.000\n")
            .unwrap_err();
        assert_eq!((err.line, err.column), (2, 31));
        assert!(err.message.contains("x coordinate"), "{}", err.message);
        assert!(parse_str("HEADER only\n").is_err());
    }

    #[test]
    fn tolerant_fields_warn() {
        let line =
            "ATOM  *****  CA  GLY A   1       1.000   2.000   3.000  x.xx  0.00          Qq9z\n";
        let p = parse_str(line).unwrap();
        let a = &p.models[0].atoms[0];
        assert_eq!(a.serial, 0);
        assert_eq!(a.occupancy, 1.0);
        assert_eq!(a.element, Some(Element::C));
        assert_eq!(a.formal_charge, 0);
        assert_eq!(p.diagnostics.count(WarningKind::InvalidField), 3);
        assert_eq!(p.diagnostics.count(WarningKind::UnknownElement), 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn chunked_parallel_parse_matches_sequential() {
        let mut text = String::new();
        for m in 1..=3 {
            text += &format!("MODEL     {m:>4}\n");
            for i in 0..50 {
                text += &format!(
                    "ATOM  {:>5}  CA  {} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00           C\n",
                    i + 1,
                    ["GLY", "ALA", "SER"][i % 3],
                    i + 1,
                    i as f64,
                    f64::from(m),
                    0.5
                );
            }
            text += "ENDMDL\n";
        }
        let flatten = |p: &Parsed| -> Vec<(i32, String, String, [f64; 3], u32)> {
            p.models
                .iter()
                .flat_map(|m| {
                    m.atoms.iter().map(move |a| {
                        (
                            m.number,
                            p.strings.get(a.name).to_owned(),
                            p.strings.get(a.res_name).to_owned(),
                            a.position,
                            a.serial,
                        )
                    })
                })
                .collect()
        };
        let sequential = parse_with(text.as_bytes(), usize::MAX, 0).unwrap();
        assert_eq!(sequential.models.len(), 3);
        for chunk in [0, 1, 37, 200, 1000] {
            let parallel = parse_with(text.as_bytes(), 0, chunk).unwrap();
            assert_eq!(flatten(&parallel), flatten(&sequential), "chunk {chunk}");
        }
        // Errors in later chunks report file-global line numbers.
        let bad_line = text.lines().count() + 1;
        let bad = format!("{text}ATOM      1  CA  GLY A   1       1.0x0   2.000   3.000\n");
        for chunk in [0, 37, 200] {
            assert_eq!(
                parse_with(bad.as_bytes(), 0, chunk).unwrap_err().line,
                bad_line
            );
        }
    }

    #[test]
    fn charges() {
        assert_eq!(parse_charge(b"2+"), Some(2));
        assert_eq!(parse_charge(b"1-"), Some(-1));
        assert_eq!(parse_charge(b"-1"), Some(-1));
        assert_eq!(parse_charge(b"  "), Some(0));
        assert_eq!(parse_charge(b"x+"), None);
    }
}
