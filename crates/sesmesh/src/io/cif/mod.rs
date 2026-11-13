// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PDBx/mmCIF reader: the `_atom_site` category of one data block (Spec
//! FR-IN-2, Design §5.4).
//!
//! The data block is the one named in the options, or else the first block
//! that contains `_atom_site`. Columns may appear in any order; `auth_*`
//! identifiers are preferred for names and numbering (they match the PDB
//! format), and `label_asym_id` / `label_seq_id` are kept alongside.

pub(crate) mod lexer;

use lexer::{Lexer, Spanned, Token};

use super::Parsed;
use super::num::{parse_f32, parse_f64, parse_i32, show};
use crate::diag::{Diagnostics, WarningKind};
use crate::error::ParseError;
use crate::structure::{AtomRecord, Element, Model, StringPool};

/// `_atom_site` items used by the reader.
#[derive(Debug, Default)]
struct Columns {
    group_pdb: Option<usize>,
    id: Option<usize>,
    type_symbol: Option<usize>,
    label_atom_id: Option<usize>,
    auth_atom_id: Option<usize>,
    label_alt_id: Option<usize>,
    label_comp_id: Option<usize>,
    auth_comp_id: Option<usize>,
    label_asym_id: Option<usize>,
    auth_asym_id: Option<usize>,
    label_seq_id: Option<usize>,
    auth_seq_id: Option<usize>,
    ins_code: Option<usize>,
    x: Option<usize>,
    y: Option<usize>,
    z: Option<usize>,
    occupancy: Option<usize>,
    b_iso: Option<usize>,
    formal_charge: Option<usize>,
    model: Option<usize>,
}

impl Columns {
    fn assign(&mut self, item: &[u8], index: usize) {
        let slot = match item.to_ascii_lowercase().as_slice() {
            b"group_pdb" => &mut self.group_pdb,
            b"id" => &mut self.id,
            b"type_symbol" => &mut self.type_symbol,
            b"label_atom_id" => &mut self.label_atom_id,
            b"auth_atom_id" => &mut self.auth_atom_id,
            b"label_alt_id" => &mut self.label_alt_id,
            b"label_comp_id" => &mut self.label_comp_id,
            b"auth_comp_id" => &mut self.auth_comp_id,
            b"label_asym_id" => &mut self.label_asym_id,
            b"auth_asym_id" => &mut self.auth_asym_id,
            b"label_seq_id" => &mut self.label_seq_id,
            b"auth_seq_id" => &mut self.auth_seq_id,
            b"pdbx_pdb_ins_code" => &mut self.ins_code,
            b"cartn_x" => &mut self.x,
            b"cartn_y" => &mut self.y,
            b"cartn_z" => &mut self.z,
            b"occupancy" => &mut self.occupancy,
            b"b_iso_or_equiv" => &mut self.b_iso,
            b"pdbx_formal_charge" => &mut self.formal_charge,
            b"pdbx_pdb_model_num" => &mut self.model,
            _ => return,
        };
        slot.get_or_insert(index);
    }
}

const ATOM_SITE: &[u8] = b"_atom_site.";

fn atom_site_item(tag: &[u8]) -> Option<&[u8]> {
    (tag.len() > ATOM_SITE.len() && tag[..ATOM_SITE.len()].eq_ignore_ascii_case(ATOM_SITE))
        .then(|| &tag[ATOM_SITE.len()..])
}

pub(crate) fn parse(data: &[u8], block: Option<&str>) -> Result<Parsed, ParseError> {
    let mut lexer = Lexer::new(data).peekable();
    let mut current_block: Option<&[u8]> = None;
    let mut blocks_seen = 0usize;

    while let Some(item) = lexer.next() {
        let Spanned {
            token,
            line,
            column,
        } = item?;
        match token {
            Token::DataBlock(name) => {
                current_block = Some(name);
                blocks_seen += 1;
            }
            _ if !block_matches(current_block, block) => {}
            Token::Loop => {
                let mut tags: Vec<&[u8]> = Vec::new();
                while let Some(Ok(Spanned {
                    token: Token::Tag(tag),
                    ..
                })) = lexer.peek()
                {
                    tags.push(tag);
                    lexer.next();
                }
                if tags.first().is_some_and(|t| atom_site_item(t).is_some()) {
                    let name = current_block.map(|b| String::from_utf8_lossy(b).into_owned());
                    return read_loop(&mut lexer, &tags, (line, column), name);
                }
            }
            Token::Tag(tag) if atom_site_item(tag).is_some() => {
                // Single atom written as key-value pairs.
                let mut tags = vec![tag];
                let mut values = Vec::new();
                values.push(next_value(&mut lexer, tag, line)?);
                while let Some(Ok(Spanned {
                    token: Token::Tag(t),
                    ..
                })) = lexer.peek()
                {
                    let t = *t;
                    if atom_site_item(t).is_none() {
                        break;
                    }
                    lexer.next();
                    tags.push(t);
                    values.push(next_value(&mut lexer, t, line)?);
                }
                let name = current_block.map(|b| String::from_utf8_lossy(b).into_owned());
                let mut reader = RowReader::new(&tags, (line, column))?;
                reader.add_row(&values)?;
                return reader.finish(name);
            }
            _ => {}
        }
    }
    Err(match block {
        Some(b) if blocks_seen > 0 => ParseError::new(
            1,
            1,
            format!("no data block '{b}' with an _atom_site category"),
        ),
        _ => ParseError::new(1, 1, "no _atom_site category found"),
    })
}

fn block_matches(current: Option<&[u8]>, wanted: Option<&str>) -> bool {
    match (current, wanted) {
        (_, None) => true,
        (Some(c), Some(w)) => c.eq_ignore_ascii_case(w.as_bytes()),
        (None, Some(_)) => false,
    }
}

type Tokens<'a> = std::iter::Peekable<Lexer<'a>>;

fn next_value<'a>(
    lexer: &mut Tokens<'a>,
    tag: &[u8],
    line: usize,
) -> Result<Spanned<'a>, ParseError> {
    match lexer.next() {
        Some(Ok(
            s @ Spanned {
                token: Token::Value(_),
                ..
            },
        )) => Ok(s),
        Some(Err(e)) => Err(e),
        _ => Err(ParseError::new(
            line,
            1,
            format!("tag {} has no value", String::from_utf8_lossy(tag)),
        )),
    }
}

fn read_loop<'a>(
    lexer: &mut Tokens<'a>,
    tags: &[&[u8]],
    loop_pos: (usize, usize),
    block_name: Option<String>,
) -> Result<Parsed, ParseError> {
    let mut reader = RowReader::new(tags, loop_pos)?;
    let mut row: Vec<Spanned<'a>> = Vec::with_capacity(tags.len());
    let mut last = loop_pos;
    while let Some(Ok(Spanned {
        token: Token::Value(_),
        ..
    })) = lexer.peek()
    {
        let value = lexer.next().expect("peeked")?;
        last = (value.line, value.column);
        row.push(value);
        if row.len() == tags.len() {
            reader.add_row(&row)?;
            row.clear();
        }
    }
    if let Some(Err(e)) = lexer.peek() {
        return Err(e.clone());
    }
    if !row.is_empty() {
        return Err(ParseError::new(
            last.0,
            last.1,
            format!(
                "_atom_site loop has {} columns but the last row has {} values",
                tags.len(),
                row.len()
            ),
        ));
    }
    reader.finish(block_name)
}

/// Converts `_atom_site` rows into atom records.
struct RowReader {
    cols: Columns,
    strings: StringPool,
    diagnostics: Diagnostics,
    models: Vec<Model>,
    /// Index into `models` of the most recently used model.
    last_model: usize,
    loop_pos: (usize, usize),
}

impl RowReader {
    fn new(tags: &[&[u8]], loop_pos: (usize, usize)) -> Result<Self, ParseError> {
        let mut cols = Columns::default();
        for (i, tag) in tags.iter().enumerate() {
            if let Some(item) = atom_site_item(tag) {
                cols.assign(item, i);
            }
        }
        for (col, name) in [
            (cols.x, "Cartn_x"),
            (cols.y, "Cartn_y"),
            (cols.z, "Cartn_z"),
        ] {
            if col.is_none() {
                return Err(ParseError::new(
                    loop_pos.0,
                    loop_pos.1,
                    format!("_atom_site has no _atom_site.{name} item"),
                ));
            }
        }
        Ok(Self {
            cols,
            strings: StringPool::new(),
            diagnostics: Diagnostics::new(),
            models: Vec::new(),
            last_model: 0,
            loop_pos,
        })
    }

    fn add_row(&mut self, row: &[Spanned<'_>]) -> Result<(), ParseError> {
        let get = |col: Option<usize>| -> Option<&[u8]> {
            match row[col?].token {
                Token::Value(v) => v.text(),
                _ => None,
            }
        };
        let coord = |col: Option<usize>, axis: &str| -> Result<f64, ParseError> {
            let s = &row[col.expect("checked in new")];
            let text = match s.token {
                Token::Value(v) => v.text().unwrap_or(b"?"),
                _ => b"",
            };
            parse_f64(text).ok_or_else(|| {
                ParseError::new(
                    s.line,
                    s.column,
                    format!("invalid {axis} coordinate '{}'", show(text)),
                )
            })
        };
        let (line, _) = (row[0].line, row[0].column);

        let mut atom = AtomRecord::new([
            coord(self.cols.x, "x")?,
            coord(self.cols.y, "y")?,
            coord(self.cols.z, "z")?,
        ]);
        atom.het = get(self.cols.group_pdb).is_some_and(|g| g.eq_ignore_ascii_case(b"HETATM"));
        if let Some(id) = get(self.cols.id) {
            match super::num::parse_i64(id).and_then(|v| u32::try_from(v).ok()) {
                Some(v) => atom.serial = v,
                None => self.diagnostics.warn(
                    WarningKind::InvalidField,
                    "invalid _atom_site.id; using 0",
                    || format!("line {line}: '{}'", show(id)),
                ),
            }
        }
        let name = get(self.cols.auth_atom_id)
            .or_else(|| get(self.cols.label_atom_id))
            .unwrap_or(b"");
        atom.name = self.strings.intern_bytes(name);
        atom.alt_loc = get(self.cols.label_alt_id)
            .and_then(|a| a.first())
            .map(|&c| char::from(c));
        let res_name = get(self.cols.auth_comp_id)
            .or_else(|| get(self.cols.label_comp_id))
            .unwrap_or(b"");
        atom.res_name = self.strings.intern_bytes(res_name);
        let label_chain = get(self.cols.label_asym_id);
        let chain = get(self.cols.auth_asym_id).or(label_chain).unwrap_or(b"");
        atom.chain = self.strings.intern_bytes(chain);
        atom.label_chain = match label_chain {
            Some(l) => self.strings.intern_bytes(l),
            None => atom.chain,
        };
        atom.label_seq = get(self.cols.label_seq_id).and_then(parse_i32);
        let seq = get(self.cols.auth_seq_id).or_else(|| get(self.cols.label_seq_id));
        atom.res_seq = match seq {
            Some(s) => parse_i32(s).unwrap_or_else(|| {
                self.diagnostics.warn(
                    WarningKind::InvalidField,
                    "invalid residue number; using 0",
                    || format!("line {line}: '{}'", show(s)),
                );
                0
            }),
            None => 0,
        };
        atom.ins_code = get(self.cols.ins_code)
            .and_then(|c| c.first())
            .map(|&c| char::from(c));
        if let Some(occ) = get(self.cols.occupancy) {
            atom.occupancy = parse_f32(occ).unwrap_or_else(|| {
                self.diagnostics.warn(
                    WarningKind::InvalidField,
                    "invalid occupancy; using 1.0",
                    || format!("line {line}: '{}'", show(occ)),
                );
                1.0
            });
        }
        if let Some(b) = get(self.cols.b_iso) {
            atom.b_factor = parse_f32(b).unwrap_or_else(|| {
                self.diagnostics.warn(
                    WarningKind::InvalidField,
                    "invalid B-factor; using 0.0",
                    || format!("line {line}: '{}'", show(b)),
                );
                0.0
            });
        }
        if let Some(q) = get(self.cols.formal_charge) {
            atom.formal_charge = parse_i32(q)
                .and_then(|v| i8::try_from(v).ok())
                .unwrap_or_else(|| {
                    self.diagnostics.warn(
                        WarningKind::InvalidField,
                        "invalid formal charge; using 0",
                        || format!("line {line}: '{}'", show(q)),
                    );
                    0
                });
        }
        let symbol = get(self.cols.type_symbol);
        atom.element = symbol.and_then(Element::from_symbol_bytes);
        if atom.element.is_none() {
            let name_str = String::from_utf8_lossy(name);
            let res_str = String::from_utf8_lossy(res_name);
            atom.element = Element::infer_from_name(&name_str, &res_str, atom.het);
            match (atom.element, symbol) {
                (Some(e), None) => self.diagnostics.warn(
                    WarningKind::InferredElement,
                    "no type_symbol; element inferred from the atom name",
                    || format!("line {line}: '{name_str}' -> {e}"),
                ),
                (Some(e), Some(s)) => self.diagnostics.warn(
                    WarningKind::UnknownElement,
                    "unrecognized type_symbol; element inferred from the atom name",
                    || format!("line {line}: '{}' -> {e}", show(s)),
                ),
                (None, _) => self.diagnostics.warn(
                    WarningKind::UnknownElement,
                    "element could not be determined",
                    || format!("line {line}: atom '{name_str}'"),
                ),
            }
        }

        let model_number = match get(self.cols.model) {
            Some(m) => parse_i32(m).unwrap_or_else(|| {
                self.diagnostics.warn(
                    WarningKind::InvalidField,
                    "invalid pdbx_PDB_model_num; using 1",
                    || format!("line {line}: '{}'", show(m)),
                );
                1
            }),
            None => 1,
        };
        self.model_mut(model_number).atoms.push(atom);
        Ok(())
    }

    fn model_mut(&mut self, number: i32) -> &mut Model {
        if self
            .models
            .get(self.last_model)
            .is_none_or(|m| m.number != number)
        {
            self.last_model = match self.models.iter().position(|m| m.number == number) {
                Some(i) => i,
                None => {
                    self.models.push(Model::new(number));
                    self.models.len() - 1
                }
            };
        }
        &mut self.models[self.last_model]
    }

    fn finish(self, data_block: Option<String>) -> Result<Parsed, ParseError> {
        if self.models.is_empty() {
            return Err(ParseError::new(
                self.loop_pos.0,
                self.loop_pos.1,
                "_atom_site category has no rows",
            ));
        }
        Ok(Parsed {
            models: self.models,
            strings: self.strings,
            diagnostics: self.diagnostics,
            data_block,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "\
data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N  N     . ALA A 1 ? 11.104 6.134 -6.504 1.00 0.00 ? 1   ALA AAA N     1
ATOM   2 C  \"C5'\" A ALA A 1 B 11.639 6.071 -5.147 0.60 10.00 ? 1   ALA AAA \"C5'\" 1
HETATM 3 FE FE    . HEM B . ? 1.0 2.0 3.0 1.00 20.00 2 201 HEM BBB FE    1
ATOM   4 N  N     . ALA A 1 ? 12.0 6.0 -6.0 1.00 0.00 ? 1   ALA AAA N     2
#
loop_
_other.x
1
";

    #[test]
    fn reads_atom_site() {
        let p = parse(DOC.as_bytes(), None).unwrap();
        assert_eq!(p.data_block.as_deref(), Some("TEST"));
        assert_eq!(p.models.len(), 2);
        let atoms = &p.models[0].atoms;
        assert_eq!(atoms.len(), 3);
        let a = &atoms[1];
        assert_eq!(a.serial, 2);
        assert_eq!(p.strings.get(a.name), "C5'");
        assert_eq!(a.alt_loc, Some('A'));
        assert_eq!(a.ins_code, Some('B'));
        assert_eq!(p.strings.get(a.chain), "AAA");
        assert_eq!(p.strings.get(a.label_chain), "A");
        assert_eq!(a.label_seq, Some(1));
        assert_eq!(a.position, [11.639, 6.071, -5.147]);
        assert_eq!(a.occupancy, 0.6);
        let fe = &atoms[2];
        assert!(fe.het);
        assert_eq!(fe.element, Some(Element::Fe));
        assert_eq!(fe.formal_charge, 2);
        assert_eq!(fe.res_seq, 201);
        assert_eq!(fe.label_seq, None);
        assert_eq!(p.models[1].number, 2);
        assert!(p.diagnostics.is_empty(), "{:?}", p.diagnostics);
    }

    #[test]
    fn block_selection() {
        let two = format!("data_EMPTY\n_entry.id X\n{DOC}");
        let p = parse(two.as_bytes(), None).unwrap();
        assert_eq!(p.data_block.as_deref(), Some("TEST"));
        assert!(parse(two.as_bytes(), Some("test")).is_ok());
        let err = parse(two.as_bytes(), Some("EMPTY")).unwrap_err();
        assert!(err.message.contains("EMPTY"), "{}", err.message);
    }

    #[test]
    fn key_value_atom_site() {
        let doc = "data_x\n_atom_site.id 7\n_atom_site.type_symbol O\n_atom_site.Cartn_x 1\n\
                   _atom_site.Cartn_y 2\n_atom_site.Cartn_z 3\n_cell.length_a 10\n";
        let p = parse(doc.as_bytes(), None).unwrap();
        let a = &p.models[0].atoms[0];
        assert_eq!(
            (a.serial, a.element, a.position),
            (7, Some(Element::O), [1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn errors() {
        let missing =
            "data_x\nloop_\n_atom_site.id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n1 2 3\n";
        let e = parse(missing.as_bytes(), None).unwrap_err();
        assert!(e.message.contains("Cartn_z"), "{}", e.message);
        let ragged = "data_x\nloop_\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n1 2 3\n4 5\n";
        let e = parse(ragged.as_bytes(), None).unwrap_err();
        assert_eq!(e.line, 7);
        assert!(e.message.contains("3 columns"), "{}", e.message);
        let bad =
            "data_x\nloop_\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n1 2 x\n";
        let e = parse(bad.as_bytes(), None).unwrap_err();
        assert_eq!((e.line, e.column), (6, 5));
        assert!(parse(b"data_x\n_entry.id y\n", None).is_err());
    }
}
