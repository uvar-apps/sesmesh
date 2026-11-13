//! Biological assemblies (Spec FR-IN-8, Plan T8.6): mmCIF
//! `pdbx_struct_assembly_gen` + `pdbx_struct_oper_list`, PDB
//! `REMARK 350 BIOMT`. An assembly is expanded into one model per input
//! model, holding the transformed copies of the listed chains (mmCIF
//! `label_asym_id`, PDB chain IDs). Copies keep their chain IDs.

use rustc_hash::FxHashMap;

use super::cif::lexer::{Lexer, Token};
use crate::error::{Error, ParseError, Result};
use crate::structure::{Model, Structure};

/// A rigid transform `x ↦ R·x + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Transform {
    r: [[f64; 3]; 3],
    t: [f64; 3],
}

impl Transform {
    const IDENTITY: Transform = Transform {
        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t: [0.0; 3],
    };

    fn apply(&self, x: [f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [0, 1, 2].map(|i| r[i][0] * x[0] + r[i][1] * x[1] + r[i][2] * x[2] + self.t[i])
    }

    /// `self ∘ other` (apply `other` first).
    fn then_after(&self, other: &Transform) -> Transform {
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| self.r[i][k] * other.r[k][j]).sum();
            }
        }
        Transform {
            r,
            t: self.apply(other.t),
        }
    }
}

/// One generator: apply the operator expression to these chains.
#[derive(Debug, Clone, PartialEq)]
struct Generator {
    assembly: String,
    expression: String,
    chains: Vec<String>,
}

/// Assembly definitions of a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct AssemblyData {
    generators: Vec<Generator>,
    operators: FxHashMap<String, Transform>,
    /// Chains are mmCIF `label_asym_id`s (else author chain IDs).
    label_chains: bool,
}

/// Summary of one assembly (for listings).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AssemblySummary {
    /// Assembly ID (mmCIF `assembly_id`, PDB `BIOMOLECULE` number).
    pub id: String,
    /// Number of transformed copies of chain sets it generates.
    pub copies: usize,
    /// Chains it uses (as named in the file's assembly records).
    pub chains: Vec<String>,
}

impl AssemblyData {
    pub(crate) fn summaries(&self) -> Result<Vec<AssemblySummary>> {
        let mut out: Vec<AssemblySummary> = Vec::new();
        for g in &self.generators {
            let copies = parse_expression(&g.expression)?
                .iter()
                .map(Vec::len)
                .product::<usize>();
            match out.iter_mut().find(|s| s.id == g.assembly) {
                Some(s) => {
                    s.copies += copies;
                    for c in &g.chains {
                        if !s.chains.contains(c) {
                            s.chains.push(c.clone());
                        }
                    }
                }
                None => out.push(AssemblySummary {
                    id: g.assembly.clone(),
                    copies,
                    chains: g.chains.clone(),
                }),
            }
        }
        Ok(out)
    }
}

/// Parses an operator expression into its factors: `1` → `[[1]]`,
/// `(1-3)` → `[[1, 2, 3]]`, `(1,2)(3-5)` → `[[1, 2], [3, 4, 5]]`,
/// `(X0)(1-10,21-25)`. The copies are the Cartesian product of the
/// factors, each applied right to left.
pub(crate) fn parse_expression(expr: &str) -> Result<Vec<Vec<String>>> {
    let bad = || Error::InvalidInput(format!("invalid assembly operator expression {expr:?}"));
    let e: String = expr.chars().filter(|c| !c.is_whitespace()).collect();
    if e.is_empty() {
        return Err(bad());
    }
    let groups: Vec<&str> = if e.contains('(') {
        let mut gs = Vec::new();
        let mut rest = e.as_str();
        while !rest.is_empty() {
            let inner = rest.strip_prefix('(').ok_or_else(bad)?;
            let close = inner.find(')').ok_or_else(bad)?;
            gs.push(&inner[..close]);
            rest = &inner[close + 1..];
        }
        gs
    } else {
        vec![e.as_str()]
    };
    groups
        .into_iter()
        .map(|g| {
            let mut items = Vec::new();
            for item in g.split(',') {
                if item.is_empty() {
                    return Err(bad());
                }
                match item.split_once('-') {
                    Some((a, b)) => {
                        let (a, b): (u32, u32) =
                            (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
                        if a > b || b - a > 1_000_000 {
                            return Err(bad());
                        }
                        items.extend((a..=b).map(|i| i.to_string()));
                    }
                    None => items.push(item.to_string()),
                }
            }
            Ok(items)
        })
        .collect()
}

/// Reads the assembly categories of an mmCIF document (the data block
/// `block`, or the first one).
pub(crate) fn parse_cif(
    data: &[u8],
    block: Option<&str>,
) -> std::result::Result<AssemblyData, ParseError> {
    let mut lexer = Lexer::new(data).peekable();
    let mut current: Option<&[u8]> = None;
    let mut seen_block = false;
    // category → rows of (item → value)
    let mut tables: FxHashMap<&'static str, Vec<FxHashMap<String, String>>> = FxHashMap::default();
    const CATS: [&str; 2] = ["_pdbx_struct_assembly_gen.", "_pdbx_struct_oper_list."];
    let cat_of = |tag: &[u8]| {
        CATS.iter()
            .copied()
            .find(|c| tag.len() > c.len() && tag[..c.len()].eq_ignore_ascii_case(c.as_bytes()))
    };
    let item =
        |tag: &[u8], cat: &str| String::from_utf8_lossy(&tag[cat.len()..]).to_ascii_lowercase();
    let wanted = |current: Option<&[u8]>| match (current, block) {
        (_, None) => true,
        (Some(c), Some(w)) => c.eq_ignore_ascii_case(w.as_bytes()),
        (None, Some(_)) => false,
    };
    while let Some(tok) = lexer.next() {
        let tok = tok?;
        match tok.token {
            Token::DataBlock(name) => {
                if seen_block && block.is_none() {
                    break; // first block only
                }
                current = Some(name);
                seen_block = true;
            }
            _ if !wanted(current) => {}
            Token::Loop => {
                let mut tags: Vec<&[u8]> = Vec::new();
                while let Some(Ok(t)) = lexer.peek() {
                    let Token::Tag(tag) = t.token else { break };
                    tags.push(tag);
                    lexer.next();
                }
                let cat = tags.first().and_then(|t| cat_of(t));
                let mut row: Vec<String> = Vec::new();
                while let Some(Ok(t)) = lexer.peek() {
                    let Token::Value(v) = t.token else { break };
                    lexer.next();
                    row.push(
                        v.text()
                            .map(|b| String::from_utf8_lossy(b).into_owned())
                            .unwrap_or_default(),
                    );
                    if row.len() == tags.len() {
                        if let Some(cat) = cat {
                            let map = tags
                                .iter()
                                .zip(row.drain(..))
                                .map(|(t, v)| (item(t, cat), v))
                                .collect();
                            tables.entry(cat).or_default().push(map);
                        }
                        row.clear();
                    }
                }
            }
            Token::Tag(tag) => {
                if let Some(cat) = cat_of(tag)
                    && let Some(Ok(v)) = lexer.peek()
                    && let Token::Value(v) = v.token
                {
                    lexer.next();
                    let value = v
                        .text()
                        .map(|b| String::from_utf8_lossy(b).into_owned())
                        .unwrap_or_default();
                    let rows = tables.entry(cat).or_default();
                    if rows.is_empty() {
                        rows.push(FxHashMap::default());
                    }
                    rows[0].insert(item(tag, cat), value);
                }
            }
            _ => {}
        }
    }
    let mut out = AssemblyData {
        label_chains: true,
        ..AssemblyData::default()
    };
    for row in tables.remove(CATS[0]).unwrap_or_default() {
        let get = |k: &str| row.get(k).cloned().unwrap_or_default();
        out.generators.push(Generator {
            assembly: get("assembly_id"),
            expression: get("oper_expression"),
            chains: get("asym_id_list")
                .split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect(),
        });
    }
    for row in tables.remove(CATS[1]).unwrap_or_default() {
        let num = |k: &str| {
            row.get(k)
                .and_then(|v| fast_float2::parse::<f64, _>(v).ok())
        };
        let mut t = Transform::IDENTITY;
        let mut ok = true;
        for (i, (row_r, row_t)) in t.r.iter_mut().zip(t.t.iter_mut()).enumerate() {
            for (j, cell) in row_r.iter_mut().enumerate() {
                match num(&format!("matrix[{}][{}]", i + 1, j + 1)) {
                    Some(v) => *cell = v,
                    None => ok = false,
                }
            }
            match num(&format!("vector[{}]", i + 1)) {
                Some(v) => *row_t = v,
                None => ok = false,
            }
        }
        if let (true, Some(id)) = (ok, row.get("id")) {
            out.operators.insert(id.clone(), t);
        }
    }
    Ok(out)
}

/// Reads `REMARK 350` biomolecules of a PDB file.
pub(crate) fn parse_pdb(data: &[u8]) -> AssemblyData {
    let mut out = AssemblyData::default();
    let mut bio = String::new();
    let mut chains: Vec<String> = Vec::new();
    let mut ops: Vec<String> = Vec::new();
    // BIOMT rows (matrix row, translation) of each operator.
    type Rows = [Option<([f64; 3], f64)>; 3];
    let mut rows: FxHashMap<String, Rows> = FxHashMap::default();
    let flush =
        |out: &mut AssemblyData, bio: &str, chains: &mut Vec<String>, ops: &mut Vec<String>| {
            if !chains.is_empty() && !ops.is_empty() {
                out.generators.push(Generator {
                    assembly: bio.to_string(),
                    expression: ops.join(","),
                    chains: std::mem::take(chains),
                });
            }
            ops.clear();
        };
    for line in data.split(|&b| b == b'\n') {
        if !line.starts_with(b"REMARK 350") {
            continue;
        }
        let text = String::from_utf8_lossy(line.get(10..).unwrap_or_default())
            .trim()
            .to_string();
        if let Some(n) = text.strip_prefix("BIOMOLECULE:") {
            flush(&mut out, &bio, &mut chains, &mut ops);
            chains.clear();
            bio = n.trim().to_string();
        } else if let Some(list) = text
            .strip_prefix("APPLY THE FOLLOWING TO CHAINS:")
            .or_else(|| text.strip_prefix("AND CHAINS:"))
        {
            if text.starts_with("APPLY") {
                flush(&mut out, &bio, &mut chains, &mut ops);
                chains.clear();
            }
            chains.extend(
                list.split(',')
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty()),
            );
        } else if let Some(rest) = text.strip_prefix("BIOMT") {
            let f: Vec<&str> = rest.split_whitespace().collect();
            // BIOMTn serial m11 m12 m13 t1
            if f.len() >= 6
                && let (Ok(row), Some(m)) = (
                    f[0].parse::<usize>(),
                    f[2..6]
                        .iter()
                        .map(|v| v.parse::<f64>().ok())
                        .collect::<Option<Vec<f64>>>(),
                )
                && (1..=3).contains(&row)
            {
                let id = format!("{bio}:{}", f[1]);
                if row == 1 && !ops.contains(&id) {
                    ops.push(id.clone());
                }
                rows.entry(id).or_default()[row - 1] = Some(([m[0], m[1], m[2]], m[3]));
            }
        }
    }
    flush(&mut out, &bio, &mut chains, &mut ops);
    for (id, r) in rows {
        if let [Some(a), Some(b), Some(c)] = r {
            out.operators.insert(
                id,
                Transform {
                    r: [a.0, b.0, c.0],
                    t: [a.1, b.1, c.1],
                },
            );
        }
    }
    out
}

/// Most copies one assembly may generate (real assemblies have at most a
/// few thousand; malformed expressions could ask for 10¹²).
const MAX_COPIES: usize = 100_000;
/// Most atoms an expanded model may hold.
const MAX_ATOMS: usize = 500_000_000;

/// Replaces every model of `structure` by assembly `id`.
pub(crate) fn expand(mut structure: Structure, data: &AssemblyData, id: &str) -> Result<Structure> {
    let gens: Vec<&Generator> = data
        .generators
        .iter()
        .filter(|g| g.assembly == id)
        .collect();
    if gens.is_empty() {
        let ids: Vec<String> = data.summaries()?.into_iter().map(|s| s.id).collect();
        return Err(Error::InvalidInput(if ids.is_empty() {
            "the input defines no biological assemblies".to_string()
        } else {
            format!("no assembly {id:?}; available: {}", ids.join(", "))
        }));
    }
    // Transforms per generator: Cartesian product, applied right to left.
    let mut plans: Vec<(Vec<Transform>, &[String])> = Vec::new();
    for g in gens {
        let factors = parse_expression(&g.expression)?;
        let copies = factors
            .iter()
            .try_fold(1usize, |n, f| n.checked_mul(f.len()))
            .filter(|&n| n <= MAX_COPIES)
            .ok_or_else(|| Error::ResourceLimit {
                what: format!("assembly {id}: operator expression {:?}", g.expression),
                limit: MAX_COPIES as u64,
            })?;
        let _ = copies;
        let mut combos = vec![Transform::IDENTITY];
        for factor in factors {
            let mut next = Vec::with_capacity(combos.len() * factor.len());
            for c in &combos {
                for op in &factor {
                    let t = data.operators.get(op).ok_or_else(|| {
                        Error::InvalidInput(format!("assembly {id}: unknown operator {op:?}"))
                    })?;
                    next.push(c.then_after(t));
                }
            }
            combos = next;
        }
        plans.push((combos, &g.chains));
    }
    let strings = &structure.strings;
    for m in &structure.models {
        let atoms: usize = plans
            .iter()
            .map(|(transforms, chains)| {
                let n = m
                    .atoms
                    .iter()
                    .filter(|a| {
                        let sym = if data.label_chains {
                            a.label_chain
                        } else {
                            a.chain
                        };
                        chains.iter().any(|c| c == strings.get(sym))
                    })
                    .count();
                n.saturating_mul(transforms.len())
            })
            .fold(0usize, usize::saturating_add);
        if atoms > MAX_ATOMS {
            return Err(Error::ResourceLimit {
                what: format!("assembly {id} with {atoms} atoms"),
                limit: MAX_ATOMS as u64,
            });
        }
    }
    let models = structure
        .models
        .iter()
        .map(|m| {
            let mut out = Model::new(m.number);
            for (transforms, chains) in &plans {
                let atoms: Vec<_> = m
                    .atoms
                    .iter()
                    .filter(|a| {
                        let sym = if data.label_chains {
                            a.label_chain
                        } else {
                            a.chain
                        };
                        chains.iter().any(|c| c == strings.get(sym))
                    })
                    .collect();
                for t in transforms {
                    out.atoms.extend(atoms.iter().map(|a| {
                        let mut b = (*a).clone();
                        b.position = t.apply(a.position);
                        b
                    }));
                }
            }
            out
        })
        .collect();
    structure.models = models;
    Ok(structure)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expressions() {
        let p = |e: &str| parse_expression(e).unwrap();
        assert_eq!(p("1"), [["1"]]);
        assert_eq!(p("1,2,5"), [["1", "2", "5"]]);
        assert_eq!(p("(1-3)"), [["1", "2", "3"]]);
        assert_eq!(p("(1,2)(3-5)"), vec![vec!["1", "2"], vec!["3", "4", "5"]]);
        assert_eq!(p("(X0)(1-2,7)"), vec![vec!["X0"], vec!["1", "2", "7"]]);
        assert_eq!(p(" ( 1 - 2 ) "), [["1", "2"]]);
        for bad in ["", "(1", "(1-)", "3-1", "1,,2", "(1)x"] {
            assert!(parse_expression(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn huge_expressions_are_refused() {
        let cif = "data_x\nloop_\n_pdbx_struct_assembly_gen.assembly_id\n\
                   _pdbx_struct_assembly_gen.oper_expression\n_pdbx_struct_assembly_gen.asym_id_list\n\
                   1 '(1-1000)(1-1000)' A\n";
        let d = parse_cif(cif.as_bytes(), None).unwrap();
        let s = Structure {
            models: vec![Model::new(1)],
            strings: crate::structure::StringPool::new(),
            source: crate::structure::Source {
                format: crate::io::Format::Mmcif,
                path: None,
                data_block: None,
            },
            diagnostics: crate::diag::Diagnostics::new(),
        };
        assert!(matches!(
            expand(s, &d, "1"),
            Err(Error::ResourceLimit { .. })
        ));
    }

    #[test]
    fn transforms_compose_right_to_left() {
        let rot = Transform {
            r: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0; 3],
        };
        let shift = Transform {
            r: Transform::IDENTITY.r,
            t: [1.0, 0.0, 0.0],
        };
        // (rot)(shift): shift first, then rotate.
        let c = rot.then_after(&shift);
        assert_eq!(c.apply([0.0; 3]), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn cif_and_pdb_records() {
        let cif = "data_x\nloop_\n_pdbx_struct_assembly_gen.assembly_id\n\
                   _pdbx_struct_assembly_gen.oper_expression\n_pdbx_struct_assembly_gen.asym_id_list\n\
                   1 '(1-2)' A,B\n2 1 C\n\
                   loop_\n_pdbx_struct_oper_list.id\n_pdbx_struct_oper_list.matrix[1][1]\n\
                   _pdbx_struct_oper_list.matrix[1][2]\n_pdbx_struct_oper_list.matrix[1][3]\n\
                   _pdbx_struct_oper_list.vector[1]\n_pdbx_struct_oper_list.matrix[2][1]\n\
                   _pdbx_struct_oper_list.matrix[2][2]\n_pdbx_struct_oper_list.matrix[2][3]\n\
                   _pdbx_struct_oper_list.vector[2]\n_pdbx_struct_oper_list.matrix[3][1]\n\
                   _pdbx_struct_oper_list.matrix[3][2]\n_pdbx_struct_oper_list.matrix[3][3]\n\
                   _pdbx_struct_oper_list.vector[3]\n\
                   1 1 0 0 0 0 1 0 0 0 0 1 0\n2 1 0 0 10 0 1 0 0 0 0 1 0\n";
        let d = parse_cif(cif.as_bytes(), None).unwrap();
        let s = d.summaries().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].id.as_str(), s[0].copies), ("1", 2));
        assert_eq!(s[0].chains, ["A", "B"]);
        assert_eq!(d.operators["2"].apply([0.0; 3]), [10.0, 0.0, 0.0]);

        let pdb = "REMARK 350 BIOMOLECULE: 1\n\
                   REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B\n\
                   REMARK 350                    AND CHAINS: C\n\
                   REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n\
                   REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n\
                   REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n\
                   REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000       50.00000\n\
                   REMARK 350   BIOMT2   2  0.000000 -1.000000  0.000000        0.00000\n\
                   REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n";
        let d = parse_pdb(pdb.as_bytes());
        let s = d.summaries().unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].copies, 2);
        assert_eq!(s[0].chains, ["A", "B", "C"]);
        assert_eq!(d.operators["1:2"].apply([1.0, 2.0, 3.0]), [49.0, -2.0, 3.0]);
    }
}
