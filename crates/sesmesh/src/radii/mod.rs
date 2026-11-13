// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Atomic radius assignment (Spec FR-RAD, Design §5.6).
//!
//! ```
//! use sesmesh::radii::{assign, RadiusOptions, RadiusSet};
//! use sesmesh::{io, Diagnostics, Selection};
//!
//! let pdb = "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00  0.00           C\n";
//! let structure = io::read_bytes(pdb.as_bytes(), &io::ReadOptions::default())?;
//! let mut diags = Diagnostics::new();
//! let atoms = structure.select(&Selection::default(), &mut diags)?;
//! let radii = assign(&atoms, &RadiusOptions::new(RadiusSet::Bondi), &mut diags)?;
//! assert_eq!(radii, [1.70]);
//! # Ok::<(), sesmesh::Error>(())
//! ```

mod protor;
mod rules;
mod tables;

pub use rules::RadiusRules;

use crate::diag::{Diagnostics, WarningKind};
use crate::error::{Error, Result};
use crate::structure::{AtomRecord, AtomSet, Element, Structure};

/// Built-in radius sets that custom rules can build on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BuiltinRadii {
    /// Bondi (1964) element radii completed with Mantina et al. (2009).
    Bondi,
    /// Alvarez (2013) element radii.
    Alvarez,
    /// ProtOr united-atom radii (Tsai et al. 1999) for standard residues,
    /// Bondi element radii for other atoms.
    ProtOr,
}

impl BuiltinRadii {
    /// Short lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            BuiltinRadii::Bondi => "bondi",
            BuiltinRadii::Alvarez => "alvarez",
            BuiltinRadii::ProtOr => "protor",
        }
    }

    /// Element radius of the set's element table (`ProtOr` uses Bondi).
    pub fn element_radius(self, element: Element) -> Option<f64> {
        let table = match self {
            BuiltinRadii::Bondi | BuiltinRadii::ProtOr => &tables::BONDI_DENSE,
            BuiltinRadii::Alvarez => &tables::ALVAREZ_DENSE,
        };
        let r = table[usize::from(element.atomic_number())];
        (r > 0.0).then_some(r)
    }

    /// All `(element, radius)` entries of the element table.
    pub fn element_table(self) -> Vec<(Element, f64)> {
        let pairs = match self {
            BuiltinRadii::Bondi | BuiltinRadii::ProtOr => tables::BONDI,
            BuiltinRadii::Alvarez => tables::ALVAREZ,
        };
        pairs
            .iter()
            .filter_map(|&(z, r)| Some((Element::from_atomic_number(z)?, r)))
            .collect()
    }

    /// Per-atom entries `(residue, atom, class, radius)` of the set (ProtOr
    /// only; empty for element-only sets), sorted by residue and atom.
    pub fn atom_table(self) -> Vec<(&'static str, &'static str, &'static str, f64)> {
        match self {
            BuiltinRadii::ProtOr => protor::table(),
            BuiltinRadii::Bondi | BuiltinRadii::Alvarez => Vec::new(),
        }
    }

    /// Radius for an atom of `residue` named `atom`. For ProtOr, unclassified
    /// atoms use their element radius.
    pub fn atom_radius(self, residue: &str, atom: &str, element: Option<Element>) -> Option<f64> {
        if self == BuiltinRadii::ProtOr
            && let Some((_, r)) = protor::lookup(residue, atom)
        {
            return Some(r);
        }
        element.and_then(|e| self.element_radius(e))
    }
}

/// The radius set to use.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RadiusSet {
    /// Bondi (1964) + Mantina (2009) element radii (default).
    Bondi,
    /// Alvarez (2013) element radii.
    Alvarez,
    /// ProtOr united-atom radii, Bondi for other atoms.
    ProtOr,
    /// The same radius for every atom.
    Uniform(f64),
    /// Custom rules over an optional built-in set.
    Custom(Box<RadiusRules>),
}

impl RadiusSet {
    /// Parses `bondi`, `alvarez`, `protor` or `uniform:<r>`.
    pub fn from_name(name: &str) -> Option<RadiusSet> {
        let lower = name.trim().to_ascii_lowercase();
        if let Some(r) = lower.strip_prefix("uniform:") {
            return r.parse().ok().map(RadiusSet::Uniform);
        }
        match lower.as_str() {
            "bondi" => Some(RadiusSet::Bondi),
            "alvarez" => Some(RadiusSet::Alvarez),
            "protor" => Some(RadiusSet::ProtOr),
            _ => None,
        }
    }
}

/// Options for [`assign`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RadiusOptions {
    /// Radius set.
    pub set: RadiusSet,
    /// Use radii provided by the file (PQR, extended XYZ) when present.
    /// Default `true`.
    pub use_file_radii: bool,
    /// Radius for atoms the set does not cover. Default 1.80 Å.
    pub unknown_radius: f64,
    /// Fail instead of using `unknown_radius`. Default `false`.
    pub strict: bool,
}

impl RadiusOptions {
    /// Default options for `set`.
    pub fn new(set: RadiusSet) -> Self {
        Self {
            set,
            use_file_radii: true,
            unknown_radius: 1.80,
            strict: false,
        }
    }
}

impl Default for RadiusOptions {
    fn default() -> Self {
        Self::new(RadiusSet::Bondi)
    }
}

/// Assigns a radius (Å) to every selected atom, in selection order.
///
/// # Errors
/// [`Error::InvalidInput`] for invalid options, or in strict mode when some
/// atoms have no radius.
pub fn assign(
    atoms: &AtomSet<'_>,
    options: &RadiusOptions,
    diags: &mut Diagnostics,
) -> Result<Vec<f64>> {
    let unknown = options.unknown_radius;
    if !(unknown.is_finite() && unknown >= 0.0) {
        return Err(Error::InvalidInput(format!(
            "unknown_radius must be finite and >= 0, got {unknown}"
        )));
    }
    if let RadiusSet::Uniform(r) = options.set
        && !(r.is_finite() && r > 0.0)
    {
        return Err(Error::InvalidInput(format!(
            "uniform radius must be finite and > 0, got {r}"
        )));
    }
    let structure = atoms.structure();
    let protor_like = match &options.set {
        RadiusSet::ProtOr => true,
        RadiusSet::Custom(rules) => rules.base() == Some(BuiltinRadii::ProtOr),
        _ => false,
    };
    let mut missing: Vec<String> = Vec::new();
    let mut radii = Vec::with_capacity(atoms.len());
    for atom in atoms.iter() {
        if options.use_file_radii
            && let Some(r) = atom.radius
        {
            radii.push(f64::from(r));
            continue;
        }
        if protor_like && atom.element.is_some_and(Element::is_hydrogen) {
            diags.warn(
                WarningKind::RadiusSetMismatch,
                "explicit hydrogens with united-atom ProtOr radii (which already include \
                 hydrogens); using element radii for them",
                || describe(structure, atom),
            );
        }
        let found = lookup(structure, atom, &options.set);
        let r = match found {
            Some(r) => r,
            None => {
                let fallback = match &options.set {
                    RadiusSet::Custom(rules) => rules.fallback(),
                    _ => None,
                };
                if let Some(f) = fallback {
                    f
                } else {
                    if options.strict {
                        if missing.len() < Diagnostics::MAX_EXAMPLES {
                            missing.push(describe(structure, atom));
                        }
                        radii.push(f64::NAN);
                        continue;
                    }
                    diags.warn(
                        WarningKind::UnknownRadius,
                        &format!(
                            "no radius in the selected set; using {:.2} Å",
                            options.unknown_radius
                        ),
                        || describe(structure, atom),
                    );
                    options.unknown_radius
                }
            }
        };
        radii.push(r);
    }
    if !missing.is_empty() {
        let total = radii.iter().filter(|r| r.is_nan()).count();
        return Err(Error::InvalidInput(format!(
            "{total} atom(s) have no radius in the selected set (strict mode), e.g. {}",
            missing.join(", ")
        )));
    }
    Ok(radii)
}

fn lookup(structure: &Structure, atom: &AtomRecord, set: &RadiusSet) -> Option<f64> {
    let res = structure.str(atom.res_name);
    let name = structure.str(atom.name);
    match set {
        RadiusSet::Bondi => BuiltinRadii::Bondi.atom_radius(res, name, atom.element),
        RadiusSet::Alvarez => BuiltinRadii::Alvarez.atom_radius(res, name, atom.element),
        RadiusSet::ProtOr => BuiltinRadii::ProtOr.atom_radius(res, name, atom.element),
        RadiusSet::Uniform(r) => Some(*r),
        RadiusSet::Custom(rules) => rules
            .lookup_rules(res, name, atom.element)
            .or_else(|| rules.base()?.atom_radius(res, name, atom.element)),
    }
}

fn describe(structure: &Structure, atom: &AtomRecord) -> String {
    format!(
        "{} {}{} {} ({})",
        structure.str(atom.res_name),
        structure.str(atom.chain),
        atom.res_seq,
        structure.str(atom.name),
        atom.element.map_or("?", Element::symbol)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{Format, ReadOptions, read_bytes};
    use crate::structure::Selection;

    const PDB: &str = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  H   ALA A   1       2.000   0.000   0.000  1.00  0.00           H
HETATM    4 FE   HEM A 201       3.000   0.000   0.000  1.00  0.00          FE
HETATM    5  X1  UNK A 301       4.000   0.000   0.000  1.00  0.00
";

    fn structure() -> Structure {
        read_bytes(PDB.as_bytes(), &ReadOptions::default()).unwrap()
    }

    fn all_atoms() -> Selection {
        Selection {
            hydrogens: true,
            ..Selection::default()
        }
    }

    #[test]
    fn bondi_with_fallback_warnings() {
        let s = structure();
        let mut d = Diagnostics::new();
        let atoms = s.select(&all_atoms(), &mut d).unwrap();
        let r = assign(&atoms, &RadiusOptions::default(), &mut d).unwrap();
        // N, C, H from Bondi; Fe and the unknown "X1" fall back to 1.80.
        assert_eq!(r, [1.55, 1.70, 1.20, 1.80, 1.80]);
        assert_eq!(d.count(WarningKind::UnknownRadius), 2);
    }

    #[test]
    fn protor_alvarez_uniform() {
        let s = structure();
        let mut d = Diagnostics::new();
        let atoms = s.select(&all_atoms(), &mut d).unwrap();
        let r = assign(&atoms, &RadiusOptions::new(RadiusSet::ProtOr), &mut d).unwrap();
        assert_eq!(&r[..3], [1.64, 1.88, 1.20]);
        assert_eq!(d.count(WarningKind::RadiusSetMismatch), 1);
        let r = assign(&atoms, &RadiusOptions::new(RadiusSet::Alvarez), &mut d).unwrap();
        assert_eq!(&r[..4], [1.66, 1.77, 1.20, 2.44]);
        let r = assign(&atoms, &RadiusOptions::new(RadiusSet::Uniform(1.5)), &mut d).unwrap();
        assert!(r.iter().all(|&x| x == 1.5));
        assert!(assign(&atoms, &RadiusOptions::new(RadiusSet::Uniform(0.0)), &mut d).is_err());
    }

    #[test]
    fn strict_mode_and_custom_rules() {
        let s = structure();
        let mut d = Diagnostics::new();
        let atoms = s.select(&all_atoms(), &mut d).unwrap();
        let mut opts = RadiusOptions {
            strict: true,
            ..RadiusOptions::default()
        };
        let e = assign(&atoms, &opts, &mut d).unwrap_err().to_string();
        assert!(e.contains("2 atom(s)"), "{e}");
        let mut rules = RadiusRules::default();
        rules.set_element(Element::Fe, 1.4).unwrap();
        rules.set_atom("UNK", "*", 2.2).unwrap();
        opts.set = RadiusSet::Custom(Box::new(rules));
        let r = assign(&atoms, &opts, &mut d).unwrap();
        assert_eq!(r, [1.55, 1.70, 1.20, 1.40, 2.20]);
    }

    #[test]
    fn file_radii_take_precedence() {
        let pqr = "ATOM 1 N ALA 1 0 0 0 -0.3 1.824\nATOM 2 CA ALA 1 1 0 0 0.1 0.0\n";
        let o = ReadOptions {
            format: Some(Format::Pqr),
            ..ReadOptions::default()
        };
        let s = read_bytes(pqr.as_bytes(), &o).unwrap();
        let mut d = Diagnostics::new();
        let atoms = s.select(&Selection::default(), &mut d).unwrap();
        assert_eq!(
            assign(&atoms, &RadiusOptions::default(), &mut d).unwrap(),
            [f64::from(1.824f32), 0.0]
        );
        let opts = RadiusOptions {
            use_file_radii: false,
            ..RadiusOptions::default()
        };
        assert_eq!(assign(&atoms, &opts, &mut d).unwrap(), [1.55, 1.70]);
    }

    #[test]
    fn set_names() {
        assert_eq!(RadiusSet::from_name("ProtOr"), Some(RadiusSet::ProtOr));
        assert_eq!(
            RadiusSet::from_name("uniform:1.5"),
            Some(RadiusSet::Uniform(1.5))
        );
        assert_eq!(RadiusSet::from_name("x"), None);
        assert_eq!(BuiltinRadii::Bondi.element_table().len(), 54);
    }
}
