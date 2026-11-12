//! In-memory representation of parsed molecular structures.

mod element;
mod pool;
mod select;

use std::path::PathBuf;

pub use element::Element;
pub use pool::{StringPool, Sym};
pub use select::{AltLocPolicy, AtomSet, ChainIdKind, ModelChoice, Selection, WATER_RESIDUES};

use crate::diag::Diagnostics;
use crate::io::Format;

/// One atom as read from a structure file.
///
/// Fields absent from the source format keep their defaults (see
/// [`AtomRecord::new`]). Strings are [`Sym`] handles into the owning
/// [`Structure::strings`] pool.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AtomRecord {
    /// Atom serial number (PDB serial, mmCIF `id`; 0 if absent).
    pub serial: u32,
    /// Atom name (`CA`, `O5'`, ...).
    pub name: Sym,
    /// Alternate location indicator.
    pub alt_loc: Option<char>,
    /// Residue (component) name.
    pub res_name: Sym,
    /// Author chain identifier (PDB chain ID, mmCIF `auth_asym_id`).
    pub chain: Sym,
    /// mmCIF `label_asym_id`; equals `chain` for other formats.
    pub label_chain: Sym,
    /// Author residue number (PDB resSeq, mmCIF `auth_seq_id`).
    pub res_seq: i32,
    /// mmCIF `label_seq_id`, if given.
    pub label_seq: Option<i32>,
    /// Insertion code.
    pub ins_code: Option<char>,
    /// Cartesian coordinates in Å.
    pub position: [f64; 3],
    /// Occupancy (1.0 if absent).
    pub occupancy: f32,
    /// Isotropic B-factor (0.0 if absent).
    pub b_factor: f32,
    /// Chemical element, if known or inferable.
    pub element: Option<Element>,
    /// Formal charge.
    pub formal_charge: i8,
    /// `true` for HETATM records (non-polymer atoms).
    pub het: bool,
    /// Radius given by the file (PQR, extended XYZ column), in Å.
    pub radius: Option<f32>,
    /// Partial charge given by the file (PQR), in e.
    pub partial_charge: Option<f32>,
}

impl AtomRecord {
    /// A record at `position` with every other field at its default.
    pub fn new(position: [f64; 3]) -> Self {
        Self {
            serial: 0,
            name: Sym::EMPTY,
            alt_loc: None,
            res_name: Sym::EMPTY,
            chain: Sym::EMPTY,
            label_chain: Sym::EMPTY,
            res_seq: 0,
            label_seq: None,
            ins_code: None,
            position,
            occupancy: 1.0,
            b_factor: 0.0,
            element: None,
            formal_charge: 0,
            het: false,
            radius: None,
            partial_charge: None,
        }
    }
}

/// One model (PDB `MODEL`, mmCIF `pdbx_PDB_model_num`, XYZ frame).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Model {
    /// Model number as given by the file (XYZ frames are numbered from 1).
    pub number: i32,
    /// Atoms in file order.
    pub atoms: Vec<AtomRecord>,
}

impl Model {
    /// An empty model with the given number.
    pub fn new(number: i32) -> Self {
        Self {
            number,
            atoms: Vec::new(),
        }
    }
}

/// Where a structure came from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Source {
    /// Format the structure was read from.
    pub format: Format,
    /// File path, when read from a file.
    pub path: Option<PathBuf>,
    /// mmCIF data block name.
    pub data_block: Option<String>,
}

/// A parsed structure: models of atoms plus the string pool they refer to.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Structure {
    /// Models in file order (at least one).
    pub models: Vec<Model>,
    /// Interned names referenced by the atom records.
    pub strings: StringPool,
    /// Provenance.
    pub source: Source,
    /// Warnings raised while reading.
    pub diagnostics: Diagnostics,
}

impl Structure {
    /// The string behind a handle stored in one of this structure's records.
    pub fn str(&self, sym: Sym) -> &str {
        self.strings.get(sym)
    }

    /// Total number of atoms over all models.
    pub fn atom_count(&self) -> usize {
        self.models.iter().map(|m| m.atoms.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atom_record_stays_compact() {
        assert!(size_of::<AtomRecord>() <= 96, "{}", size_of::<AtomRecord>());
    }
}
