//! Atom selection: model, alternate locations, hydrogens, waters, hetero
//! atoms, chains and residue exclusions (Spec FR-SEL, Design §5.6).

use rustc_hash::FxHashMap;

use super::{AtomRecord, Element, Model, StringPool, Structure, Sym};
use crate::diag::{Diagnostics, WarningKind};
use crate::error::{Error, Result};

/// Residue names treated as water by [`Selection::waters`].
pub const WATER_RESIDUES: &[&str] = &[
    "HOH", "DOD", "WAT", "H2O", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "SPC", "T3P", "T4P", "T5P",
];

/// Which model to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ModelChoice {
    /// The first model in the file (default).
    #[default]
    First,
    /// The model with this number (PDB `MODEL n`, mmCIF model number).
    Number(i32),
    /// The model at this 0-based position (useful for XYZ frames).
    Index(usize),
}

/// How alternate conformations are resolved. Decisions are made per residue
/// so that a residue never mixes atoms of different conformers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AltLocPolicy {
    /// Keep the conformer with the highest total occupancy in each residue
    /// (ties: the one appearing first). Default.
    #[default]
    HighestOccupancy,
    /// Keep the first conformer appearing in each residue.
    First,
    /// Keep this conformer where present; elsewhere fall back to
    /// [`AltLocPolicy::HighestOccupancy`].
    Only(char),
    /// Keep every atom.
    All,
}

/// Which chain identifier [`Selection::chains`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ChainIdKind {
    /// Author chain IDs (PDB chain ID, mmCIF `auth_asym_id`). Default.
    #[default]
    Auth,
    /// mmCIF `label_asym_id`.
    Label,
}

/// Declarative atom selection. `Selection::default()` keeps the first model,
/// resolves alternate locations by occupancy, and drops hydrogens and waters.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Selection {
    /// Model to use.
    pub model: ModelChoice,
    /// Alternate-location policy.
    pub alt_loc: AltLocPolicy,
    /// Keep hydrogen (and deuterium) atoms. Default `false`.
    pub hydrogens: bool,
    /// Keep waters (see [`WATER_RESIDUES`]). Default `false`.
    pub waters: bool,
    /// Keep hetero (HETATM) atoms. Default `true`.
    pub hetero: bool,
    /// Keep only these chains (`None`: all chains).
    pub chains: Option<Vec<String>>,
    /// Which identifier `chains` refers to.
    pub chain_ids: ChainIdKind,
    /// Residue names to drop (case-insensitive).
    pub exclude_residues: Vec<String>,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            model: ModelChoice::First,
            alt_loc: AltLocPolicy::HighestOccupancy,
            hydrogens: false,
            waters: false,
            hetero: true,
            chains: None,
            chain_ids: ChainIdKind::Auth,
            exclude_residues: Vec::new(),
        }
    }
}

/// The atoms of one model chosen by a [`Selection`].
#[derive(Debug, Clone)]
pub struct AtomSet<'a> {
    structure: &'a Structure,
    model: usize,
    indices: Vec<u32>,
}

impl<'a> AtomSet<'a> {
    /// The structure the atoms belong to.
    pub fn structure(&self) -> &'a Structure {
        self.structure
    }

    /// The model the atoms belong to.
    pub fn model(&self) -> &'a Model {
        &self.structure.models[self.model]
    }

    /// Indices of the selected atoms within [`AtomSet::model`], ascending.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Number of selected atoms.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// `true` if no atom is selected.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Selected atoms in file order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a AtomRecord> + '_ {
        let atoms = &self.structure.models[self.model].atoms;
        self.indices.iter().map(move |&i| &atoms[i as usize])
    }

    /// Positions of the selected atoms.
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.iter().map(|a| a.position).collect()
    }
}

impl Structure {
    /// Applies `selection` and returns the chosen atoms.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the model or a requested chain does not
    /// exist; [`Error::EmptySelection`] if no atom survives the filters.
    pub fn select(&self, selection: &Selection, diags: &mut Diagnostics) -> Result<AtomSet<'_>> {
        self.select_with(selection, diags, |_, _| true)
    }

    /// Like [`Structure::select`], additionally keeping only atoms for which
    /// `predicate` returns `true`.
    ///
    /// # Errors
    /// See [`Structure::select`].
    pub fn select_with<F>(
        &self,
        selection: &Selection,
        diags: &mut Diagnostics,
        mut predicate: F,
    ) -> Result<AtomSet<'_>>
    where
        F: FnMut(&AtomRecord, &StringPool) -> bool,
    {
        let model_index = self.resolve_model(selection.model)?;
        let model = &self.models[model_index];
        let atoms = &model.atoms;

        let keep_alt = resolve_alt_locs(self, atoms, selection.alt_loc, diags);
        let chain_filter = self.chain_filter(model, selection)?;
        let excluded: Vec<Sym> = selection
            .exclude_residues
            .iter()
            .flat_map(|name| self.syms_matching(name))
            .collect();
        let waters: Vec<Sym> = WATER_RESIDUES
            .iter()
            .flat_map(|name| self.syms_matching(name))
            .collect();

        let mut indices = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            if !keep_alt.as_ref().is_none_or(|k| k[i])
                || (!selection.hydrogens && atom.element.is_some_and(Element::is_hydrogen))
                || (!selection.waters && waters.contains(&atom.res_name))
                || (!selection.hetero && atom.het)
                || excluded.contains(&atom.res_name)
                || chain_filter.as_ref().is_some_and(|chains| {
                    let id = match selection.chain_ids {
                        ChainIdKind::Auth => atom.chain,
                        ChainIdKind::Label => atom.label_chain,
                    };
                    !chains.contains(&id)
                })
                || !predicate(atom, &self.strings)
            {
                continue;
            }
            indices.push(u32::try_from(i).expect("fewer than 2^32 atoms per model"));
        }

        if indices.is_empty() {
            return Err(Error::EmptySelection(format!(
                "model {} has {} atoms but none passed the filters (hydrogens: {}, waters: {}, \
                 hetero: {}, chains: {}, excluded residues: {})",
                model.number,
                atoms.len(),
                keep_word(selection.hydrogens),
                keep_word(selection.waters),
                keep_word(selection.hetero),
                selection
                    .chains
                    .as_ref()
                    .map_or_else(|| "all".to_owned(), |c| c.join(",")),
                if selection.exclude_residues.is_empty() {
                    "none".to_owned()
                } else {
                    selection.exclude_residues.join(",")
                },
            )));
        }
        Ok(AtomSet {
            structure: self,
            model: model_index,
            indices,
        })
    }

    fn resolve_model(&self, choice: ModelChoice) -> Result<usize> {
        let found = match choice {
            ModelChoice::First => (!self.models.is_empty()).then_some(0),
            ModelChoice::Number(n) => self.models.iter().position(|m| m.number == n),
            ModelChoice::Index(i) => (i < self.models.len()).then_some(i),
        };
        found.ok_or_else(|| {
            let numbers: Vec<String> = self.models.iter().map(|m| m.number.to_string()).collect();
            Error::InvalidInput(format!(
                "model {choice:?} not found (models: {})",
                if numbers.is_empty() {
                    "none".to_owned()
                } else {
                    numbers.join(", ")
                }
            ))
        })
    }

    /// Symbols whose string equals `name` ignoring ASCII case.
    fn syms_matching<'s>(&'s self, name: &'s str) -> impl Iterator<Item = Sym> + 's {
        self.strings
            .iter()
            .filter(move |(_, s)| s.eq_ignore_ascii_case(name))
            .map(|(sym, _)| sym)
    }

    fn chain_filter(&self, model: &Model, selection: &Selection) -> Result<Option<Vec<Sym>>> {
        let Some(requested) = &selection.chains else {
            return Ok(None);
        };
        let id_of = |a: &AtomRecord| match selection.chain_ids {
            ChainIdKind::Auth => a.chain,
            ChainIdKind::Label => a.label_chain,
        };
        let mut present: Vec<Sym> = model.atoms.iter().map(id_of).collect();
        present.sort_unstable();
        present.dedup();
        let mut syms = Vec::with_capacity(requested.len());
        for chain in requested {
            match self.strings.find(chain) {
                Some(sym) if present.contains(&sym) => syms.push(sym),
                _ => {
                    let available: Vec<&str> = present.iter().map(|&s| self.str(s)).collect();
                    return Err(Error::InvalidInput(format!(
                        "chain '{chain}' not found in model {} (available: {})",
                        model.number,
                        available.join(", ")
                    )));
                }
            }
        }
        Ok(Some(syms))
    }
}

fn keep_word(keep: bool) -> &'static str {
    if keep { "kept" } else { "dropped" }
}

/// Per-atom keep flags for the alt-loc policy, or `None` when every atom is
/// kept (policy `All` or no alternate locations).
fn resolve_alt_locs(
    structure: &Structure,
    atoms: &[AtomRecord],
    policy: AltLocPolicy,
    diags: &mut Diagnostics,
) -> Option<Vec<bool>> {
    if policy == AltLocPolicy::All || atoms.iter().all(|a| a.alt_loc.is_none()) {
        return None;
    }
    type ResidueKey = (Sym, i32, Option<char>, Sym);
    // Per residue: (letter, total occupancy) in order of first appearance.
    let mut residues: FxHashMap<ResidueKey, Vec<(char, f64)>> = FxHashMap::default();
    let key = |a: &AtomRecord| (a.chain, a.res_seq, a.ins_code, a.label_chain);
    for atom in atoms {
        if let Some(letter) = atom.alt_loc {
            let letters = residues.entry(key(atom)).or_default();
            match letters.iter_mut().find(|(c, _)| *c == letter) {
                Some((_, occ)) => *occ += f64::from(atom.occupancy),
                None => letters.push((letter, f64::from(atom.occupancy))),
            }
        }
    }
    let chosen: FxHashMap<ResidueKey, char> = residues
        .into_iter()
        .map(|(k, letters)| {
            let highest = || {
                letters
                    .iter()
                    .fold(None::<(char, f64)>, |best, &(c, occ)| match best {
                        Some((_, b)) if b >= occ => best,
                        _ => Some((c, occ)),
                    })
                    .map(|(c, _)| c)
                    .expect("residue has at least one alt-loc")
            };
            let pick = match policy {
                AltLocPolicy::First => letters[0].0,
                AltLocPolicy::Only(c) if letters.iter().any(|(l, _)| *l == c) => c,
                _ => highest(),
            };
            (k, pick)
        })
        .collect();

    let keep: Vec<bool> = atoms
        .iter()
        .map(|a| match a.alt_loc {
            None => true,
            Some(c) => {
                let k = chosen[&key(a)] == c;
                if !k {
                    diags.warn(
                        WarningKind::AltLocResolved,
                        "atoms of alternate conformations discarded",
                        || {
                            format!(
                                "{}{}{} {} alt {c}",
                                structure.str(a.chain),
                                a.res_seq,
                                a.ins_code.map(String::from).unwrap_or_default(),
                                structure.str(a.name)
                            )
                        },
                    );
                }
                k
            }
        })
        .collect();
    Some(keep)
}
