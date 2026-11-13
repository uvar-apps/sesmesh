// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Chemical elements and element inference from atom names.

use std::fmt;
use std::num::NonZeroU8;

/// Element symbols indexed by atomic number − 1.
const SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Lookup table from a normalized one- or two-letter symbol to an atomic
/// number: index `(first - b'A') * 27 + (second - b'a' + 1)`, `second = 0`
/// for one-letter symbols. 0 means "no such element".
const LOOKUP: [u8; 26 * 27] = build_lookup();

const fn build_lookup() -> [u8; 26 * 27] {
    let mut table = [0u8; 26 * 27];
    let mut z = 0;
    while z < SYMBOLS.len() {
        let s = SYMBOLS[z].as_bytes();
        let second = if s.len() == 2 { s[1] - b'a' + 1 } else { 0 };
        table[(s[0] - b'A') as usize * 27 + second as usize] = (z + 1) as u8;
        z += 1;
    }
    table
}

/// A chemical element, identified by its atomic number (1–118).
///
/// Deuterium (`D`) and tritium (`T`) symbols map to [`Element::H`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Element(NonZeroU8);

macro_rules! element_consts {
    ($($name:ident = $z:literal),* $(,)?) => {
        $(
            #[doc = concat!("Element `", stringify!($name), "` (Z = ", stringify!($z), ").")]
            // Named exactly like the chemical symbol (`Element::Fe`).
            #[allow(non_upper_case_globals)]
            pub const $name: Element = Element(NonZeroU8::new($z).unwrap());
        )*
    };
}

impl Element {
    element_consts! {
        H = 1, C = 6, N = 7, O = 8, F = 9, Na = 11, Mg = 12, P = 15, S = 16, Cl = 17, K = 19,
        Ca = 20, Mn = 25, Fe = 26, Co = 27, Ni = 28, Cu = 29, Zn = 30, Se = 34, Br = 35, I = 53,
        Hg = 80,
    }

    /// Element with atomic number `z`, if `1 <= z <= 118`.
    pub const fn from_atomic_number(z: u8) -> Option<Self> {
        if z as usize >= 1 && z as usize <= SYMBOLS.len() {
            match NonZeroU8::new(z) {
                Some(nz) => Some(Self(nz)),
                None => None,
            }
        } else {
            None
        }
    }

    /// Atomic number (1–118).
    pub const fn atomic_number(self) -> u8 {
        self.0.get()
    }

    /// Standard symbol, e.g. `"Fe"`.
    pub fn symbol(self) -> &'static str {
        SYMBOLS[self.0.get() as usize - 1]
    }

    /// `true` for hydrogen (including deuterium and tritium).
    pub const fn is_hydrogen(self) -> bool {
        self.0.get() == 1
    }

    /// Parses a symbol case-insensitively, ignoring surrounding ASCII
    /// whitespace (`"FE"`, `"fe"` and `" Fe"` are iron). `D` and `T` are
    /// hydrogen.
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        Self::from_symbol_bytes(symbol.as_bytes())
    }

    /// Byte-slice variant of [`Element::from_symbol`].
    pub fn from_symbol_bytes(symbol: &[u8]) -> Option<Self> {
        match symbol.trim_ascii() {
            [a] => Self::lookup(*a, None),
            [a, b] => Self::lookup(*a, Some(*b)),
            _ => None,
        }
    }

    fn lookup(first: u8, second: Option<u8>) -> Option<Self> {
        let first = first.to_ascii_uppercase();
        if !first.is_ascii_uppercase() {
            return None;
        }
        if second.is_none() && (first == b'D' || first == b'T') {
            return Some(Self::H);
        }
        let second = match second {
            None => 0,
            Some(c) if c.is_ascii_alphabetic() => c.to_ascii_lowercase() - b'a' + 1,
            Some(_) => return None,
        };
        let z = LOOKUP[(first - b'A') as usize * 27 + second as usize];
        Self::from_atomic_number(z)
    }

    /// Infers the element from a PDB atom-name field (columns 13–16, exactly
    /// as written, including leading spaces), used when columns 77–78 are
    /// blank.
    ///
    /// The PDB convention right-justifies the element symbol in columns
    /// 13–14, so `" CA "` is a carbon and `"CA  "` calcium. Files that do not
    /// follow the convention are handled heuristically: a two-letter reading
    /// is used only for single-atom residues named after the element (ions
    /// such as `CA`, `ZN`) or for non-hydrogen hetero atoms; otherwise the
    /// first letter is used (so `"HG21"` in THR is hydrogen, not mercury).
    pub fn infer_from_pdb_name(name_field: &[u8], res_name: &str, het: bool) -> Option<Self> {
        let first = name_field.first().copied().unwrap_or(b' ');
        if first == b' ' || first.is_ascii_digit() {
            // Right-justified one-letter element (or a leading digit as in
            // old hydrogen names like "1HG1").
            let letter = name_field
                .iter()
                .skip(1)
                .find(|c| c.is_ascii_alphabetic())?;
            return Self::lookup(*letter, None);
        }
        Self::infer_from_name(&String::from_utf8_lossy(name_field), res_name, het)
    }

    /// Infers the element from a free-form atom name (mmCIF without
    /// `type_symbol`, PQR, XYZ labels). Leading digits are skipped; trailing
    /// digits and primes are ignored. See [`Element::infer_from_pdb_name`] for
    /// the one- vs two-letter heuristic.
    pub fn infer_from_name(name: &str, res_name: &str, het: bool) -> Option<Self> {
        let letters: Vec<u8> = name
            .bytes()
            .skip_while(|c| !c.is_ascii_alphabetic())
            .take_while(u8::is_ascii_alphabetic)
            .collect();
        let first = *letters.first()?;
        if letters.len() >= 2
            && let Some(two) = Self::lookup(first, Some(letters[1]))
        {
            let named_after = res_name.trim().eq_ignore_ascii_case(two.symbol());
            let hetero_heavy = het && !first.eq_ignore_ascii_case(&b'H');
            if named_after || hetero_heavy || Self::lookup(first, None).is_none() {
                return Some(two);
            }
        }
        Self::lookup(first, None)
    }
}

impl fmt::Debug for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_symbols_round_trip() {
        assert_eq!(SYMBOLS.len(), 118);
        for z in 1..=118u8 {
            let e = Element::from_atomic_number(z).unwrap();
            assert_eq!(e.atomic_number(), z);
            assert_eq!(Element::from_symbol(e.symbol()), Some(e), "{}", e.symbol());
            assert_eq!(
                Element::from_symbol(&e.symbol().to_ascii_uppercase()),
                Some(e)
            );
        }
        assert_eq!(Element::from_atomic_number(0), None);
        assert_eq!(Element::from_atomic_number(119), None);
    }

    #[test]
    fn symbol_parsing() {
        assert_eq!(Element::from_symbol(" fe "), Some(Element::Fe));
        assert_eq!(Element::from_symbol("D"), Some(Element::H));
        assert_eq!(Element::from_symbol("T"), Some(Element::H));
        assert_eq!(Element::from_symbol("X"), None);
        assert_eq!(Element::from_symbol("Xx"), None);
        assert_eq!(Element::from_symbol("C1"), None);
        assert_eq!(Element::from_symbol(""), None);
        assert_eq!(Element::from_symbol("Fe3"), None);
        assert_eq!(Element::Hg.symbol(), "Hg");
        assert_eq!(format!("{:?}", Element::Cl), "Cl");
    }

    #[test]
    fn pdb_name_inference() {
        let inf =
            |name: &str, res: &str, het| Element::infer_from_pdb_name(name.as_bytes(), res, het);
        assert_eq!(inf(" CA ", "ALA", false), Some(Element::C));
        assert_eq!(inf("CA  ", "CA", true), Some(Element::Ca));
        assert_eq!(inf(" N  ", "ALA", false), Some(Element::N));
        assert_eq!(inf("1HG1", "VAL", false), Some(Element::H));
        assert_eq!(inf("HG21", "THR", false), Some(Element::H));
        assert_eq!(inf("HD21", "ASN", false), Some(Element::H));
        assert_eq!(inf("ZN  ", "ZN", true), Some(Element::Zn));
        assert_eq!(inf("FE  ", "HEM", true), Some(Element::Fe));
        assert_eq!(inf("CL1 ", "LIG", true), Some(Element::Cl));
        assert_eq!(inf("HO1 ", "GLC", true), Some(Element::H));
        // Non-standard left-justified carbon alpha in an ATOM record.
        assert_eq!(inf("CA  ", "ALA", false), Some(Element::C));
        assert_eq!(inf("    ", "ALA", false), None);
    }

    #[test]
    fn free_name_inference() {
        assert_eq!(
            Element::infer_from_name("C1'", "DA", false),
            Some(Element::C)
        );
        assert_eq!(
            Element::infer_from_name("OP1", "DA", false),
            Some(Element::O)
        );
        assert_eq!(Element::infer_from_name("Cl", "", true), Some(Element::Cl));
        assert_eq!(Element::infer_from_name("Na2", "", false), Some(Element::N));
        assert_eq!(
            Element::infer_from_name("NA", "NA", true),
            Some(Element::Na)
        );
        assert_eq!(Element::infer_from_name("1H", "", false), Some(Element::H));
        assert_eq!(Element::infer_from_name("Xq", "", false), None);
        assert_eq!(Element::infer_from_name("", "", false), None);
    }
}
