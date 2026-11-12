//! ProtOr united-atom radii (J. Tsai, R. Taylor, C. Chothia, M. Gerstein,
//! *J. Mol. Biol.* 290, 253 (1999)) for standard amino acids, nucleotides and
//! a few common modified residues.
//!
//! The residue/atom → class assignment is transcribed from FreeSASA's
//! `share/protor.config` (MIT licence) into [`protor_table`]. Hydrogens are not
//! classified: ProtOr radii already include them.

#[path = "protor_table.rs"]
mod protor_table;

use protor_table::{ATOMS, CLASSES};

/// ProtOr class name and radius for an atom of a residue (names compared
/// case-insensitively).
pub(crate) fn lookup(residue: &str, atom: &str) -> Option<(&'static str, f64)> {
    let (residue, atom) = (residue.to_ascii_uppercase(), atom.to_ascii_uppercase());
    let i = ATOMS
        .binary_search_by(|&(r, a, _)| (r, a).cmp(&(residue.as_str(), atom.as_str())))
        .ok()?;
    let (class, radius) = CLASSES[ATOMS[i].2 as usize];
    Some((class, radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_and_consistent() {
        assert!(
            ATOMS
                .windows(2)
                .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1))
        );
        assert!(ATOMS.iter().all(|a| (a.2 as usize) < CLASSES.len()));
        assert_eq!(ATOMS.len(), 506);
    }

    #[test]
    fn known_classes() {
        assert_eq!(lookup("ALA", "CB"), Some(("C4H3", 1.88)));
        assert_eq!(lookup("ala", "c"), Some(("C3H0", 1.61)));
        assert_eq!(lookup("SER", "OG"), Some(("O2H1", 1.46)));
        assert_eq!(lookup("ALA", "O"), Some(("O1H0", 1.42)));
        assert_eq!(lookup("MET", "SD"), Some(("S2H0", 1.77)));
        assert_eq!(lookup("ALA", "XX"), None);
        assert_eq!(lookup("HEM", "FE"), None);
        assert!(lookup("DA", "C1'").is_some());
    }
}
