//! Integration tests for the structure layer (Plan Phase 1) on vendored
//! fixtures and, when present, the downloaded benchmark set.
//!
//! Expected counts are independent of this code: RCSB `deposited_atom_count`
//! (non-hydrogen atoms of the first model) and `awk` over the PDB files.
//!
//! The fixtures are gzip-compressed, so these tests need the `gzip` feature.
#![cfg(feature = "gzip")]

use std::path::{Path, PathBuf};

use sesmesh::io::{Format, ReadOptions, read_bytes, read_path};
use sesmesh::radii::{RadiusOptions, RadiusSet, assign};
use sesmesh::structure::{AltLocPolicy, ModelChoice};
use sesmesh::{Diagnostics, Element, Error, Selection, Structure, WarningKind};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repo().join("tests/data/fixtures").join(name)
}

fn read(name: &str) -> Structure {
    read_path(fixture(name), &ReadOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A file from `tests/data/cache` (`cargo xtask fetch`), if downloaded.
fn cached(name: &str) -> Option<PathBuf> {
    let p = repo().join("tests/data/cache").join(name);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping: {name} not in tests/data/cache (run `cargo xtask fetch`)");
        None
    }
}

fn everything() -> Selection {
    let mut s = Selection::default();
    s.hydrogens = true;
    s.waters = true;
    s.alt_loc = AltLocPolicy::All;
    s
}

fn heavy_atoms_model_1(s: &Structure) -> usize {
    s.models[0]
        .atoms
        .iter()
        .filter(|a| !a.element.is_some_and(Element::is_hydrogen))
        .count()
}

/// (entry, atoms over all models, RCSB deposited_atom_count, models)
const ENTRIES: &[(&str, usize, usize, usize)] = &[
    ("1CRN", 327, 327, 1),
    ("1UBQ", 660, 660, 1),
    ("3FX2", 1386, 1386, 1),
    ("2PTN", 1712, 1712, 1),
    ("1EJG", 843, 424, 1),
    ("1L2Y", 304 * 38, 154, 38),
];

#[test]
fn fixtures_parse_in_both_formats_with_reference_counts() {
    for &(id, atoms, deposited, models) in ENTRIES {
        for ext in ["pdb.gz", "cif.gz"] {
            let s = read(&format!("{id}.{ext}"));
            assert_eq!(s.atom_count(), atoms, "{id}.{ext} atoms");
            assert_eq!(s.models.len(), models, "{id}.{ext} models");
            assert_eq!(heavy_atoms_model_1(&s), deposited, "{id}.{ext} heavy atoms");
            assert!(
                s.models[0].atoms.iter().all(|a| a.element.is_some()),
                "{id}.{ext}: every atom has an element"
            );
            let expected = if ext.starts_with("pdb") {
                Format::Pdb
            } else {
                Format::Mmcif
            };
            assert_eq!(s.source.format, expected);
        }
    }
}

#[test]
fn pdb_and_mmcif_agree_atom_by_atom() {
    for &(id, ..) in ENTRIES {
        let pdb = read(&format!("{id}.pdb.gz"));
        let cif = read(&format!("{id}.cif.gz"));
        assert_eq!(pdb.models.len(), cif.models.len());
        for (mp, mc) in pdb.models.iter().zip(&cif.models) {
            assert_eq!(mp.number, mc.number, "{id}");
            assert_eq!(mp.atoms.len(), mc.atoms.len(), "{id} model {}", mp.number);
            for (i, (a, b)) in mp.atoms.iter().zip(&mc.atoms).enumerate() {
                let ctx = format!("{id} model {} atom {i}", mp.number);
                assert_eq!(pdb.str(a.name), cif.str(b.name), "{ctx} name");
                assert_eq!(pdb.str(a.res_name), cif.str(b.res_name), "{ctx} residue");
                assert_eq!(pdb.str(a.chain), cif.str(b.chain), "{ctx} chain");
                assert_eq!(a.res_seq, b.res_seq, "{ctx} res_seq");
                assert_eq!(a.ins_code, b.ins_code, "{ctx} ins_code");
                assert_eq!(a.alt_loc, b.alt_loc, "{ctx} alt_loc");
                assert_eq!(a.position, b.position, "{ctx} position");
                assert_eq!(a.occupancy, b.occupancy, "{ctx} occupancy");
                assert!((a.b_factor - b.b_factor).abs() <= 0.005, "{ctx} b_factor");
                assert_eq!(a.element, b.element, "{ctx} element");
                assert_eq!(a.het, b.het, "{ctx} het");
                assert_eq!(a.formal_charge, b.formal_charge, "{ctx} charge");
            }
        }
    }
}

#[test]
fn fixtures_parse_without_warnings() {
    for &(id, ..) in ENTRIES {
        for ext in ["pdb.gz", "cif.gz"] {
            let s = read(&format!("{id}.{ext}"));
            assert!(s.diagnostics.is_empty(), "{id}.{ext}: {:?}", s.diagnostics);
        }
    }
}

#[test]
fn default_selection_drops_waters_and_hydrogens() {
    let mut d = Diagnostics::new();
    // 1UBQ: 660 atoms, 58 waters (awk).
    let s = read("1UBQ.cif.gz");
    assert_eq!(s.select(&Selection::default(), &mut d).unwrap().len(), 602);
    // 3FX2: 1386 atoms, 253 waters, 31 FMN hetero atoms (awk).
    let s = read("3FX2.pdb.gz");
    assert_eq!(s.select(&Selection::default(), &mut d).unwrap().len(), 1133);
    let mut no_het = Selection::default();
    no_het.hetero = false;
    assert_eq!(s.select(&no_het, &mut d).unwrap().len(), 1102);
    // 1L2Y: 154 heavy atoms per model; model 7 by number and by index.
    let s = read("1L2Y.pdb.gz");
    let mut sel = Selection::default();
    sel.model = ModelChoice::Number(7);
    let set = s.select(&sel, &mut d).unwrap();
    assert_eq!((set.len(), set.model().number), (154, 7));
    sel.model = ModelChoice::Index(37);
    assert_eq!(s.select(&sel, &mut d).unwrap().model().number, 38);
    sel.model = ModelChoice::Number(39);
    assert!(matches!(
        s.select(&sel, &mut d),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn alt_loc_policies_on_1ejg() {
    // Expected counts from awk over 1EJG.pdb (all atoms, hydrogens included).
    let s = read("1EJG.pdb.gz");
    let count = |policy| {
        let mut sel = everything();
        sel.alt_loc = policy;
        let mut d = Diagnostics::new();
        let n = s.select(&sel, &mut d).unwrap().len();
        (n, d.count(WarningKind::AltLocResolved))
    };
    assert_eq!(count(AltLocPolicy::All), (843, 0));
    assert_eq!(count(AltLocPolicy::First), (637, 843 - 637));
    assert_eq!(count(AltLocPolicy::HighestOccupancy), (641, 843 - 641));
    // No residue may mix conformers.
    for policy in [
        AltLocPolicy::First,
        AltLocPolicy::HighestOccupancy,
        AltLocPolicy::Only('C'),
    ] {
        let mut sel = everything();
        sel.alt_loc = policy;
        let set = s.select(&sel, &mut Diagnostics::new()).unwrap();
        let mut seen = std::collections::HashMap::new();
        for a in set.iter() {
            if let Some(c) = a.alt_loc {
                let prev = seen.insert((a.chain, a.res_seq, a.ins_code), c);
                assert!(prev.is_none_or(|p| p == c), "{policy:?} mixes conformers");
            }
        }
    }
}

#[test]
fn chain_and_residue_filters() {
    let s = read("3FX2.cif.gz");
    let mut d = Diagnostics::new();
    let mut sel = Selection::default();
    sel.exclude_residues = vec!["fmn".into()];
    assert_eq!(s.select(&sel, &mut d).unwrap().len(), 1102);
    sel.chains = Some(vec!["A".into()]);
    assert_eq!(s.select(&sel, &mut d).unwrap().len(), 1102);
    sel.chains = Some(vec!["Z".into()]);
    let err = s.select(&sel, &mut d).unwrap_err().to_string();
    assert!(err.contains("chain 'Z' not found"), "{err}");
    let mut only_water = Selection::default();
    only_water.waters = true;
    let n = s
        .select_with(&only_water, &mut d, |a, pool| pool.get(a.res_name) == "HOH")
        .unwrap()
        .len();
    assert_eq!(n, 253);
    let none = s.select_with(&Selection::default(), &mut d, |_, _| false);
    assert!(matches!(none, Err(Error::EmptySelection(_))));
}

#[test]
fn radii_cover_standard_proteins() {
    for set in [RadiusSet::Bondi, RadiusSet::ProtOr, RadiusSet::Alvarez] {
        let s = read("1CRN.pdb.gz");
        let mut d = Diagnostics::new();
        let atoms = s.select(&Selection::default(), &mut d).unwrap();
        let r = assign(&atoms, &RadiusOptions::new(set.clone()), &mut d).unwrap();
        assert_eq!(r.len(), 327);
        assert!(r.iter().all(|&x| (1.4..2.0).contains(&x)), "{set:?}");
        assert!(d.is_empty(), "{set:?}: {d:?}");
    }
}

#[test]
fn small_formats() {
    let pqr = read("alanine.pqr");
    assert_eq!(pqr.source.format, Format::Pqr);
    let mut d = Diagnostics::new();
    let mut sel = Selection::default();
    sel.hydrogens = true;
    let atoms = pqr.select(&sel, &mut d).unwrap();
    let r = assign(&atoms, &RadiusOptions::default(), &mut d).unwrap();
    assert_eq!(r[0], f64::from(1.824f32));
    assert_eq!(r[5], f64::from(0.6f32));

    let water = read("water.xyz");
    assert_eq!(water.atom_count(), 3);
    assert_eq!(water.models[0].atoms[0].element, Some(Element::O));

    let mut opts = ReadOptions::default();
    opts.radius_column = Some("radius".into());
    let methane = read_path(fixture("methane.extxyz"), &opts).unwrap();
    let atoms = methane.select(&everything(), &mut d).unwrap();
    let r = assign(&atoms, &RadiusOptions::new(RadiusSet::Alvarez), &mut d).unwrap();
    assert_eq!(r, [1.70f32, 1.10, 1.10, 1.10, 1.10].map(f64::from));
}

#[test]
fn detection_from_content_and_gzip_bytes() {
    let gz = std::fs::read(fixture("1CRN.cif.gz")).unwrap();
    let s = read_bytes(&gz, &ReadOptions::default()).unwrap();
    assert_eq!((s.source.format, s.atom_count()), (Format::Mmcif, 327));
    assert_eq!(s.source.data_block.as_deref(), Some("1CRN"));
    let s = read_bytes(
        &std::fs::read(fixture("water.xyz")).unwrap(),
        &ReadOptions::default(),
    )
    .unwrap();
    assert_eq!(s.source.format, Format::Xyz);
    let missing = read_path(fixture("does-not-exist.pdb"), &ReadOptions::default());
    assert!(matches!(missing, Err(Error::Io { path: Some(_), .. })));
}

#[test]
fn large_mmcif_beyond_pdb_format_limits() {
    // 4V6X: 237 685 atoms, > 99 999 atoms and multi-character chain IDs.
    let Some(path) = cached("4V6X.cif.gz") else {
        return;
    };
    let s = read_path(path, &ReadOptions::default()).unwrap();
    assert_eq!(heavy_atoms_model_1(&s), 237_685);
    assert!(s.models[0].atoms.iter().any(|a| a.serial > 99_999));
    assert!(s.models[0].atoms.iter().any(|a| s.str(a.chain).len() > 1));
}

#[test]
#[ignore = "parses a 2.4 M-atom file; run with --include-ignored (nightly CI)"]
fn very_large_mmcif_3j3q() {
    let Some(path) = cached("3J3Q.cif.gz") else {
        return;
    };
    let s = read_path(path, &ReadOptions::default()).unwrap();
    assert_eq!(heavy_atoms_model_1(&s), 2_440_800);
}
