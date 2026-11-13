// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Radius sets: built-in tables, a uniform radius and custom TOML rules.
//!
//! ```text
//! cargo run --release -p sesmesh --example custom_radii
//! ```

use sesmesh::radii::{RadiusOptions, RadiusRules, RadiusSet, assign};
use sesmesh::{Diagnostics, Selection, Sphere, SurfaceParams, compute_surface, io};

fn main() -> sesmesh::Result<()> {
    let pdb = "\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.246   2.390   0.000  1.00  0.00           O
HETATM    5 ZN    ZN A 101       4.000   0.000   1.000  1.00  0.00          ZN
";
    let structure = io::read_bytes(pdb.as_bytes(), &io::ReadOptions::default())?;
    let mut diags = Diagnostics::new();
    let atoms = structure.select(&Selection::default(), &mut diags)?;

    // Custom rules on top of Bondi: a zinc radius and a per-atom override.
    let rules = RadiusRules::from_toml_str(
        r#"
        base = "bondi"
        [elements]
        Zn = 1.39
        [atoms]
        "GLY:CA" = 1.80
        "#,
    )?;
    for (name, set) in [
        ("bondi", RadiusSet::Bondi),
        ("alvarez", RadiusSet::Alvarez),
        ("protor", RadiusSet::ProtOr),
        ("uniform:1.6", RadiusSet::Uniform(1.6)),
        ("custom", RadiusSet::Custom(Box::new(rules))),
    ] {
        let radii = assign(&atoms, &RadiusOptions::new(set), &mut diags)?;
        let spheres = Sphere::from_atoms(&atoms, &radii)?;
        let out = compute_surface(&spheres, &SurfaceParams::ses(1.4).with_spacing(0.3))?;
        println!("{name:12} radii {radii:?} → area {:.1} Å²", out.stats.area);
    }
    for w in diags.warnings() {
        eprintln!("warning: {w}");
    }
    Ok(())
}
