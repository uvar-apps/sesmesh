// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structure file → SES mesh → OBJ.
//!
//! ```text
//! cargo run --release -p sesmesh --example file_to_obj -- <input> [output.obj] [spacing]
//! ```
//!
//! Without arguments it meshes the bundled crambin fixture into the system
//! temporary directory.

use std::path::PathBuf;

use sesmesh::export::obj::{ObjOptions, write_obj_path};
use sesmesh::io::{ReadOptions, read_path};
use sesmesh::radii::{RadiusOptions, assign};
use sesmesh::{Diagnostics, Selection, Sphere, SurfaceParams, compute_surface};

fn main() -> sesmesh::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/fixtures/1CRN.cif.gz"),
        PathBuf::from,
    );
    let output = args.next().map_or_else(
        || std::env::temp_dir().join("sesmesh-example.obj"),
        PathBuf::from,
    );
    let spacing: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);

    // Read, select (first model, no hydrogens or waters), assign radii.
    let structure = read_path(&input, &ReadOptions::default())?;
    let mut diags = Diagnostics::new();
    let atoms = structure.select(&Selection::default(), &mut diags)?;
    let radii = assign(&atoms, &RadiusOptions::default(), &mut diags)?;
    for w in diags.warnings() {
        eprintln!("warning: {w}");
    }

    let spheres = Sphere::from_atoms(&atoms, &radii)?;
    let out = compute_surface(&spheres, &SurfaceParams::ses(1.4).with_spacing(spacing))?;
    println!(
        "{}: {} atoms → {} triangles, {} components ({} cavities), area {:.1} Å², volume {:.1} Å³",
        input.display(),
        atoms.len(),
        out.stats.triangles,
        out.stats.components,
        out.stats.cavities,
        out.stats.area,
        out.stats.volume
    );
    write_obj_path(&output, &out.mesh, &ObjOptions::default())?;
    println!("wrote {}", output.display());
    Ok(())
}
