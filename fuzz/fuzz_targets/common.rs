// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// Shared by the format fuzz targets: parse, then run selection and radius
// assignment on whatever parsed. Every outcome must be Ok or Err, never a
// panic.
use sesmesh::io::{Format, ReadOptions, read_bytes};
use sesmesh::radii::{RadiusOptions, assign};
use sesmesh::{Diagnostics, Selection};

pub fn exercise(data: &[u8], format: Option<Format>) {
    let mut options = ReadOptions::default();
    options.format = format;
    // Keep gzip bombs within the fuzzer's memory limit.
    options.max_decompressed_bytes = 64 << 20;
    let Ok(structure) = read_bytes(data, &options) else { return };
    let mut diags = Diagnostics::new();
    for selection in [Selection::default(), everything()] {
        if let Ok(atoms) = structure.select(&selection, &mut diags) {
            let _ = assign(&atoms, &RadiusOptions::default(), &mut diags);
        }
    }
}

fn everything() -> Selection {
    let mut s = Selection::default();
    s.hydrogens = true;
    s.waters = true;
    s.alt_loc = sesmesh::structure::AltLocPolicy::First;
    s
}
