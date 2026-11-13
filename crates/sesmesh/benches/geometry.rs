// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SAS arrangement construction (Plan T2.9). Inputs: `tests/data/cache/`
//! (`cargo xtask fetch --set small --set bench`), default selection, Bondi
//! radii, probe 1.4 Å. Missing files are skipped.

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use sesmesh::geom::arrangement_stats;
use sesmesh::io::{ReadOptions, read_path};
use sesmesh::radii::{RadiusOptions, assign};
use sesmesh::{Diagnostics, Selection, Sphere};

fn spheres(rel: &str) -> Option<Vec<Sphere>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    let structure = read_path(&path, &ReadOptions::default()).ok()?;
    let mut diags = Diagnostics::new();
    let atoms = structure.select(&Selection::default(), &mut diags).ok()?;
    let radii = assign(&atoms, &RadiusOptions::default(), &mut diags).ok()?;
    Sphere::from_atoms(&atoms, &radii).ok()
}

fn arrangement(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrangement");
    group.sample_size(10);
    for (name, rel) in [
        ("1CRN", "tests/data/fixtures/1CRN.cif.gz"),
        ("4HHB", "tests/data/cache/4HHB.cif.gz"),
        ("1AON", "tests/data/cache/1AON.cif.gz"),
        ("4V6X", "tests/data/cache/4V6X.cif.gz"),
    ] {
        let Some(spheres) = spheres(rel) else {
            eprintln!("skipping {name}: {rel} not found");
            continue;
        };
        let stats = arrangement_stats(&spheres, 1.4).expect("valid input");
        eprintln!("{name}: {stats:?}");
        group.bench_function(name, |b| {
            b.iter(|| arrangement_stats(black_box(&spheres), 1.4).expect("valid input"));
        });
    }
    group.finish();
}

criterion_group!(benches, arrangement);
criterion_main!(benches);
