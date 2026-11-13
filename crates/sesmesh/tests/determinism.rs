// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Determinism suite (Plan T5.5, NFR-DET): byte-identical OBJ output for
//! 1/2/4/8/16 threads and repeated runs, on real structures.

#![cfg(all(feature = "parallel", feature = "gzip"))]

use std::path::PathBuf;

use sesmesh::export::obj::{ObjOptions, write_obj};
use sesmesh::io::{ReadOptions, read_path};
use sesmesh::radii::{RadiusOptions, assign};
use sesmesh::{Diagnostics, Selection, Sphere, SurfaceParams, compute_surface};

fn spheres(rel: &str) -> Option<Vec<Sphere>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(rel);
    let structure = read_path(&path, &ReadOptions::default()).ok()?;
    let mut diags = Diagnostics::new();
    let atoms = structure.select(&Selection::default(), &mut diags).ok()?;
    let radii = assign(&atoms, &RadiusOptions::default(), &mut diags).ok()?;
    Sphere::from_atoms(&atoms, &radii).ok()
}

fn obj(spheres: &[Sphere], h: f64, threads: usize) -> Vec<u8> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| {
            let out = compute_surface(spheres, &SurfaceParams::ses(1.4).with_spacing(h)).unwrap();
            let mut buf = Vec::new();
            write_obj(&mut buf, &out.mesh, &ObjOptions::default()).unwrap();
            buf
        })
}

fn check(rel: &str, h: f64, threads: &[usize]) {
    let Some(s) = spheres(rel) else {
        eprintln!("skipping {rel}: not found (cargo xtask fetch --set bench)");
        return;
    };
    let reference = obj(&s, h, 1);
    assert!(reference.len() > 50_000);
    for &t in threads {
        assert!(obj(&s, h, t) == reference, "{rel}: {t} threads differ");
    }
    assert!(obj(&s, h, 1) == reference, "{rel}: repeated run differs");
}

#[test]
fn crambin_any_thread_count() {
    let h = if cfg!(debug_assertions) { 0.7 } else { 0.4 };
    check("fixtures/1CRN.cif.gz", h, &[2, 4, 8, 16]);
}

#[test]
fn trypsin_any_thread_count() {
    let h = if cfg!(debug_assertions) { 1.0 } else { 0.5 };
    check("fixtures/2PTN.cif.gz", h, &[3, 16]);
}

/// Larger inputs from the download cache (nightly: `--include-ignored`).
#[test]
#[ignore = "needs tests/data/cache (cargo xtask fetch --set bench); slow"]
fn large_structures_any_thread_count() {
    check("cache/4HHB.cif.gz", 0.25, &[2, 8, 16]);
    check("cache/1AON.cif.gz", 0.5, &[4, 16]);
}
