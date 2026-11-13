// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Field sampling on surface bricks (Plan T3.9, T6.7): arrangement + brick
//! classification + SIMD kernel, multi-threaded, at every instruction set
//! the CPU supports. Inputs from
//! `tests/data/{fixtures,cache}` (`cargo xtask fetch --set small --set bench`);
//! missing files are skipped. Probe 1.4 Å, Bondi radii, 8³-cell bricks.

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use sesmesh::field::{SimdLevel, SurfaceKind, sample_surface_bricks};
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

fn field(c: &mut Criterion) {
    let mut group = c.benchmark_group("field");
    group.sample_size(10);
    let cases = [
        (
            "1CRN",
            "tests/data/fixtures/1CRN.cif.gz",
            &[0.5, 0.25][..],
            true,
        ),
        (
            "4HHB",
            "tests/data/cache/4HHB.cif.gz",
            &[0.5, 0.25][..],
            true,
        ),
        ("1AON", "tests/data/cache/1AON.cif.gz", &[0.5][..], false),
    ];
    for (name, rel, spacings, all_levels) in cases {
        let Some(spheres) = spheres(rel) else {
            eprintln!("skipping {name}: {rel} not found");
            continue;
        };
        for &h in spacings {
            // Every instruction set this CPU supports (Plan T6.7), or just
            // the default for the large input.
            let mut levels: Vec<SimdLevel> = if all_levels {
                SimdLevel::ALL.to_vec()
            } else {
                vec![SimdLevel::Auto]
            };
            let mut seen = Vec::new();
            levels.retain(|l| match l.effective() {
                Some(name) if *l != SimdLevel::Auto || !all_levels => {
                    let new = !seen.contains(&name);
                    seen.push(name);
                    new
                }
                _ => false,
            });
            for level in levels {
                let run = || {
                    sample_surface_bricks(&spheres, 1.4, SurfaceKind::Ses, h, 8, level)
                        .expect("valid")
                };
                let stats = run();
                eprintln!("{name} h={h} {stats:?}");
                group.bench_function(format!("{name}/h{h}/{}", stats.simd), |b| {
                    b.iter(|| black_box(run()));
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, field);
criterion_main!(benches);
