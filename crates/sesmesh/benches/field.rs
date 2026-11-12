//! Field sampling on surface bricks (Plan T3.9): arrangement + brick
//! classification + SIMD kernel, multi-threaded. Inputs from
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
    for (name, rel, spacings, scalar_too) in cases {
        let Some(spheres) = spheres(rel) else {
            eprintln!("skipping {name}: {rel} not found");
            continue;
        };
        for &h in spacings {
            let levels: &[SimdLevel] = if scalar_too {
                &[SimdLevel::Auto, SimdLevel::Scalar]
            } else {
                &[SimdLevel::Auto]
            };
            for &level in levels {
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
