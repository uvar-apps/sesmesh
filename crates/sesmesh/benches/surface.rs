//! End-to-end surface computation and OBJ formatting (Plan T4.7, T4.6):
//! arrangement, classification, sampling, Marching Cubes with exact
//! refinement, assembly and validation. Inputs from
//! `tests/data/{fixtures,cache}` (`cargo xtask fetch --set small --set
//! bench`); missing files are skipped. Probe 1.4 Å, Bondi radii.

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use sesmesh::export::obj::{ObjOptions, write_obj};
use sesmesh::io::{ReadOptions, read_path};
use sesmesh::radii::{RadiusOptions, assign};
use sesmesh::{Diagnostics, Selection, Sphere, SurfaceParams, compute_surface};

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

fn surface(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface");
    group.sample_size(10);
    let cases = [
        ("1CRN", "tests/data/fixtures/1CRN.cif.gz", &[0.5, 0.25][..]),
        ("4HHB", "tests/data/cache/4HHB.cif.gz", &[0.5, 0.25][..]),
        ("1AON", "tests/data/cache/1AON.cif.gz", &[0.5][..]),
    ];
    for (name, rel, spacings) in cases {
        let Some(spheres) = spheres(rel) else {
            eprintln!("skipping {name}: {rel} not found");
            continue;
        };
        for &h in spacings {
            let params = SurfaceParams::ses(1.4).with_spacing(h);
            let out = compute_surface(&spheres, &params).expect("valid");
            eprintln!(
                "{name} h={h}: {} vertices, {} triangles, {:.2} evals/vertex, {:?}",
                out.stats.vertices,
                out.stats.triangles,
                out.stats.refine_evaluations as f64 / out.stats.vertices.max(1) as f64,
                out.stats.timings
            );
            group.bench_function(format!("{name}/h{h}"), |b| {
                b.iter(|| black_box(compute_surface(&spheres, &params).expect("valid")));
            });
            if name == "4HHB" && h == 0.25 {
                let mut buf = Vec::with_capacity(128 << 20);
                group.bench_function("obj/4HHB/h0.25", |b| {
                    b.iter(|| {
                        buf.clear();
                        write_obj(&mut buf, &out.mesh, &ObjOptions::default()).expect("write");
                        black_box(buf.len())
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, surface);
criterion_main!(benches);
