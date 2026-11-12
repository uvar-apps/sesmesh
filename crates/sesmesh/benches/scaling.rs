//! Thread scaling of `compute_surface` (Plan T5.7): 1, 2, 4, 8 threads and
//! all cores, on 4HHB at 0.5 Å (`tests/data/cache`, from `cargo xtask fetch
//! --set bench`), falling back to the 2PTN fixture.

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
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

fn scaling(c: &mut Criterion) {
    let (name, s) = [
        ("4HHB", "tests/data/cache/4HHB.cif.gz"),
        ("2PTN", "tests/data/fixtures/2PTN.cif.gz"),
    ]
    .into_iter()
    .find_map(|(n, p)| spheres(p).map(|s| (n, s)))
    .expect("2PTN fixture");
    let params = SurfaceParams::ses(1.4).with_spacing(0.5);
    let mut group = c.benchmark_group("scaling");
    group.sample_size(10);
    let max = std::thread::available_parallelism().map_or(1, usize::from);
    let mut counts = vec![1, 2, 4, 8];
    counts.retain(|&t| t < max);
    counts.push(max);
    for threads in counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        group.bench_function(format!("{name}/h0.5/t{threads}"), |b| {
            b.iter(|| pool.install(|| black_box(compute_surface(&s, &params).expect("valid"))));
        });
    }
    group.finish();
}

criterion_group!(benches, scaling);
criterion_main!(benches);
