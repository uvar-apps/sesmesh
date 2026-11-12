//! Parser throughput (Plan T1.11). Inputs come from `tests/data/cache/`
//! (`cargo xtask fetch --set bench`) and `data/`; missing files are skipped.
//! Set `SESMESH_BENCH_LARGE=1` to include 3J3Q (2.44 M atoms, ~200 MB mmCIF).

use std::hint::black_box;
use std::io::Read;
use std::path::PathBuf;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sesmesh::io::{Format, ReadOptions, read_bytes};

fn load(rel: &str) -> Option<Vec<u8>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    let raw = std::fs::read(&path).ok()?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&raw[..])
            .read_to_end(&mut out)
            .ok()?;
        Some(out)
    } else {
        Some(raw)
    }
}

fn parsers(c: &mut Criterion) {
    let mut cases = vec![
        ("pdb/4HHB", "tests/data/cache/4HHB.pdb.gz", Format::Pdb),
        ("pdb/1AON", "data/1aon.pdb", Format::Pdb),
        ("mmcif/4HHB", "tests/data/cache/4HHB.cif.gz", Format::Mmcif),
        ("mmcif/1AON", "tests/data/cache/1AON.cif.gz", Format::Mmcif),
        ("mmcif/4V6X", "tests/data/cache/4V6X.cif.gz", Format::Mmcif),
    ];
    if std::env::var_os("SESMESH_BENCH_LARGE").is_some() {
        cases.push(("mmcif/3J3Q", "tests/data/cache/3J3Q.cif.gz", Format::Mmcif));
    }
    let mut group = c.benchmark_group("parse");
    group.sample_size(10);
    for (name, rel, format) in cases {
        let Some(data) = load(rel) else {
            eprintln!("skipping {name}: {rel} not found");
            continue;
        };
        let mut options = ReadOptions::default();
        options.format = Some(format);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_function(name, |b| {
            b.iter(|| read_bytes(black_box(&data), &options).expect("parses"));
        });
    }
    group.finish();
}

criterion_group!(benches, parsers);
criterion_main!(benches);
