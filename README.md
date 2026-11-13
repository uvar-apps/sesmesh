# sesmesh

Fast, accurate **Solvent Excluded Surface** (SES, also called the Connolly or
molecular surface) meshing for molecular structures, written in Rust.

* Reads PDB, PDBx/mmCIF and XYZ files (optionally gzip-compressed).
* Computes the SES exactly via its implicit distance representation, sampled
  on a sparse, narrow-band brick grid. SAS and van der Waals surfaces are also
  supported.
* Writes closed, consistently oriented triangle meshes with analytic normals
  (Wavefront OBJ, later also PLY and MSMS formats).
* Lets you control resolution and level of detail (grid spacing, presets,
  decimation).
* Multi-threaded (rayon) and SIMD-accelerated with runtime CPU dispatch
  (AVX2 / AVX-512 / NEON).
* Ships as a reusable library crate plus a command-line tool.

> **Status: Phases 0–8 done (decimation pending).** Structure files → closed, oriented SES / SAS /
> vdW meshes with analytic normals → OBJ, from the library and the `sesmesh`
> CLI, multi-threaded and SIMD-accelerated, with cavity handling, components,
> vertex colours, per-atom areas, JSON statistics, density and smoothing
> controls, PLY/MSMS output, biological assemblies and batch mode. Decimation,
> the validation campaign and the release follow (Phases 8–10, see the
> [implementation plan](docs/03-implementation-plan.md)).

## Quick start

```bash
cargo build --release
target/release/sesmesh 1crn.cif -o 1crn.obj            # SES, 0.5 Å grid
target/release/sesmesh 4hhb.cif.gz -q high --timings    # 0.25 Å grid
target/release/sesmesh ligand.xyz --radii alvarez --probe 1.2 -s 0.1 -o lig.obj
target/release/sesmesh 1crn.cif --cavities remove --vertex-colors element --stats stats.json
target/release/sesmesh 1k4r.cif.gz --assembly 1 -q draft -o capsid.ply    # 545k-atom capsid
target/release/sesmesh *.cif.gz --output-dir meshes --keep-going           # batch
target/release/sesmesh info 1crn.pdb
target/release/sesmesh radii --set protor
```

Run `sesmesh surface --help` for all options (selection, radii, surface kind,
cavities, resolution, normals, colours, side files, threads). Output formats
are described in [docs/04-output-formats.md](docs/04-output-formats.md). Exit codes: 0 success, 1 runtime
error, 2 usage error, 3 input parse error, 4 resource limit.

As a library:

```rust
use sesmesh::export::obj::{write_obj_path, ObjOptions};
use sesmesh::{compute_surface, io, radii, Diagnostics, Selection, Sphere, SurfaceParams};

fn main() -> sesmesh::Result<()> {
    let structure = io::read_path("1crn.cif", &io::ReadOptions::default())?;
    let mut diags = Diagnostics::new();
    let atoms = structure.select(&Selection::default(), &mut diags)?;
    let radii = radii::assign(&atoms, &radii::RadiusOptions::default(), &mut diags)?;
    let spheres = Sphere::from_atoms(&atoms, &radii)?;
    let out = compute_surface(&spheres, &SurfaceParams::ses(1.4).with_spacing(0.25))?;
    println!("{} triangles, area {:.1} Å²", out.stats.triangles, out.stats.area);
    write_obj_path("1crn.obj", &out.mesh, &ObjOptions::default())
}
```

## Documentation

* [Product specification](docs/01-product-specification.md): requirements,
  literature survey, references
* [Design](docs/02-design.md): mathematics, architecture, algorithms
* [Implementation plan](docs/03-implementation-plan.md): phases, tasks,
  milestones
* [Output formats](docs/04-output-formats.md): OBJ, CSV side files, JSON
  statistics

## Workspace layout

| Path | Contents |
|---|---|
| `crates/sesmesh` | Library crate |
| `crates/sesmesh-cli` | `sesmesh` command-line tool |
| `crates/sesmesh-testkit` | Test oracles and analytic references (development only) |
| `xtask` | Development automation (`cargo xtask ...`) |
| `tests/data` | Test fixtures, download sets and checksums ([README](tests/data/README.md)) |
| `docs` | Specification, design and plan |

## Building

Requires Rust 1.89 or newer.

```bash
cargo build --release
```

```bash
cargo test --workspace
```

To download the optional test and benchmark structures from the RCSB PDB:

```bash
cargo xtask fetch --set small
```

## Licence

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT licence ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work, as defined in the Apache-2.0 licence,
shall be dual licensed as above, without any additional terms or conditions.
