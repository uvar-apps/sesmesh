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

> **Status: Phases 0–9 done (decimation and cross-tool references pending).** Structure files → closed, oriented SES / SAS /
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
* [Validation report](docs/validation-report.md): accuracy, robustness,
  large inputs, performance
* [CLI manual](docs/cli.md): every option, examples, exit codes
* Library: crate documentation (`cargo doc --open -p sesmesh`) and
  [examples](crates/sesmesh/examples/)
* [API review](docs/api-review.md) (semver policy) and
  [release process](docs/release.md)

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

## How it works and what to cite

sesmesh evaluates the exact implicit SES distance function of Quan & Stamm
[1] on a sparse narrow-band grid, using the SAS arrangement (atom spheres,
intersection circles and probe positions) of Connolly's analytic surface
[2, 3]; it extracts the zero level set with Marching Cubes [4] (uniform
face rule, Flying Edges-style row skipping [5]), places every vertex on the
exact surface and takes normals from the closest point. Optional smoothing
uses Taubin's λ|μ filter [6] followed by exact re-projection. The SES itself
goes back to Lee & Richards [7] and Richards [8].

If you use sesmesh in published work, please cite the methods above:

1. C. Quan, B. Stamm, "Mathematical analysis and calculation of molecular
   surfaces," *J. Comput. Phys.* 322:760–782, 2016.
   doi:[10.1016/j.jcp.2016.07.007](https://doi.org/10.1016/j.jcp.2016.07.007)
2. M. L. Connolly, "Analytical molecular surface calculation," *J. Appl.
   Cryst.* 16(5):548–558, 1983.
   doi:[10.1107/S0021889883010985](https://doi.org/10.1107/S0021889883010985)
3. M. L. Connolly, "Solvent-accessible surfaces of proteins and nucleic
   acids," *Science* 221:709–713, 1983.
4. W. E. Lorensen, H. E. Cline, "Marching cubes: a high resolution 3D surface
   construction algorithm," *Comput. Graph.* 21(4):163–169, 1987.
   doi:[10.1145/37402.37422](https://doi.org/10.1145/37402.37422)
5. W. Schroeder, R. Maynard, B. Geveci, "Flying edges: a high-performance
   scalable isocontouring algorithm," *IEEE LDAV* 2015.
   doi:[10.1109/LDAV.2015.7348069](https://doi.org/10.1109/LDAV.2015.7348069)
6. G. Taubin, "A signal processing approach to fair surface design,"
   *SIGGRAPH '95*, 351–358, 1995.
7. B. Lee, F. M. Richards, "The interpretation of protein structures:
   estimation of static accessibility," *J. Mol. Biol.* 55(3):379–400, 1971.
   doi:[10.1016/0022-2836(71)90324-X](https://doi.org/10.1016/0022-2836(71)90324-X)
8. F. M. Richards, "Areas, volumes, packing, and protein structure," *Annu.
   Rev. Biophys. Bioeng.* 6:151–176, 1977.

Radius sets: Bondi (1964) with Mantina et al. (2009), Alvarez (2013), and
ProtOr (Tsai et al. 1999, classification from FreeSASA). The full reference
list is in the [specification](docs/01-product-specification.md#12-references).

## Licence

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT licence ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work, as defined in the Apache-2.0 licence,
shall be dual licensed as above, without any additional terms or conditions.
