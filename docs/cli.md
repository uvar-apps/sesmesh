# sesmesh command-line manual

```text
sesmesh [surface] <INPUTS>... [OPTIONS]     compute surfaces (default command)
sesmesh info <INPUT> [OPTIONS]              summarize a structure
sesmesh radii [--set bondi|alvarez|protor]  print a radius table
sesmesh completions bash|zsh|fish|powershell|elvish   shell completion script
sesmesh man                                 manual page (roff)
sesmesh --help | --version
```

`sesmesh 1crn.cif` is short for `sesmesh surface 1crn.cif`. Every option has
built-in help (`sesmesh surface --help`). Output formats are described in
[04-output-formats.md](04-output-formats.md); accuracy and performance in
[validation-report.md](validation-report.md).

## Quick examples

```bash
sesmesh 1crn.cif                                   # SES at 0.5 Å → 1crn.obj
sesmesh 4hhb.cif.gz -q high -o hb.ply              # 0.25 Å, binary PLY
sesmesh 1aon.cif.gz --cavities remove --group-components --vertex-colors element
sesmesh ligand.xyz --radii alvarez --probe 1.2 -s 0.1 -o ligand.obj
sesmesh 1k4r.cif.gz --assembly 1 -q draft -o capsid.obj   # 545 k-atom capsid
sesmesh 1aon.cif.gz -q high --decimate 0.25 -o gro.ply  # fine grid, 4× fewer triangles
sesmesh 3j3q.cif.gz --max-memory 4G --auto-coarsen # coarsen until it fits
sesmesh 2ptn.pdb --stats - -o /dev/null            # statistics only (JSON)
sesmesh *.cif.gz --output-dir meshes --keep-going  # batch
sesmesh info 1k4r.cif.gz                           # models, chains, assemblies
```

## Inputs

PDB (fixed columns, hybrid-36), PDBx/mmCIF, XYZ / extended XYZ and PQR, each
optionally gzip-compressed. The format comes from the file name, then the
content.

| Option | Default | Meaning |
|---|---|---|
| `--format auto\|pdb\|mmcif\|xyz\|pqr` | `auto` | Force the input format |
| `--model N` | first model | Model number (PDB `MODEL`, mmCIF `pdbx_PDB_model_num`) |
| `--frame N` | | 0-based model position (XYZ trajectories) |
| `--block NAME` | first block with atoms | mmCIF data block |
| `--assembly ID` | deposited coordinates | Build a biological assembly (mmCIF `pdbx_struct_assembly_gen`, PDB `REMARK 350`); `sesmesh info` lists them |

## Atom selection

| Option | Default | Meaning |
|---|---|---|
| `--altloc highest\|first\|all\|<letter>` | `highest` | Alternate locations, decided per residue (`highest`: largest total occupancy) |
| `--hydrogens` | off | Keep hydrogen (and deuterium) atoms |
| `--waters` | off | Keep waters (HOH, WAT, DOD, …) |
| `--no-hetatm` | off | Drop hetero atoms (ligands, ions) |
| `--chains A,B` | all | Keep only these chains |
| `--chain-ids auth\|label` | `auth` | Which chain IDs `--chains` uses |
| `--exclude-resn NAME,...` | | Drop residues by name |

## Radii

| Option | Default | Meaning |
|---|---|---|
| `--radii bondi\|alvarez\|protor\|uniform:<r>\|file:<path>` | `bondi` | Radius set; `file:` reads TOML rules (below) |
| `--no-file-radii` | off | Ignore radii from the input (PQR, extended XYZ) |
| `--unknown-radius R` | 1.8 | Radius for atoms the set does not cover (with a warning) |
| `--strict-radii` | off | Fail instead of using the unknown radius |

TOML rules (`--radii file:my.toml`), layered over a base set:

```toml
base = "bondi"        # bondi | alvarez | protor | none
fallback = 1.8        # optional
[elements]
Fe = 2.0
[atoms]               # "RESIDUE:ATOM", either part may be "*"
"HEM:FE" = 1.6
"*:OXT" = 1.52
```

`sesmesh radii --set protor` prints a table.

## Surface

| Option | Default | Meaning |
|---|---|---|
| `--surface ses\|sas\|vdw` | `ses` | Solvent excluded (Connolly), solvent accessible, or van der Waals surface |
| `--probe R` | 1.4 | Probe radius in Å (ignored for vdW) |
| `--cavities keep\|remove\|only` | `keep` | Internal cavities: keep (exact SES), remove with their content (MSMS-like), or keep only them |
| `--min-cavity-volume V` | 0 | Drop cavities below V Å³ |
| `--min-component-area A` | 0 | Drop components below A Å² |
| `--drop-fragments` | off | Drop sub-grid fragments: tiny solid SES components without a convex atom patch, cut off by the grid from reentrant sheets thinner than the spacing (Design §9.1) |

## Resolution

| Option | Default | Meaning |
|---|---|---|
| `-q, --quality draft\|low\|medium\|high\|ultra` | `medium` | Grid spacing 1.0 / 0.7 / 0.5 / 0.25 / 0.1 Å |
| `-s, --grid-spacing H` | | Grid spacing in Å (overrides `-q`) |
| `--density D` | | Target vertices per Å² (h = √(1.5/D); overrides `-q`) |
| `--smooth N` | 0 | Taubin smoothing rounds, then exact re-projection onto the surface |
| `--decimate R` | | Decimate to the fraction R (0–1] of the triangles |
| `--max-triangles N` | | Decimate to at most N triangles (with `--decimate`: the smaller target wins) |
| `--decimate-error E` | no bound | Never collapse an edge whose quadric error exceeds E Å; wins over the targets |
| `--refine exact\|linear` | `exact` | Vertex placement: exact root on each grid edge, or linear interpolation (faster) |
| `--brick-size N` | 10 | Brick edge in cells (expert) |

Errors decrease as h²: on proteins about −0.2 % area and −0.02 % volume at
0.25 Å, −0.8 % / −0.08 % at 0.5 Å. Triangles grow as 1/h² (≈ 3 per Å² at
1.0 Å, 48 at 0.25 Å).

Decimation runs after smoothing: quadric-error half-edge collapses, so no
new positions are created and every retained vertex keeps its exact place on
the surface, its analytic normal and its atom. The result stays a closed,
oriented manifold; collapses that would flip a triangle, create a sliver or
shrink a component below a tetrahedron are skipped, so a target may not be
reached exactly. A fine grid decimated (`-q high --decimate 0.25`) keeps
small features better than a coarse grid with the same triangle count.

## Output

| Option | Default | Meaning |
|---|---|---|
| `-o, --output PATH` | `<input stem>.<ext>` in `--output-dir` | Output file; `-` for standard output (OBJ, PLY); a `.gz` suffix compresses any format |
| `--output-format obj\|ply\|msms` | from the `-o` extension (without `.gz`) | `.ply` → PLY, `.vert`/`.face` → MSMS, otherwise OBJ |
| `--output-dir DIR` | current directory | Output directory for default names (batch mode); created if missing |
| `--precision N` | 4 | Decimals in OBJ (0–9) |
| `--no-normals` | off | Omit normals |
| `--face-normals` | off | Area-weighted face-averaged normals instead of the analytic ones |
| `--group-components` | off | One OBJ group per component (`component_N`, `cavity_N`) |
| `--vertex-colors element\|chain\|patch\|component` | | Per-vertex colours (`v x y z r g b` in OBJ, `uchar` in PLY) |
| `--palette FILE` | bundled palette | Element colours in the `atom-colors.json` format |
| `--vertex-data FILE` | | Per-vertex CSV: nearest atom, patch, component |
| `--atom-areas FILE` | | Per-atom SES area CSV |
| `--stats FILE` | | Statistics JSON (`-`: standard output) |

## Batch mode

Several inputs write `<stem>.<ext>` into `--output-dir`. `-o` is then a usage
error, and `--stats` / `--atom-areas` / `--vertex-data` name directories
(`<stem>.stats.json`, `<stem>.atoms.csv`, `<stem>.vertices.csv`). Without
`--keep-going` the batch stops at the first failing input; with it, every
input is tried and the summary reports "N of M inputs failed".

Missing directories are created (with their parents) before the first input
is read: `--output-dir` and the side-file directories of a batch, and for a
single input the parent directories of `-o` and of the side files (not for
`-`). A directory that cannot be created is a runtime error (exit code 1).

## Performance and diagnostics

| Option | Default | Meaning |
|---|---|---|
| `-j, --threads N` | all logical cores | Worker threads (output is identical for any N) |
| `--simd auto\|scalar\|sse4.2\|avx2\|avx512\|neon` | `auto` | Field kernel instruction set; x86 levels are capped at what the CPU supports |
| `--max-memory BYTES` | | Refuse inputs whose estimated memory exceeds the limit (`8G`, `512M`); exit code 4 |
| `--auto-coarsen` | off | With `--max-memory`: grow the grid spacing until the estimate fits instead of refusing (the spacing used is reported) |
| `--quiet` | off | Print errors only |
| `--progress` | off | Progress on standard error |
| `--timings` | off | Stage timings and the kernel used |

Standard error shows warnings (e.g. unknown radii), a summary (atoms,
vertices, triangles, components, cavities, area, volume) and the output path.

Ctrl-C (SIGINT) cancels the computation within about 100 ms and exits with
code 130; in batch mode no further input is started. A closed standard
output (`sesmesh x.cif -o - | head`) is not an error.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Runtime error (I/O, empty selection, invalid input data, batch failure) |
| 2 | Usage error (unknown option, invalid value, conflicting options) |
| 3 | Input parse error (malformed file, unknown format, invalid palette) |
| 4 | Resource limit exceeded (`--max-memory`, decompression limit, assembly size) |
| 130 | Interrupted (Ctrl-C) |

## `sesmesh info`

Prints the file format and data block, models, atom counts, chains with atom
counts, residues, element composition, hetero/hydrogen/water/alt-loc counts,
the default selection, bounding box, radius coverage for `--radii`, and the
biological assemblies with their copy counts. Takes the input and selection
options of `surface`.

## `sesmesh radii`

Prints the element table of a radius set (`Z symbol radius`) and, for
`protor`, the per-atom classes.

## `sesmesh completions` and `sesmesh man`

`sesmesh completions <SHELL>` prints a completion script for bash, zsh, fish,
PowerShell or elvish; `sesmesh man` prints the manual page (roff). Release
archives contain `sesmesh.1` and the bash, zsh, fish and PowerShell scripts.

```bash
sesmesh completions zsh > ~/.zfunc/_sesmesh        # zsh (with fpath+=~/.zfunc)
sesmesh completions bash > ~/.local/share/bash-completion/completions/sesmesh
sesmesh completions fish > ~/.config/fish/completions/sesmesh.fish
sesmesh man > ~/.local/share/man/man1/sesmesh.1
```
