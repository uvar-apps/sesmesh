# Cross-tool comparison harness

Measures sesmesh against MSMS, NanoShaper, ESES, EDTSurf, UCSF ChimeraX and
sesa on the same spheres (GitHub issues #5, #6; x86 measurements #8, #10).
Results and discussion: [`docs/comparison.md`](../../docs/comparison.md);
condensed data: `docs/validation/compare/`; stored references for the
tests: `tests/data/reference/`.

| File | Purpose |
|---|---|
| `Containerfile` | Ubuntu 24.04 image with every program (MSMS, sesa and ChimeraX from their downloads, EDTSurf from the distribution, Rust, trimesh) |
| `build-tools.sh` | Builds sesmesh, `meshcheck`, NanoShaper and ESES inside the container into the mounted bench directory |
| `meshcheck/` | Stand-alone Rust helper: writes the sphere files (`spheres`), analyses any program's mesh against the exact SES (`check`), probes the field at points (`probe`) |
| `run.py` | Runs one program on entries at a grid spacing and appends one JSON object per run (time, memory, program-reported numbers, meshcheck analysis) |
| `campaign.sh` | Starts detached, CPU-pinned, memory-capped container streams of `run.py` invocations |
| `perf.sh` | The performance matrix (two streams, one per host) |
| `report.py` | JSONL → `runs.csv`, `cavities.csv`, `performance.csv`, `summary.md` and `tests/data/reference/*.toml` (standard library only) |

## Reproduce

Downloads (into `~/sesmesh-bench/dl/` on the Linux host):

* MSMS 2.6.2 Linux tarball, from the Sanner lab GitHub releases
  (non-commercial licence);
* `sesaV001.tar.gz` from github.com/wlincong/SESexact (GPL-3);
* UCSF ChimeraX 1.12 Ubuntu 24.04 `.deb` (accept the non-commercial
  licence on the download page; SHA-256
  `851d6f89…db30ab`);
* sources into `~/sesmesh-bench/src/`: this repository (`sesmesh/`),
  `git clone https://github.com/shi-scala/NanoShaper`,
  `git clone https://github.com/WeilabMSU/ESES`;
* structures: `cargo xtask fetch --set validation --set sweep --set bench`,
  then copy `tests/data/cache` to `~/sesmesh-bench/data/cache`.

```bash
podman build -t sesmesh-compare -f ~/sesmesh-bench/src/sesmesh/tools/compare/Containerfile ~/sesmesh-bench
```

```bash
podman run --rm -v ~/sesmesh-bench:/work sesmesh-compare bash /work/src/sesmesh/tools/compare/build-tools.sh
```

```bash
podman run --rm -v ~/sesmesh-bench:/work sesmesh-compare python3 /work/src/sesmesh/tools/compare/run.py synth
```

```bash
podman run --rm -v ~/sesmesh-bench:/work sesmesh-compare python3 /work/src/sesmesh/tools/compare/run.py prepare @validation @sweep 1JJ2 3J3Q 1K4R:asm1
```

Campaigns (each stream pinned to its own CPUs; see `docs/comparison.md`
§2 for the split actually used):

```bash
bash ~/sesmesh-bench/src/sesmesh/tools/compare/campaign.sh stream acc-msms 4 12g -- "run --tool msms --spacing 0.5 --threads 1 @validation"
```

```bash
bash ~/sesmesh-bench/src/sesmesh/tools/compare/perf.sh a
```

```bash
python3 tools/compare/report.py RUNS_DIR docs/validation/compare --references tests/data/reference
```

## Notes

* The sphere files keep three decimals of the coordinates (as in the PDB
  archive) in every format, and write every atom as an `ATOM` record.
* ESES reads PQR by fixed columns, only `ATOM` records, and aborts on atom
  names that do not start with C, H, O, N, P or S; its own file
  (`*.eses.pqr`) writes the element's type letter (the type only selects
  ESES's visualization colours).
* NanoShaper 0.7.5 needs `#include <memory>` in `Configuration.h` with
  GCC 13 (`build-tools.sh` adds it) and `-DENABLE_EXE=ON`; it writes a date
  into its OFF header, so determinism hashes skip comment lines.
* MSMS writes component k of `-all_components` to `mesh_k.vert/.face`;
  `meshcheck` reads them all.
* macOS's `openrsync` silently transfers nothing with `-c`: sync with plain
  `rsync -a`, and touch the Rust sources before rebuilding (rsync keeps
  older mtimes than the last build's artefacts).
