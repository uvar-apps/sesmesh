# Fuzz targets

Requires a nightly toolchain and [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

| Target | Input |
|---|---|
| `pdb`, `mmcif`, `xyz`, `pqr` | parse as that format, then select atoms and assign radii |
| `detect` | format detection and gzip handling on arbitrary bytes |
| `cif_lexer` | the CIF tokenizer alone |

```bash
cargo +nightly fuzz run pdb -- -max_total_time=60
```

Seed a corpus from the fixtures (decompressed) to start from realistic inputs:

```bash
mkdir -p fuzz/corpus/pdb && for f in tests/data/fixtures/*.pdb.gz; do gzcat "$f" > "fuzz/corpus/pdb/$(basename "$f" .gz)"; done
```

Crashes found by fuzzing are minimized and added as regression tests in the
library (`crates/sesmesh`).

## Long campaigns

`fuzz/campaign.sh` runs the ≥ 24 CPU-hour campaign of Plan T9.8 (#22) on
x86-64 Linux with a nightly toolchain and cargo-fuzz: `build` builds every
target (AddressSanitizer, release, debug assertions); `run TARGET WORKERS
SECONDS` seeds `corpus/TARGET` from the fixtures and runs libFuzzer in fork
mode with WORKERS processes for SECONDS (WORKERS × SECONDS ≥ 86 400 for
24 CPU-hours), keeping crashes, timeouts and out-of-memory inputs in
`artifacts/TARGET/` and going on. GNU time's report in `logs/TARGET.time`
gives the CPU time actually spent.

```bash
bash fuzz/campaign.sh build
```

```bash
bash fuzz/campaign.sh run pdb 4 22500
```
