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
