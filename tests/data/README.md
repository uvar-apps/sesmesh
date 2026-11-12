# Test data

| Location | Content | In git |
|---|---|---|
| `tests/data/fixtures/` | Small structure files tests need offline (see its [README](fixtures/README.md)). PDB archive data is CC0. Keep the total under 5 MB. | yes |
| `tests/data/sets.toml` | Named download sets (`small`, `bench`, later `validation` and `sweep`). Edited by hand. | yes |
| `tests/data/sets.lock` | SHA-256 of every downloaded file. Written by `cargo xtask fetch`. | yes |
| `tests/data/cache/` | Downloaded files. | no (git-ignored) |
| `tests/data/reference/` (Plan T9.2) | Numbers produced offline by external tools (MSMS, NanoShaper, EDTSurf, sesa) with tool versions and command lines. Only derived numbers, never the tools' output files. | yes |

## Fetching

```bash
cargo xtask sets
```

```bash
cargo xtask fetch --set small
```

The first download of a file records its checksum in `sets.lock` (commit it).
Later downloads must match. If the PDB remediates an entry and the file
changes, the fetch fails. Review the change, then run
`cargo xtask fetch --set <name> --update-lock`.

Tests that need downloaded data must skip with a clear message when the file
is not in the cache, so `cargo test` keeps working offline.

## Line endings

`.gitattributes` normalizes text to LF. Fixtures that must keep CRLF (parser
tests) go in a directory marked `-text` in `.gitattributes`.
