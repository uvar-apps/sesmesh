# Validation data (Plan Phase 9)

Machine-written results of the Phase 9 campaigns; the analysis is in
[`../validation-report.md`](../validation-report.md).

| File | Written by | Content |
|---|---|---|
| `accuracy.csv` | `cargo run --release -p xtask -- validate` | Validation set (`[sets.validation]` in `tests/data/sets.toml`): area, volume, counts and time per entry and grid spacing, errors relative to the h → 0 extrapolation, `vertices·h²/area` |
| `sweep.csv` | `cargo run --release -p xtask -- sweep` | Robustness sweep (`[sets.sweep]`, 200 random entries, seed 2026): status, counts, time per entry |
| `large.csv` | manual runs (see the report) | Large inputs: time and peak memory |

Reproduce:

```bash
cargo xtask fetch --set validation --set sweep --set bench
```

```bash
cargo run --release -p xtask -- validate
```

```bash
cargo run --release -p xtask -- sweep
```

Inputs: default selection (first model, no hydrogens or waters, highest
occupancy alt-locs), Bondi radii, probe 1.4 Å, SES with cavities kept,
default brick size and SIMD level.
