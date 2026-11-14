# Data behind the paper

| File | Source | How it was made |
|---|---|---|
| `analytic.csv` | `paper/scripts/analytic` (this repository's library and `sesmesh-testkit`) | `make data` (or `cargo run --release` in `paper/scripts/analytic`): the one- and two-sphere references of `crates/sesmesh/tests/surface.rs` meshed at h = 1, 0.5, 0.25, 0.125 Å; areas/volumes against the testkit's quadrature references, largest \|F\| at a vertex, largest normal deviation |
| `scaling.csv` | `docs/02-design.md` §11.1 ("Measured scaling (Phase 5)") | transcribed by hand: wall time (s) of the total run for 4HHB and 1AON at 0.5 Å and of the 1AON mesh stage, 1–10 threads |

The validation, sweep and large-input data are read directly from
`docs/validation/{accuracy,sweep,large}.csv` by `paper/scripts/make_data.py`;
its outputs go to `paper/generated/` (tables, macros and pgfplots data).
