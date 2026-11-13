# sesmesh validation report (Phase 9)

Date: 2026-11-13. Machine: Apple Silicon, 10 cores (4 performance + 6
efficiency), macOS, release build, all cores. Inputs: default selection
(first model, no hydrogens or waters), Bondi radii, probe 1.4 Å, SES with
cavities kept, default brick size (10) and SIMD level (NEON). The machine
carried a heavy background load during the runs (load average 15–27 from
other applications), so wall times are pessimistic. Raw data:
[`validation/`](validation/README.md).

## 1. Method and its limits

No external surface tool (MSMS, NanoShaper, EDTSurf, sesa, ChimeraX) was
available, so — by decision — the references are internal:

* **Analytic** single- and two-sphere surfaces (Phase 4,
  `crates/sesmesh/tests/surface.rs`).
* **h → 0 extrapolation** for real structures: sesmesh's vertices lie on
  the exact SES (≤ 10⁻⁴·h), so the area and volume of the inscribed mesh
  converge as A(h) = A₀ + c·h². A₀ and V₀ are least-squares fits over
  h ∈ {0.5, 0.35, 0.25} Å; errors are reported relative to them.

This measures the **discretization** error of the mesh. It does not detect
differences in surface *definition* (radii handling, probe-region details)
from other tools; those need the cross-tool comparison of issue #5, moved
to after 1.0.

## 2. Accuracy (T9.3)

Validation set: 53 entries from 154 to 237 685 atoms (peptides, proteins,
DNA, nucleosomes, antibodies, spike trimer, GroEL–GroES, ribosomal subunits,
80S ribosome); `[sets.validation]` in `tests/data/sets.toml`.

| Spacing | Area error: mean | worst | Volume error: mean | worst |
|---|---|---|---|---|
| 0.5 Å | −0.82 % | −0.93 % | −0.08 % | −0.34 % |
| 0.25 Å | −0.20 % | −0.23 % | −0.02 % | −0.08 % |

* Convergence is clean O(h²): halving h divides the area error by 4.1.
* Area is always underestimated (inscribed mesh); the volume error is an
  order of magnitude smaller for proteins because convex and concave patches
  bias it in opposite directions. Small, highly curved molecules (peptides,
  the 1BNA DNA dodecamer) have the largest volume errors (−0.28…−0.34 % at
  0.5 Å).
* **NFR-ACC protein bounds** (area 1.0 % / volume 0.5 % at 0.25 Å; 3 % / 1 %
  at 0.5 Å) are met with a wide margin relative to the discretization
  reference.
* The analytic bound for atom-sized single spheres and pairs (≤ 0.5 % at
  0.25 Å) is still missed for volume (0.8–1.35 %, Phase 4). Proteins do not
  show this bias. **Recommendation for issue #4:** restate the analytic bound
  as an O(h²) convergence requirement (area ≤ 0.7 %, volume ≤ 1.5 % for
  1.2–1.8 Å spheres at 0.25 Å) and keep the protein bounds; no method
  change needed. **Decided 2026-11-13 (#4):** adopted (Spec §5.3), encoded
  in `two_spheres_match_reference`.

## 3. Density calibration (T9.5)

vertices · h² / area over the 53 entries:

| Spacing | Mean | SD | Range |
|---|---|---|---|
| 1.0 Å | 1.5006 | 0.0070 | 1.487–1.521 |
| 0.7 Å | 1.4985 | 0.0043 | 1.488–1.515 |
| 0.5 Å | 1.5000 | 0.0033 | 1.492–1.508 |
| 0.35 Å | 1.4997 | 0.0020 | 1.495–1.506 |
| 0.25 Å | 1.4999 | 0.0021 | 1.495–1.504 |

The theoretical constant 1.5 (Design §8.6) holds to 0.2 % (h ≤ 0.5 Å:
1.4999 ± 0.0025); `VERTICES_PER_AREA_H2` stays 1.5 and the density test was
tightened from ±25 % to ±5 %. Presets were not changed.

## 4. Robustness sweep (T9.4)

200 random current PDB entries (seed 2026, from 263 196 entries;
`[sets.sweep]`), mmCIF, 0.5 Å — reduced from the planned 1 000 by decision.

* **200 ok, 0 input errors, 0 failures.** Every mesh passed the built-in
  validation (closed, consistently oriented, edge-manifold, no degenerate or
  duplicate triangles).
* Sizes 197–129 900 atoms (median 3 727); 9 714 cavities in total; up to
  1 098 components in one entry; 108 s of compute in total, 11.8 s for the
  largest entry.
* *After 1.0:* the component counts of `accuracy.csv` and `sweep.csv` (made
  with 1.0.0) include sub-grid fragments, which are now dropped by default
  (Design §9.1): at 0.25 Å, 1 358 in 36 of the 53 validation entries (all but
  one non-cavity component per entry, 1AOI three); the sweep set meshed at
  0.5 Å has 1 026 in 119 of 200 entries (at most 2.7 Å² each; the smallest
  true solid component has 36 Å²). The accuracy figures above are
  unaffected.

## 5. Large inputs and memory (T9.7, NFR-MEM)

| Input | Spacing | Atoms | Triangles | Wall time | Peak RSS | Budget (150 B/tri + 200 B/atom) |
|---|---|---|---|---|---|---|
| 1AON | 0.25 Å | 58 870 | 17.5 M | 9.3 s | 2.28 GB | 2.63 GB ✓ |
| 4V6X | 0.5 Å | 237 685 | 18.0 M | 13.3 s | 2.65 GB | 2.75 GB ✓ |
| 1K4R assembly 1 | 1.0 Å | 545 040 | 9.9 M | 14.9 s | 2.25 GB | 1.59 GB ✗ |
| 3J3Q | 1.0 Å | 2 440 800 | 48.7 M | 70.1 s | 5.31 GB | 7.79 GB ✓ (spec: < 16 GB) |

* Per triangle the peak is 110–150 B, within the budget.
* Per atom the SAS arrangement and structure data take ≈ 600–800 B
  (≈ 20 neighbour pairs, 2 exposed circles with blockers and 1.5 triple
  points per atom), above the 200 B/atom budget. It only shows when there
  are few triangles per atom (the capsid at 1 Å: 18 per atom). Either the
  budget is revised (≈ 1 KB/atom) or the arrangement storage is compacted
  (e.g. f32 circle data, dropping the neighbour lists after gathering).
* The memory guard's estimate now uses 150 B/triangle + 800 B/sphere.

**Update (v1.0 development, issue #21).** The arrangement was compacted and
its build, the brick classification and the per-brick meshes were moved to
flat per-chunk arrays (Design §11.3 "As built"); meshes are byte-identical.
Re-measured in one session with the 0.9.1 code and the new code (same
machine and load; macOS peak RSS varies between sessions with memory
pressure, so only same-session pairs compare):

| Input | Spacing | Triangles | Peak RSS 0.9.1 | Peak RSS v1-dev | Old budget (200 B/atom) | Revised budget (800 B/atom) |
|---|---|---|---|---|---|---|
| 1AON | 0.25 Å | 17.5 M | 2.38 GB | 2.14 GB (−10 %) | 2.63 GB ✓ | 2.67 GB ✓ |
| 4V6X | 0.5 Å | 18.0 M | 2.75 GB | 2.40 GB (−13 %) | 2.75 GB ✓ | 2.89 GB ✓ |
| 1K4R assembly 1 | 1.0 Å | 9.9 M | 2.42 GB | 1.74 GB (−28 %) | 1.59 GB ✗ (+9 %) | 1.92 GB ✓ |
| 3J3Q | 1.0 Å | 48.7 M | 7.89 GB | 7.16 GB (−9 %) | 7.79 GB ✓ | 9.25 GB ✓ |

* Field data per atom (heap, counted with an allocating probe on the
  capsid): ≈ 1 150 → 560 B (neighbour rows 296 → 95 B, circles 462 → 197 B,
  blockers 162 → 132 B, arcs 81 → 0 B, triple points 67 → 58 B); the input
  structure and spheres add ≈ 150 B. The 200 B/atom budget is not reachable
  with exact f64 geometry (the circles alone take ≈ 200 B).
* **Decision (2026-11-13, #21):** NFR-MEM revised to 150 B per triangle +
  800 B per atom (Spec §5.2), the measured figure and the memory guard's
  estimate; every large input is within it.

## 6. Performance (NFR-PERF)

| Structure | Spacing | Target | Measured (10 cores, loaded) | Ratio |
|---|---|---|---|---|
| 1CRN | 0.25 Å | < 60 ms | 41–61 ms | ≈ 1 |
| 4HHB | 0.5 Å | < 100 ms | 0.19–0.22 s | ≈ 2× |
| 4HHB | 0.25 Å | < 350 ms | 0.45–0.54 s | 1.3–1.5× |
| 1AON | 0.5 Å | < 1.5 s | 2.7–3.1 s | 1.8–2× |
| 1JJ2 | 0.5 Å | < 2.5 s | 4.4 s | 1.8× |
| 4V6X | 0.5 Å | < 6 s | 13.3–14.0 s | 2.2× |
| 3J3Q | 1.0 Å | < 60 s | 70 s | 1.2× |

The provisional targets are missed by up to ≈ 2× on this machine under
load; the reference machines of the spec (8 performance cores; x86 AVX2)
were not available. Levers, in order: the SIMD sampling kernel's validity
tests (issue #13, ≈ 38 % of the time), x86 SIMD widths (issue #8), quieter
and homogeneous hardware for the measurement itself (issue #10).

**Decision (2026-11-13, #20):** for 1.0 the targets are these measurements,
rounded up (Spec §5.1: 1CRN < 60 ms, 4HHB < 250 / 600 ms, 1AON < 3.5 s,
1JJ2 < 5 s, 4V6X < 15 s, 3J3Q < 75 s); the provisional targets remain goals
for the work after 1.0.

## 7. Not done in this phase

* Cross-tool references (T9.2) and the benchmark comparison with other tools
  (T9.6): no tools installed — issues #5, #6.
* The 24 CPU-hour fuzzing campaign per parser (T9.8): moved to after 1.0
  (#22). For 1.0 every target ran 10 minutes (pdb, mmcif, cif_lexer, xyz,
  pqr, detect, assembly; ≈ 18 M inputs from the fixture corpus) without a
  crash, timeout or out-of-memory report.
* Brick size, culling and refinement tuning (T9.5) was done in Phase 6
  (Design §7.5, §7.7); not repeated.

## 8. Reproduce

```bash
cargo xtask fetch --set validation --set sweep --set bench
```

```bash
cargo run --release -p xtask -- validate
```

```bash
cargo run --release -p xtask -- sweep
```

Large inputs: `sesmesh <input> [--assembly 1] -s <h> -o /dev/null --stats
/dev/stdout` under `/usr/bin/time -l` (peak RSS).
