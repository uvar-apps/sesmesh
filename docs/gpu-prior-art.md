# GPU SES prior art, and what it means for Phase 11

Review, 2026-11-20, branch `phase-11-gpu-hip`. It was written before any more
GPU code, after T11.3 (CPU-gathered lists, device sampling) measured slower
than the CPU. It reviews published GPU SES methods (§1–§2), maps them onto
the sesmesh pipeline under two rules (§3–§4), and proposes a revised plan
(§5). The two rules:

* no f64 on the device;
* the default output stays byte-identical.

It supersedes the parts of `docs/05-gpu-hip.md` that assume device f64: §1
(design bullets), §5 (exact refinement on the device), §8 and §10
(T11.4–T11.7). Machine times below are from `docs/05-gpu-hip.md` §2: 16
Zen 5 cores (Ryzen AI Max+ 395), sesmesh 1.1.2.

## 1. Summary

* **No published GPU method produces an exact SES mesh.** Each takes one of
  two shortcuts:
  * an approximate distance field on a grid, meshed or ray-marched
    (Hermosilla et al. 2017, QuickSES 2019);
  * the analytic patches with no mesh, for ray casting (SESDF 2024, GPU
    contour-buildup 2011, Schäfer & Krone 2019).

  All of them compute in f32.
* **The grid methods trade accuracy for speed.** Hermosilla's field is off
  by up to √3·h by construction. Against Chimera at 256³ (3 967 atoms) the
  error was 0.23 Å mean and 1.43 Å maximum, biased outward. sesmesh places
  its vertices within 10⁻⁵ Å of the exact surface.
* **A comparable GPU does not beat sesmesh's CPU path.** Two measurements,
  both at 0.5 Å:

  | Run | Molecule (atoms) | Hardware | Time | Output |
  |---|---|---|---|---|
  | QuickSES (reported) | 1GRU (58 688) | GTX 1080 Ti (11.3 TFLOP/s f32, 484 GB/s) | 1.89 s | approximate, mesh not written |
  | sesmesh | 1AON (58 870) | 16 Zen 5 cores | 1.32 s | exact, OBJ formatted |

  The Radeon 8060S has the 1080 Ti's f32 rate (12.3 TFLOP/s measured) but
  less than half its bandwidth (201 GB/s measured). A QuickSES-style
  pipeline would therefore not be faster here.
* **The GPU clearly wins at the SAS arrangement.** SESDF computes the
  circles, intersections and segments in f32:

  | Molecule | SESDF, RTX 2080 | sesmesh f64 arrangement, 16 Zen 5 cores | Arrangement share of a sesmesh run |
  |---|---|---|---|
  | 1AON | 10.75 ms | 0.13 s | 4 % (0.25 Å) |
  | 3J3Q | 0.70 s | 4.0 s | 14 % (1 Å) |

  An f32 arrangement makes different decisions near degeneracies (buried
  or intersected circles, hidden vertices), so its meshes would differ. It
  is not a candidate for the exact path.
* **The prior art keeps everything on the device.** It holds the data
  there, gathers neighbours from a device cell grid, sizes allocations by
  count–scan–fill, and recomputes derived data instead of storing it. T11.3
  did the opposite:
  * the CPU gathered per-sub-block copies of candidates and blockers;
  * it staged 3.1 GB of them for 1AON at 0.25 Å and 25.9 GB for 3J3Q at
    1 Å, 40–56 % of it blocker lists;
  * the device then ran only the kernel, 14–20 % of the mesh stage.

  That shape cannot gain more than the kernel's share (1.11–1.17×), even
  with free transfers.
* **What is possible for sesmesh:**
  * **Exact output:** the device does classification, gathering and
    sampling in f32 with widened margins (§3.2); the f64 refinement stays
    on the CPU. Spike S1 (§5.2) measured it: every value the mesh reads is
    bit-identical to the CPU's on 980 runs, and the projected gain is
    1.23–1.43× end to end. Moving the refinement gather too (§3.3) adds
    at most 0.05×.
  * **Faster than that** needs f32 refinement on the device. Positions
    would differ in the last bits (§3.4). This is opt-in only, with a
    ceiling of about 2.5–3.7×.
  * **Earlier claim withdrawn.** The "exact ≈ 1.6×" of `docs/05-gpu-hip.md`
    assumed f64 refinement on the device.

## 2. Methods reviewed

### 2.1 Hermosilla, Krone, Guallar, Vázquez, Vinacua, Ropinski (2017)

*Interactive GPU-based generation of solvent-excluded surfaces*, The Visual
Computer 33:869–881. All steps are compute shaders in f32. Rendering is by
ray marching; no mesh is made.

1. **Atom grid.** Atoms are hashed into cells of size r_p + max vdW
   radius.
2. **Probe intersection.** Each grid point tests the 27 neighbouring cells
   and is classified as:
   * outside (value r_p);
   * inside, when d < r − r_g (value −r_g);
   * boundary.
3. **Distance refinement.** Each boundary point looks for the nearest
   outside grid point within r_p + r_g and takes d = r_p − |p_n − p_c|.
   Only 8³ bricks that hold boundary points are refined.
4. **Progressive mip levels,** up to 512³ (620 MB).

* **Accuracy.** The true distance lies in [d, d + √3·r_g], so the error
  is one grid diagonal. Small cavities can be missed.
* **Speed** (GTX 970):
  * 1K4R (545 k atoms): 15.9 ms at the 128³ base level, 1.65 s through all
    levels to 512³ (0.98 Å cells);
  * 1CWP (227 k atoms): 2.47 s to 0.58 Å;
  * 3EXG (83 k atoms): 1.24 s to 0.51 Å.
* **Cost.** The refinement neighbourhood grows as (r_p/h)³.

### 2.2 QuickSES (Martinez, Krone, Baaden 2019)

*QuickSES: a library for fast computation of solvent excluded surfaces*,
MolVA 2019. A CUDA implementation of §2.1 that outputs a mesh. Source:
github.com/nezix/QuickSES (MIT; read at 4040f6b).

* **Field.** As §2.1:
  * thrust sort of atom hashes;
  * `probeIntersection` over 27 cells;
  * boundary list by sort and count;
  * `distanceFieldRefine`, a brute-force (2⌈r_p/h⌉+1)³ stencil search for
    the nearest outside voxel.
* **Slices.** The grid is cut into sub-grids of up to 300³ that overlap
  by ⌈r_p/h⌉. Each slice is meshed separately. The meshes are merged "by
  offsetting the triangle indices", so vertices are not shared across slice
  seams.
* **Marching Cubes on the GPU.** Count, scan, compact, generate, with
  linear interpolation of the f32 field.
* **Welding.** Vertices are rounded to a tolerance, then thrust
  `sort` + `unique` + `lower_bound`. This merges by position, not by
  topology.
* **On the CPU:** a closest atom per vertex, Laplacian smoothing, OBJ
  output. The probe radius is fixed at 1.4 Å.
* **Reported speed** (GTX 1080 Ti, 0.5 Å, mean of 5 runs, disk writes
  excluded):

  | PDB | Atoms | Time | Peak memory |
  |---|---|---|---|
  | 1CRN | 327 | 333 ms | 10 MB |
  | 3EAM | 13 505 | 627 ms | 439 MB |
  | 1GRU | 58 688 | 1 888 ms | 805 MB |
  | 4V59 | 169 748 | 2 893 ms | 695 MB |

* **Authors' future work:** shared-memory stencils, several voxels per
  thread, GPU smoothing.
* **Related:** SiESTA-Surf, a Python wrapper of the same code.

### 2.3 SESDF (Plateau-Holleville, Maria, Mérillou, Montes; IEEE TVCG 2024/25)

*Efficient GPU computation of large protein solvent-excluded surface*.
CUDA. Source: github.com/PlathC/SESDF (read at a63faf1).

* **Representation.** The analytic SES as the patches of Quan & Stamm
  (2016): convex, toroidal and concave (Fig. 3–6 of the paper). They are
  rendered by ray casting and sphere tracing. There is no mesh ("producing
  such geometry … could be interesting", future work).
* **Pipeline,** all in f32:
  1. Atom cell grid built by hash and sort.
  2. Circle classification (buried, full, intersected), in one pass over
     the neighbours.
  3. SAS intersections, processed cooperatively by warps: N(a_i) is held
     in shared memory, with a bit mask of pending triplets.
  4. Two-pass intersection gathering, with atomic counts per circle.
  5. Segments by a warp-level angular sort.
  6. Neighbour search for the concave patches.
* **Memory.** Exact counts before allocation (count, then fill). Derived
  data (circle centre, radius, segment angles) is recomputed on demand.
  The neighbourhood is capped at 128 atoms, and at 16 intersections per
  circle.
* **Speed and memory** (RTX 2080, mean of 1 000 runs):

  | Molecule | Time | Memory |
  |---|---|---|
  | 1AON | 10.75 ms | 185 MB |
  | 3J3Q | 700 ms | 7.6 GB |

  The CUDA contour-buildup of Krone et al. 2011 (MegaMol) on the same GPU:
  * 1AON: 11.95 ms, 5.5 GB;
  * fails beyond 65 k atoms.

### 2.4 Others

These are cited by the papers above. None of them produce a mesh:

* **GPU contour-buildup** (Krone et al. 2011): analytic, ray cast,
  memory-bound (above).
* **Schäfer & Krone, MolVA 2019:** analytic, every atom triplet tested,
  ray cast; aimed at dynamic data.
* **Parulek & Viola:** an implicit SES that is ray cast directly, with
  reported numerical errors.
* **Bruckner:** a Gaussian surface (not the SES).
* **EDTSurf** (Xu & Zhang 2009): a CPU Euclidean distance transform. It is
  slower than QuickSES and coarsens the grid for large inputs.
* **Patent US10529444B1:** an SDF on a grid, mostly on the GPU.

## 3. Mapping onto sesmesh

### 3.1 Stages, shares, and what an f32-only device can take

Shares of mesh-stage thread time, from `docs/05-gpu-hip.md` §2.2:

| Stage | Share | Device in f32? | Output byte-identical? |
|---|---|---|---|
| Read, arrangement (f64) | 0.13 s of 3.18 s (1AON 0.25 Å); 4.0 + 1.1 s of 27.7 s (3J3Q 1 Å) | arrangement only as an f32 re-implementation (SESDF) | no (different decisions near degeneracies): stays on the CPU |
| Classification (sub-blocks) | 2.3–8.9 % | yes, with widened margins | yes (§3.2) |
| Sampling gather | 8.9–11.3 % | yes, with widened margins | yes (§3.2) |
| Kernel | 13.6–20.0 % | yes, done (K2) | yes, verified bit for bit |
| Refinement gather | 13.9–15.3 % | lists only (integers) | probably (§3.3) |
| Vertex refinement (f64 Newton/Illinois/bisection) | 40.8–58.7 % | f32 only | no (§3.4) |
| Edge scan, cells | 1.6–5.6 % | possible | yes; too small to matter |
| Assembly, measure, write | 0.37 + 0.18 s (1AON 0.25 Å) | — | stays on the CPU |

### 3.2 Exact classification and sampling gather on the device

The CPU makes these decisions in f64 with no slack (`docs/02-design.md`
§7.3–§7.4). The device cannot repeat them in f32 exactly. It can make the
same tests with margins widened by more than the f32 error, so that:

* its sub-block states are at least as conservative as the CPU's;
* its lists are supersets of the CPU's.

The output is unchanged for three reasons:

* **Only two things reach the mesh:**
  * the sign of each sample, which sets the MC cases;
  * the f32 value at both ends of each crossing edge, which gives the
    first Newton iterate.

  Refinement then uses the CPU's own `gather_local` lists.
* **Extra sampled sub-blocks change no sign.** A sub-block the CPU marks
  uniform has the uniform sign with margin ε_c = 2·10⁻³ Å, against
  ≈ 3·10⁻⁵ Å of kernel error. If the device samples it, the value differs
  from ±δ but the sign does not. No crossing edge lies in a uniform
  sub-block (Design §7.3), so no edge end changes.
* **Extra candidates or blockers cannot change a value at an edge end.**
  An extra one sits beyond the CPU's test radius G = r_p + δ + ρ (or
  R_k + G). It can affect a sample only when the sample's closest-point
  distance is within f32 error of the cutoff D = r_p + δ, that is, when
  |F̃| is within f32 error of δ. Crossing-edge ends have |F| ≤ h < δ = √3·h,
  so they are at least 0.73·h away from any such case.

  The kernel's minimum does not depend on candidate order, so the device
  may gather in any order.

This needs the exact f32 inputs the CPU computes (centre − frame, radii,
circles and vertices as f32). The CPU computes them once, uploads them once,
and the device keeps them resident with a cell grid of elements. No
per-sub-block copies are staged.

The test changes too. Device sample values equal the CPU's wherever
|F̃| < δ − margin. Outside that, values may differ while the output stays
the same; the byte-identical output gate decides.

**Ceiling.** Device time zero, CPU classification, gathering and kernel
removed (05 §2.3, second row): 1.32× (4HHB 0.25 Å), 1.36× (1AON 0.5 Å),
1.41× (1AON 0.25 Å), 1.23× (3J3Q 1 Å).

Three things lower it in practice:

* **The device is not free.** At 1AON 0.5 Å, K2 takes 60 ms where the 16
  cores spend ≈ 182 ms, and gathering adds to that.
* **The CPU still feeds the device:** batches, the refinement gather and
  refinement.
* **The APU shares power and memory bandwidth** between the CPU cores and
  the GPU. Measured in S1 (§5.2): a busy device slows the 16 cores by
  5–12 %, and busy cores slow the device by 1.5–1.7×.

### 3.3 Refinement gather on the device (exact, uncertain)

The CPU's `gather_local` decisions are also f64. A device superset of the
lists (as integer indices), read by the unchanged f64 `eval_local`, gives
the same minimum by the argument of §3.2.

One difference: `eval_local` also returns the closest point, the feature
and the nearest atom. With exact ties between candidates (symmetric
synthetic inputs), the winner depends on list order. The CPU's order comes
from f64 sort keys (class-S by distance), which f32 cannot reproduce. So
this part is byte-identical only if:

* ties are measured not to occur on the suites; or
* the device emits lists in an order the CPU's order can be derived from.

**Ceiling with it** (adding 13.9–15.3 %; S1 measured 7–10 %, §5.2):

| Input | Ceiling |
|---|---|
| 4HHB 0.25 Å | 1.50× |
| 1AON 0.5 Å | 1.61× |
| 1AON 0.25 Å | 1.65× |
| 3J3Q 1 Å | 1.42× |

**S1 outcome (§5.2).** The tie risk is gone if the CPU re-runs its own f64
tests and sorts over the device's lists: the lists then come out identical,
order included. But the cell scans this removes are only 17–27 % of
`gather_local`; the f64 tests, sorts and blocker filtering stay on the CPU.
The gain is 0.005–0.5 s, at most 0.05× end to end.

### 3.4 f32 refinement on the device (opt-in, not byte-identical)

Vertex refinement is 41–59 % of the mesh stage in f64. Only an f32 version
on the device removes it. This changes:

* **Connectivity: unchanged.** The MC cases come from the same f32 samples.
* **Positions: last bits differ.**
  * The f32 field error is ≈ 3·10⁻⁵ Å in the current centred frame; it is
    smaller with sub-block-local coordinates (to measure).
  * The vertex bound becomes TOL plus that error, ≈ 4·10⁻⁵ Å instead of
    10⁻⁵ Å.
  * That is still about four orders of magnitude below the grid methods
    and far below the MC discretisation error (−0.2 % area at 0.25 Å).
* **Edges that do not converge in f32** (non-monotone F, Newton stalls of
  the 1.1.1 fix) are flagged and refined on the CPU in f64.
* **Attributes:** the nearest atom and patch kind can differ at ties.

**Ceiling,** with device time zero and only the cell pass left
(05 §2.3, third row): 2.5–3.7×.

### 3.5 What is not adopted

* **Grid distance fields** (Hermosilla, QuickSES): approximate by about a
  grid diagonal, and on this class of GPU not faster than sesmesh's exact
  CPU path (§1).
* **Position welding and slice-wise meshes** (QuickSES): sesmesh already
  shares vertices by edge ownership, which welds by topology.
* **f32 arrangement** (SESDF): fast, but not byte-identical and not
  validated like the f64 arrangement (sweep, fuzzing). It could become a
  later opt-in for 3J3Q-sized inputs, after a robustness study.

## 4. Why T11.3 is slower than the CPU

* **Staged copies.** For every sub-block the CPU copies the samples, the
  candidates (as f32 values) and each candidate's blocker list (as indices
  into the sub-block's near list). Blocker lists are the largest part (S
  and C blockers: 1.25 GB of 3.1 GB for 1AON 0.25 Å, 14.4 GB of 25.9 GB for
  3J3Q 1 Å). Replacing candidate values by global indices would only save
  about 1.5× (measured).
* **The CPU keeps all its gathering.** With the device kernel at most
  three times faster than the 16 cores' kernel share, the saving is at most
  the 14–20 % kernel share. Staging costs about the same, so the result is
  parity or worse (measured 2–27 % slower).
* **Lesson from the prior art:** keep the inputs resident, and gather and
  classify where the kernel runs.

## 5. Revised Phase 11 plan

### 5.1 Spikes

* **Dropped:** every device f64 path:
  * the f64 self-test (removed);
  * T11.4 K5 exact f64;
  * T11.5 f64 gathers;
  * "exact refinement on the device when its f64 is fast", including
    MI250X and MI300.
* **Kept:** T11.0–T11.2 (committed); the chunk-length test; the batch
  pipeline shape (two buffers alternating, device on its own thread). The
  staged K2 kernels and `Sampler` were removed with T11.3′.

**Measure before building.** Two spikes, each a few days, on dumps from the
CPU build, as in T11.0:

| ID | Spike | Measures | Decides |
|---|---|---|---|
| S1 | Device classification + sampling gather (f32, widened margins) + K2 on a resident arrangement and element grid. 1AON 0.25/0.5 Å, 3J3Q 1 Å. Also the refinement-gather lists (§3.3) and their tie cases. | Values equal the CPU's for every sample with \|F̃\| < δ − margin. Device time per stage. List sizes. Device time with the 16 cores busy refining at the same time (shared power). | Whether an exact GPU mode is worth building: projected end-to-end gain from measured device times, against the ceilings of §3.2–§3.3. |
| S2 | f32 refinement kernel on dumped crossing edges and local lists, in sub-block coordinates. | Device time. Share of flagged edges. Maximum distance from the CPU vertices. \|F\| at the vertices (f64 on the CPU). Attribute differences. | Whether a fast mode is worth building, and its bound. |

### 5.2 S1 results (2026-11-21)

Code: `crates/sesmesh/src/field/gpu_s1.rs` (ignored test `s1`, feature
`hip`) and the `sesmesh_resident_*` kernels in
`crates/sesmesh-gpu/kernels/hip/sesmesh.hip`, driven by
`sesmesh_gpu::hip::Resident`. The host is the 16-core Zen 5 / Radeon 8060S
APU of `docs/05-gpu-hip.md` §2. The device:

* holds the arrangement resident (the f32 frame-local copies the CPU
  kernels use, neighbour and blocker CSRs, the CPU's three cell grids);
* per batch of bricks runs three kernels: count (cheap uniform tests and
  list sizes), fill (lists, then the Lipschitz test at the sub-block
  centre), sample (owner rule, SES/SAS/vdW evaluation);
* sizes the list buffer from the counts (exact, no budget).

The slack on every f32 test is 10⁻³ Å + 16 ulp of the largest coordinate.

**Correctness.** Compared with the CPU's `Engine::evaluate` brick by brick:

* the corpus: SES at 1, 0.5 and 0.25 Å on the 256 gate inputs, SAS and
  vdW at 1 and 0.5 Å on the 53 validation inputs (980 runs);
* plus 3J3Q at 1 Å (2.44 M atoms, 142.6 M samples evaluated by both).

Results:

* no sub-block that the device calls uniform is non-uniform on the CPU or
  of the other sign;
* no sample changes sign, and no crossing-edge end changes;
* every sample both sides evaluate is equal bit for bit, at the clamp too.

The device samples 0.0–0.2 % more sub-blocks than the CPU.

**Times** (best of 3, 16 threads; "busy": device and CPU pass run at the
same time):

| Input | Device | Device, CPU busy | CPU evaluate | CPU, device busy |
|---|---|---|---|---|
| 4HHB 0.25 Å | 0.014 s | 0.024 s | 0.068 s | 0.072 s |
| 1AON 0.5 Å | 0.082 s | 0.120 s | 0.371 s | 0.388 s |
| 1AON 0.25 Å | 0.223 s | 0.373 s | 0.985 s | 1.047 s |
| 3J3Q 1 Å | 1.400 s | 2.394 s | 5.401 s | 6.037 s |

* The device takes 21–26 % of the CPU's time for the same stage.
* Sampling is 78–87 % of the device time; count and fill are the rest.
* SAS and vdW are smaller: 1AON SAS 0.5 Å 0.006 s vs 0.037 s; vdW
  0.5/0.25 Å 0.005/0.019 s vs 0.023/0.099 s.
* Device memory at 1AON 0.25 Å: 24 MB resident, 194 MB of lists, 592 MB of
  values. At 3J3Q 1 Å: 1.03 GB resident, 1.2 GB of lists, 681 MB of values.
  T11.3′ batches would hold only a batch's lists and values.

**Projection.** End-to-end CPU time minus the CPU evaluate pass, plus the
slowdown of the refining cores while the device runs (perfect overlap
otherwise; the device time with the CPU busy is at most 17 % of the CPU
work left, so it can hide behind it):

| Input | CPU today | Projected | Gain | Ceiling (§3.2) |
|---|---|---|---|---|
| 4HHB 0.25 Å | 0.28 s | 0.213 s | 1.31× | 1.32× |
| 1AON 0.5 Å | 1.32 s | 0.96 s | 1.38× | 1.36× |
| 1AON 0.25 Å | 3.18 s | 2.22 s | 1.43× | 1.41× |
| 3J3Q 1 Å | 27.7 s | 22.6 s | 1.23× | 1.23× |

The measured evaluate pass is a little larger than the share estimates of
§3.1, hence projections at or slightly above the ceilings. Not included: the
first batch, which cannot overlap, and per-batch launch and scan overheads.

**Refinement-gather lists (§3.3).** The device also gathered the
non-contained spheres within reach (the `near` list). For every sub-block
that `extract.rs` gathers for (66–81 % of the sampled ones), the CPU ran
its exact f64 tests and sorts over the device lists (`gather_local_from`)
and compared the result with `gather_local`:

| Input | Sub-blocks | Device ids each | Lists differing | `gather_local` | From device lists |
|---|---|---|---|---|---|
| 4HHB 0.25 Å | 23 770 | 98 | 0 | 0.020 s | 0.015 s |
| 1AON 0.5 Å | 74 681 | 211 | 0 | 0.134 s | 0.105 s |
| 1AON 0.25 Å | 323 064 | 103 | 0 | 0.295 s | 0.220 s |
| 3J3Q 1 Å | 654 379 | 546 | 0 | 2.876 s | 2.383 s |

Results:

* The lists are identical, order included, so ties cannot differ.
* The saving is 17–27 % of `gather_local`, 0.5 s at most (3J3Q): 1.43× →
  1.48× at 1AON 0.25 Å, 1.23× → 1.25× at 3J3Q.
* The device list buffer grows by 31–43 % (1AON 0.25 Å: 194 → 278 MB;
  3J3Q 1 Å: 1.2 → 1.57 GB).

`gather_local` is 7–10 % of the run, below the 13.9–15.3 % thread-time
share of §3.1. Not worth building in this form (T11.4′ below).

**Decision asked.** Build T11.3′ (exact device sampling, ≈ 1.23–1.43×), and
run S2 (the f32 refinement that the remaining time is mostly spent in).

### 5.3 Tasks

**Then, depending on those decisions:**

| ID | Task | Gate |
|---|---|---|
| T11.3′ | Exact device sampling. Arrangement uploaded once (f32 copies the CPU computes). Device element grid, classification, gathering, K2, per batch of bricks. CPU refinement overlapped. Allocation from counts and the device's reported memory, no fixed budgets. | Byte-identical output on the existing gate (1 536 SES, 424 SAS/vdW, 210 option-set runs, test suite); identical under thread counts and batch sizes; cancel < 100 ms; times for 4HHB, 1AON and 3J3Q on 16 cores vs device. |
| T11.4′ | Refinement-gather lists from the device. S1: exact, but ≤ 0.05× for 31–43 % more list memory; not planned. | Byte-identical as in T11.3′. |
| T11.5′ | Opt-in fast mode: f32 refinement on the device, f64 CPU fallback for flagged edges. | Same connectivity; vertex bound from S2; the Phase 9 accuracy study repeated; area and volume against the exact output; repeat runs identical. |
| T11.6′ | Benchmarks, docs, release (old T11.8–T11.9). | Local validation list plus GPU tests. |

**T11.3′ status (2026-11-21): done.** `sesmesh::field::gpu` with
`sesmesh_gpu::hip::{Resident, Batch}`:

* the arrangement is uploaded once per surface (`Engine::resident`);
* a device thread classifies, gathers and samples batches of one mesh
  chunk (≤ 64 bricks) per pool thread into a ring of three buffers;
* the pool's threads take chunks in order and mesh each as soon as its
  batch is ready. A first version alternated two buffers with a barrier
  after every batch; waiting for each batch's slowest chunk cost more than
  the overlap gained (1AON 0.25 Å: 1.77 s mesh stage, against 1.71 s for a
  single batch with no overlap and 1.46 s without the barrier);
* the device path is used when the `f32` kernels are (`f32_ok`) and bricks
  split into 2×2×2 sub-blocks (even brick size ≥ 4), else the CPU samples.

Gate, on the 16-core Zen 5 / Radeon 8060S host:

* the 2 170 runs of §5.2's corpus and option sets give the CPU's bytes;
  the CPU's output equals 1.1.2's on the 1 960 corpus runs;
* the device tests: the CPU's sign everywhere and its value bit for bit
  except where the CPU writes ±δ; meshes equal for batches of 1, 2 and all
  chunks; cancellation;
* `cancellation_is_prompt` also runs on the device (< 100 ms).

Times, best of 5 (device open, 45 ms per process, excluded):

| Input | CPU total | HIP total | Gain | Mesh stage CPU → HIP | Gain |
|---|---|---|---|---|---|
| 4HHB 0.25 Å | 240 ms | 196 ms | 1.23× | 164 → 115 ms | 1.43× |
| 1AON 0.5 Å | 1 393 ms | 1 030 ms | 1.35× | 1 050 → 668 ms | 1.57× |
| 1AON 0.25 Å | 3 407 ms | 2 355 ms | 1.45× | 2 516 → 1 458 ms | 1.73× |
| 3J3Q 1 Å | 29.0 s | 23.6 s | 1.23× | 21.3 → 15.9 s | 1.34× |

These match the S1 projection (§5.2: 1.31×, 1.38×, 1.43×, 1.23×). The rest
of the time is f64 vertex refinement and the CPU stages around the mesh
stage (arrangement, assembly, measure, write), which S2 and T11.5′
address.

Not in Phase 11, on the CPU:

* **SIMD f64 refinement** (lanes = edges, AVX-512/NEON). It is
  byte-identical, and it attacks the same 41–59 % without a GPU.
* **An f32 arrangement** (§3.5).

## 6. Sources

* P. Hermosilla et al., "Interactive GPU-based generation of
  solvent-excluded surfaces", The Visual Computer 33 (2017) 869–881.
* J. Martinez, M. Krone, M. Baaden, "QuickSES: a library for fast
  computation of solvent excluded surfaces", MolVA 2019,
  doi:10.2312/molva.20191095; github.com/nezix/QuickSES.
* C. Plateau-Holleville, M. Maria, S. Mérillou, M. Montes, "Efficient GPU
  computation of large protein solvent-excluded surface", IEEE TVCG (2025),
  doi:10.1109/TVCG.2024.3380100; github.com/PlathC/SESDF.
* M. Schäfer, M. Krone, "A massively parallel CUDA algorithm to compute and
  visualize the solvent excluded surface for dynamic molecular data", MolVA
  2019, doi:10.2312/molva.20191094.
* M. Krone et al., "Parallel contour-buildup algorithm for the molecular
  surface", IEEE BioVis 2011.
* C. Quan, B. Stamm, "Mathematical analysis and calculation of molecular
  surfaces", J. Comput. Phys. 322 (2016) 760–782.
* D. Xu, Y. Zhang, "Generating triangulated macromolecular surfaces by
  Euclidean distance transform", PLoS ONE 4 (2009) e8140.
