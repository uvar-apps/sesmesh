# GPU back end (HIP): analysis, design and plan

Draft, 2026-11-20, branch `phase-11-gpu-hip`. Issue #24 (HIP), with #25
(CUDA) and #26 (Metal) to reuse the shared parts; tracking issue #19. This
document records the measurements behind the design (§2), the design (§3–§9)
and the plan (§10, Phase 11 of `docs/03-implementation-plan.md`).
The decisions taken are listed in §11.

> **Superseded in part (2026-11-21).** Device f64 is dropped: no f64 runs
> on the GPU. The parts that assume it (§1 design bullets, §5 exact
> refinement on the device, §8, §10 T11.3–T11.9) are replaced by the
> prior-art review and revised plan in `docs/gpu-prior-art.md` (§5), which
> Phase 11 now follows: spike S1, then T11.3′ (exact device sampling), S2,
> T11.5′ (opt-in f32 refinement), T11.6′ (benchmarks, docs, release).

## 1. Summary

* **The mesh stage is the target, and refinement is its largest part.** On
  16 Zen 5 cores the mesh stage is 74 % of a run (1AON 0.25 Å, 3J3Q 1 Å).
  Within it, vertex refinement and its candidate gathering take 55–74 % of
  the thread time. The sampling kernel takes only 14–20 %. Offloading the
  kernel alone gives at most 1.11–1.17× end to end, against the 1.7–2.5×
  that #24 assumed from older profiles.
* **The device can reproduce the CPU bit for bit.**
  * On the Radeon 8060S (gfx1151), f32 and f64 `+ − × ÷ √` give the same
    bits as the CPU on 4 M random inputs each.
  * A direct HIP port of `ses_kernel`, built with `-ffp-contract=off`,
    returned the CPU's values exactly for all 24.5 M real samples dumped
    from 4HHB (0.25 Å) and 1AON (0.5 Å).
  * Every lane computes the minimum over the valid candidates, a value that
    does not depend on the order or on the wave width. So a GPU back end can
    write byte-identical output, keeping the 1.x output policy, the
    determinism tests and the validation results without re-validation.
* **f32 on the device is fast, f64 is not (on this GPU).**
  * The ported kernel runs ≈ 3× faster than 16 Zen 5 cores (≈ 43–47× one
    core).
  * The same kernel computed in f64 is 12× slower, which is ≈ 4× slower than
    the 16 cores. Exact (f64) refinement on gfx1151 therefore gains little
    over the CPU.
  * f64 is fast on the data-centre parts (MI250X, MI300).
* **Design.** One device pipeline for the whole mesh stage
  (classification, gathering, sampling, edge finding, refinement), with two
  refinement precisions:
  * *exact* (the default, decided 2026-11-20): f64, byte-identical to the
    CPU back end; refinement runs on the device when its f64 is fast,
    otherwise on the CPU;
  * *fast*: opt-in f32 refinement with lattice-anchored coordinates and an
    f64 fallback for flagged edges; same mesh connectivity, positions
    within the refinement bound.

  Expected end-to-end speed-ups on this APU against its own 16 cores (§8):
  * exact: about 1.4–1.8×;
  * fast: about 2.5–3×;
  * the rest of the run (arrangement, measure, write) then dominates.

## 2. Measurements

### 2.1 Machines and method

* **Apple M4** (10 cores; 4 P + 6 E) and **AMD Ryzen AI Max+ 395** (16 Zen 5
  cores, SMT off, Radeon 8060S iGPU, 128 GB LPDDR5X; one of the two
  comparison machines of `docs/comparison.md`). The Linux machine was idle:
  load ≈ 0, GPU activity 0 %, no GPU processes.
* **Profiling.** sesmesh 1.1.2 with temporary counters added
  (`tools/hip-spike/profile-and-dump.patch`, not merged). They add up the
  nanoseconds spent in each part of the mesh stage, over all threads:
  * sub-block classification (`uniform_sign` in `Engine::evaluate`);
  * sampling gather (`gather_ses`);
  * kernel (`ses_kernel`);
  * refinement gather (`gather_local`);
  * vertex refinement (`vertex`, including attributes);
  * the edge scan and the cell (MC) pass.
* **Inputs.** CLI runs with `-o /dev/null` (OBJ formatting still runs),
  `--timings`.

### 2.2 Where the time goes

Mesh stage, share of thread time (16 threads on Zen 5 unless noted):

| Input | Mesh / total wall | Classify | Sampling gather | Kernel | Refine gather | Vertex refinement | Edge scan + cells |
|---|---|---|---|---|---|---|---|
| 4HHB 0.25 Å | 0.18 / 0.28 s | 8.9 % | 10.3 % | 19.2 % | 13.9 % | 42.0 % | 5.6 % |
| 1AON 0.5 Å | 0.98 / 1.32 s | 4.7 % | 11.3 % | 19.7 % | 15.3 % | 46.2 % | 2.8 % |
| 1AON 0.25 Å | 2.34 / 3.18 s | 8.7 % | 10.6 % | 20.0 % | 14.4 % | 40.8 % | 5.4 % |
| 3J3Q 1 Å (2.44 M atoms) | 20.6 / 27.7 s | 2.3 % | 8.9 % | 13.6 % | 15.0 % | 58.7 % | 1.6 % |
| 1AON 0.25 Å, 1 thread, Zen 5 | 31.0 / 36.7 s | 8.7 % | 10.7 % | 20.1 % | 14.4 % | 40.9 % | 3.9 % |
| 1AON 0.25 Å, 1 thread, M4 | 28.1 / 32.9 s | 7.4 % | 9.9 % | 32.7 % | 14.2 % | 31.7 % | 4.1 % |

* **Kernel speed differs by machine.** The AVX-512 kernel is relatively
  faster on Zen 5 than NEON is on the M4, while the scalar f64 refinement is
  slower (12.7 s against 8.9 s for 1AON 0.25 Å, one thread).
* **Outside the mesh stage** (16 threads, Zen 5):
  * 1AON 0.25 Å: measure 0.37 s, write 0.18 s, geometry 0.13 s;
  * 3J3Q 1 Å: geometry (arrangement) 4.0 s, read 1.1 s, measure 1.1 s.
* **Candidate lists per sub-block,** on average. Sizes set the device data
  layout:

| Input | Samples evaluated | Inside | Near | S (blockers each) | C (blockers each) | V |
|---|---|---|---|---|---|---|
| 4HHB 0.25 Å | 178 | 10 | 29 | 20 (17) | 31 (11) | 8 |
| 1AON 0.5 Å | 172 | 17 | 49 | 34 (21) | 63 (12) | 18 |
| 3J3Q 1 Å | 170 | 39 | 111 | 77 (24) | 173 (13) | 66 |

About 7 KB of f32 data per sub-block at 0.25–0.5 Å and ≈ 19 KB at 1 Å.

### 2.3 Amdahl bounds

These bounds assume the offloaded work takes no time and the rest of the run
is unchanged (16 Zen 5 threads):

| Offloaded | 4HHB 0.25 Å | 1AON 0.5 Å | 1AON 0.25 Å | 3J3Q 1 Å |
|---|---|---|---|---|
| Kernel | 1.14× | 1.17× | 1.17× | 1.11× |
| Classification + sampling gather + kernel | 1.32× | 1.36× | 1.41× | 1.23× |
| Whole mesh stage except the cell pass | 2.5× | 3.6× | 3.3× | 3.7× |

Only the last row is worth a back end: refinement and both gathers must move
too.

### 2.4 Device and spike

* **Device and software.**
  * Radeon 8060S, gfx1151 (RDNA 3.5): 40 CUs, reported by HIP as 20
    multiprocessors in WGP mode.
  * Wave32 by default, wave64 available.
  * Unified memory: a 2 GB carve-out plus ≈ 122 GB GTT.
  * ROCm 7.16 (HIP 7.16, AMD clang 24); Rust 1.100.
* **Code.** `tools/hip-spike/` holds:
  * hand-written `extern "C"` bindings to `libamdhip64` (about 25
    functions);
  * code objects built by `hipcc --genco -O3 -ffp-contract=off
    -fno-fast-math`, loaded with `hipModuleLoadData`.

| Measurement | Result |
|---|---|
| f32 `+ − × ÷ √`, distance expression, `min` vs CPU (4 M inputs, incl. subnormals) | 0 mismatches |
| f64 `+ − × ÷ √`, distance expression, f64→f32 vs CPU (4 M inputs) | 0 mismatches |
| f32 mul+add / fma | 12.3 / 11.4 TFLOP/s |
| f64 mul+add / fma | 0.23 / 0.46 TFLOP/s (1/25 of f32) |
| √ f32 / f64 | 431 / 17.3 Gop/s |
| ÷ f32 / f64 | 518 / 18.9 Gop/s |
| Launch | 1.7 µs queued, 12 µs launch + synchronize |
| Streaming (read + write) from `hipMalloc`, `hipHostMalloc`, managed memory | 201 GB/s for all three |
| CPU writing pinned host memory | 55 GB/s (one thread) |

Real kernel: `ses.hip` is a line-by-line port of `ses_kernel`, with one
workgroup per sub-block and one lane per sample. It ran on the sub-blocks
dumped by the patched CPU build, and its output was compared with the CPU
values bit for bit:

| Input | Samples | GPU f32, wave32 | GPU f32, wave64 | GPU f64 compute, wave64 | CPU kernel, 16 Zen 5 threads (wall share) | CPU, 1 thread |
|---|---|---|---|---|---|---|
| 4HHB 0.25 Å | 5.2 M | 11.7 ms | 9.2 ms (1.7 ns/sample) | 114 ms | ≈ 31 ms | 429 ms |
| 1AON 0.5 Å | 19.3 M | 80.8 ms | 60.2 ms (3.1 ns/sample) | 718 ms | ≈ 182 ms | 2 585 ms |

* The f32 results show **0 mismatches**.
* The f64 variant differs from the f32 values by at most 1.2·10⁻⁶ Å. That
  is the f32 arithmetic error, about 25× below the input-rounding error of
  Design §7.5.
* Wave64 is ≈ 1.3× faster than wave32 on RDNA (dual-issue VALU).
* The port has not been tuned: no LDS staging and no sample reordering.

### 2.5 What follows

1. **A byte-identical GPU back end is possible.** The device must receive
   the same f32/f64 inputs and run the same IEEE operations in the same
   order (§3).
2. **On consumer RDNA the device's f64 is slower than the CPU's.** So exact
   refinement there stays on the CPU, or is shared with the device (§5).
3. **The big gain needs f32 refinement.** That gain is the *fast* mode (§6).
4. **Transfers are not a concern on an APU.** Device-visible host memory
   streams at the same rate as device memory, so batches need no copies.
   Discrete GPUs need staged copies (§7).

## 3. Exactness by replication

The CPU result of each brick is a deterministic function of the arrangement,
the lattice and the options. The device reproduces it when:

* **Same inputs.** Device code computes its inputs from the same `f64`
  arrangement arrays and lattice parameters, with the same expressions as
  the host code: `Lattice::local`, sub-block centres, `(c − frame) as f32`.
  Constants such as `√3` come from the host.
* **Same operations.** Each operation is written in the same order as in
  Rust, which evaluates `a + b + c` left to right. HIP is compiled with
  `-ffp-contract=off` and without fast math. The HIP defaults stay in force:
  correctly rounded f32 `÷` and `√` (`-fhip-fp32-correctly-rounded-divide-sqrt`)
  and denormals kept. Explicit `fma` is never used in exact kernels. The
  spike checks these properties (§2.4); a test in the GPU crate keeps
  checking them.
* **Same decisions.** Candidate selection compares f64 distances
  (`d < R + ρ`, `|d − R| ≤ reach`). The device evaluates those comparisons
  in f64 too, never in f32 with a margin: an extra "inside" sphere or
  blocker can flip a rounding-level decision.
* **Order only where it matters.**
  * The sampling kernel's value is the minimum over valid candidates. It
    does not depend on candidate order, and wave votes only skip work.
  * `eval_local` keeps the *first* candidate on exact ties of the distance,
    and its feature and closest point drive the Newton step. Device
    refinement therefore walks S, C and V in the same nearest-first
    `(distance, index)` order and the same class order as
    `SesField::gather_local`.
  * The order of the `inside` and `near` lists does not matter; `near` is
    sorted.

"Byte-identical" means identical to the CPU back end of the same binary. The
arrangement stays on the host, which keeps the known last-bit differences
between arm64 and x86-64 builds (they come from the arrangement) out of the
device.

## 4. Pipeline

One device pipeline per batch of bricks. Kernels map one-to-one onto
today's per-brick functions, so each can be tested against its CPU
counterpart:

| Kernel | Replaces | Work unit | Precision |
|---|---|---|---|
| K1 classify | `lattice::uniform_sign` for the 8 sub-blocks of each brick | 1 lane per sub-block | f64 (global `ses_eval` distance only) |
| K2 gather + sample | `gather_ses` + `ses_kernel` (`gather_spheres` + `sphere_kernel` for SAS/vdW) | 1 workgroup per non-uniform sub-block; lanes = owned samples | f64 decisions, f32 kernel |
| K3 edges | sign-change scan of `mesh_brick` | 1 lane per sample, 3 owned edges | — |
| K4 refine gather | `SesField::gather_local` | 1 workgroup per sub-block with crossing edges | f64 |
| K5 refine | `vertex` (Newton/Illinois, bisection, `gradient`) | 1 lane per edge, edges grouped by sub-block | f64 (exact) or f32 (fast, §6) |

The CPU keeps:

* the arrangement (f64, ill-conditioned near degenerate triples);
* the MC cell pass and triangle codes;
* vertex attributes (normal, patch kind, atom) from K5's last evaluation;
* assembly and post-processing.

K5 returns per edge `t`, the last evaluation (value, feature, closest point)
and its evaluation count, so `attributes()` and `SurfaceStats` stay
unchanged.

* **Ownership.** Sample ownership (first non-uniform sub-block), seam
  sharing (P7), edge ownership (Design §8.2) and sub-block indexing are
  unchanged. A brick's samples are written to its own `(b+1)³` value array
  as today.
* **Data on the device.**
  * Resident for the whole run, copied once or mapped on an APU: the
    arrangement arrays (centres, expanded and vdW radii, status, sphere cell
    grid, `neighbors` CSR, circles and `circle_blockers` CSR, circle and
    vertex cell grids).
  * Per batch: brick keys, sub-block states, gathered candidates (CSR per
    sub-block), values, edge records and refinement results.
* **Workgroup-local candidates.**
  * Candidate loops are uniform across a workgroup, so loads of candidate
    data are broadcasts.
  * K2 and K4 build their lists into workgroup memory (LDS, 64 KB) when they
    fit, else into a global scratch region. At 1 Å the average is ≈ 19 KB;
    the largest lists still need measuring.
  * Sorting (≤ a few hundred keys) is a bitonic sort in LDS on
    `(distance, index)`, the same total order as the CPU's
    `sort_unstable_by`.
  * Blocker filtering keeps CSR row order and tests membership by binary
    search in the sorted `near` list, as `gather_local` does. The CPU
    sampling gather uses a global `local_of` map instead; the device cannot
    share one between concurrent workgroups.
* **Port in two steps.**
  1. The CPU produces K2's and K4's lists (today's code, packed into batch
     arenas). The device runs the kernel and refinement. This gives early,
     exact results with little device code.
  2. The device takes over gathering and classification. Device f64 for
     these is cheap: about 10³ f64 square roots per sub-block, ≈ 25 ms for
     1AON 0.25 Å at 17 Gop/s.

## 5. Scheduling, hybrid execution, cancellation

* **Batches.** Bricks are processed in index-ordered batches sized to a
  device budget and to ≲ 20–50 ms of device time. Two batches are in flight
  on two streams, so the CPU work of batch *n* (cell pass, attributes)
  overlaps the device work of batch *n + 1*. Output is assembled in brick
  order as today, so it does not depend on batch size or completion order.
* **Hybrid (exact mode only).**
  * CPU and device results for a brick are identical, so CPU threads may
    take bricks or edges from the same queue while the device works, with
    dynamic, timing-dependent splits, and output stays byte-identical.
  * On gfx1151 the 16 cores are as fast as the device at f64 refinement,
    so edges are shared.
  * Fast mode gives device-specific vertex positions, so every brick goes to
    the device (or the split is static), keeping output deterministic.
* **Cancellation and progress.** The cancel flag is checked between batches
  (Spec: return within 100 ms). Progress counts finished bricks, as the CPU
  path does. The library never prints.
* **SAS/vdW and > 2 000 Å extents.**
  * SAS and vdW use K2 with the sphere kernel and skip K4/K5 candidate
    classes.
  * Inputs that need the CPU's f64 sampling path (`f32_ok` false) stay on
    the CPU in exact mode. Fast mode's anchored coordinates remove that
    limit on the device.

## 6. Fast mode: f32 refinement

This is the plan of #24, *Precision*, made opt-in. Sampling stays the exact
f32 kernel, so signs, and therefore triangles and connectivity, equal the
CPU's. Only vertex positions and attributes may differ.

* **Lattice-anchored coordinates.**
  * Each feature point (sphere centre, circle centre, SAS vertex, circle
    axis and basis vectors as they are) is stored as an integer lattice
    index `n` plus an f32 residual `r = c − h·n` (`|r| ≤ h√3/2`),
    computed in f64 on the host.
  * A point `x = o + h·i + h·t·e` on an edge then gives
    `x − c = h·(i − n) − r + h·t·e`, with the integer difference exact.
  * Expected error ≈ 10⁻⁶ Å (§2.4: 1.2·10⁻⁶ Å arithmetic error at 0.5 Å),
    against the refinement tolerance of 10⁻⁵ Å.
* **The iteration** is the same safeguarded Newton/Illinois with bisection
  fallback, in f32. Positions are returned as `t` (f32) on the edge and
  formed in f64 on the host, as today.
* **Flagged edges** are redone with the exact f64 evaluation, on the device
  (K5 exact) or on the CPU. An edge is flagged when:
  * it did not converge;
  * an end value lies within a few f32 steps of 0;
  * a validity test (`|q − c|² ≥ (R − tol)²`) or the best-versus-runner-up
    comparison came within a few f32 steps of its decision.
* **Validation needs no CPU prototype.** The device runs both precisions on
  every edge, so the flag criteria and error bounds come from comparing K5
  fast with K5 exact on the corpus. This replaces "step 0" of #24. Targets:
  * vertex deviation from exact ≤ the refinement tolerance;
  * the 1.1.1 non-monotone cases (1RCL, 2IEM, 1GZM, 8Y7M, 1AON at 0.5 Å,
    6VXX at 0.25 Å) all flagged or within the bound;
  * flagged share reported (expected ≪ 1 %);
  * area and volume within 10⁻⁴ relative.

## 7. Memory

* **APU (`hipDeviceAttributeIntegrated` = 1).**
  * Batch arenas are device-visible host allocations (`hipHostMalloc`),
    written by the CPU and read by the device with no copies (201 GB/s
    measured).
  * The arrangement is registered with `hipHostRegister` or copied once,
    which takes ≈ 10 ms per GB.
* **Discrete GPUs.**
  * The arrangement is copied once.
  * Batches use pinned staging buffers and asynchronous copies on the batch
    streams.
  * Device memory sets the batch size: `hipMemGetInfo` minus the resident
    arrangement minus a margin.
* **Memory guard** (`estimate_memory`, `--max-memory`).
  * Host memory is the same as on the CPU path, plus pinned batch arenas.
  * Device memory is checked against the resident arrangement plus two
    batches. When the arrangement does not fit, run on the CPU (`auto`) or
    stop with the memory error (explicit `hip`).
  * 3J3Q needs ≈ 2 GB of arrangement.

## 8. Expected performance (to be measured)

The estimates below extrapolate the measured kernel speeds (§2.4):

* the f32 refinement evaluation is costed at ≈ 2× a sample evaluation, plus
  iteration divergence;
* device gathers at ≈ 10 % of device time;
* the CPU keeps the cell pass and attributes, overlapped.

| Mode (gfx1151 + 16 Zen 5) | 1AON 0.25 Å (3.18 s on CPU) | 3J3Q 1 Å (27.7 s on CPU) |
|---|---|---|
| Kernel only | ≈ 2.8 s (1.15×) | ≈ 25 s (1.1×) |
| Exact: device classify/gather/sample, CPU + device f64 refinement | ≈ 1.8–2.0 s (1.6–1.8×) | ≈ 17–19.5 s (1.4–1.6×) |
| Fast: whole mesh stage on the device | ≈ 1.2–1.3 s (2.5×) | ≈ 9–10 s (2.8–3×) |

* **Remaining CPU time** (1AON 0.25 Å, the 0.84 s outside the mesh stage)
  then dominates: measure (validate, components, area/volume), OBJ
  formatting, and for 3J3Q the 4 s arrangement. These are CPU tasks outside
  this phase.
* **On MI250X/MI300** (f64 at half or full f32 rate), exact mode should
  approach the fast-mode row. Not measured, and not planned for now
  (decision 2026-11-20, §11).

## 9. Crates, API, build, distribution

* **One library API, one CLI.** `sesmesh` stays the only library crate
  users depend on, and `sesmesh-cli` the only CLI. GPU back ends are Cargo
  features: `hip` now, `cuda` and `metal` later. There are no per-back-end
  crates with their own APIs, for three reasons:
  * Exact mode reuses the CPU path's own code (gathering, packing,
    attributes) and is tested against it. That code is `pub(crate)` in
    `sesmesh`. A back-end crate built on top of `sesmesh` would need those
    internals made public, and they would become semver surface.
  * One `Backend` option gives users run-time selection and fallback
    (`auto`) through the same API in every build.
  * crates.io names are permanent, and every crate is one more to publish
    in lockstep.
* **Implementation crate: `sesmesh-gpu`** (decided 2026-11-20; the name was
  free on crates.io that day). One crate for all vendors, with features `hip`, `cuda`
  and `metal`:
  * All the `unsafe` code:
    * FFI, with `dlopen` (through `libloading`) of `libamdhip64.so`, and
      later of `libcuda.so`, so that a binary built with a feature still
      runs, and falls back, on machines without that runtime;
    * Metal through a target-gated dependency (macOS only).
  * Kernel sources:
    * HIP C++ shared with CUDA (a small macro header covers the
      differences);
    * Metal MSL separately;
    * the `#[repr(C)]` batch descriptors, shared by all vendors.
  * A safe API in plain types (as built in T11.1, except the launchers):
    * `Runtime::load()` (the runtime library) and `Runtime::open(index)`,
      which returns a `Device` with the embedded code object loaded and the
      self-test passed. `DeviceInfo`: name, arch, CUs, native wave size,
      integrated, memory, code object, runtime version. An f64 rate class
      comes with tuning (T11.8), if it is needed;
    * the loaded module stays internal to `Device`;
    * `DeviceBuffer<T>` and `HostBuffer<T>` (pinned, device-visible,
      `Deref` to `[T]`) for a sealed set of plain number types;
    * `Stream`, `Event`;
    * typed launchers that take slices and plain parameter structs. K1,
      K2 and K4 (T11.3′) are `Resident` (the arrangement uploaded once:
      frame-local `f32` copies, blocker tables, the CPU's cell grids)
      and `Batch` (pinned per-batch buffers): `Resident::run` classifies,
      gathers and samples a batch of bricks into a `Batch`. The kernels
      bound every index by the uploaded lengths and the validated tables,
      so bad input gives wrong values, never out-of-bounds accesses, and
      the API stays safe. (The first T11.3, `Sampler` with CPU-gathered
      jobs in staging words, was slower than the CPU and was removed;
      `docs/gpu-prior-art.md` §4.)
  * It knows no sesmesh type; the packing code lives in `sesmesh`. No
    third-party types in its API.
  * **One crate rather than `sesmesh-hip`, `-cuda` and `-metal`.** HIP and
    CUDA share kernel source and all three share the descriptor ABI. A
    Cargo package cannot include files from a sibling crate, so separate
    crates would need a fourth, common one. Splitting would not buy
    isolation either: the build script runs only the enabled vendors'
    compilers, and target-gated dependencies keep Metal off Linux.
  * **Versioning.** Published with every release at the workspace version,
    from the first release in which `sesmesh` depends on it (until then
    `publish = false`). `sesmesh` requires it as `=x.y.z`. It is documented as an
    implementation detail whose API may change in minor releases.
* **`sesmesh`:**
  * Optional dependency on `sesmesh-gpu` behind the feature `hip` (later
    `cuda`, `metal`), off by default. `unsafe_code = "forbid"` stays.
  * `SurfaceParams::backend: Backend` (`with_backend`), with
    `#[non_exhaustive] enum Backend { Cpu, Auto, Hip(HipOptions) }`
    (`Backend::hip(index)`, `name()`, `is_compiled()`).
    `HipOptions { device }` is `#[non_exhaustive]` and defaults to device 0;
    `refine: GpuRefine` (default `Exact`) is added with fast mode (T11.7).
  * Default `Cpu`: default output stays byte-identical.
  * `Auto` uses device 0 when it opens and passes its self-test, else the
    CPU. It warns (`WarningKind::GpuFallback`) only when a GPU runtime is
    installed but its device cannot be used, not when there is no GPU.
  * An explicit `Hip` with no usable device returns `Error::Backend`.
    Without the feature, `Hip` returns `Error::Unsupported`, so the API is
    the same in every build.
  * Devices are opened once per process (runtime library and devices are
    cached), so repeated `compute_surface` calls do not repeat the
    self-test; `SurfaceEngine::new` opens the device and fails early.
  * `SurfaceStats::backend: &'static str` (`"cpu"`, `"hip"`) and
    `SurfaceStats::device: Option<String>` (`"AMD Radeon 8060S Graphics
    (gfx1151, gfx11-generic)"`), `Timings::device` (time to open the
    device, ≈ 45 ms on gfx1151, mostly HIP initialisation), plus the
    flagged-edge count in fast mode.
* **CLI:**
  * `--backend cpu|auto|hip`, `--gpu-device N` (with `--backend hip`
    only), `--gpu-refine exact|fast` (T11.7);
  * `--timings` names the back end, device and device-open time; the CPU
    line is unchanged;
  * JSON stats gain `parameters.backend`, `parameters.gpu_device`,
    `engine.backend`, `engine.device`, `timings_ms.device`, later
    `engine.flagged_edges`;
  * OBJ/PLY/MSMS headers do not name the back end, so the files are the
    same on every back end;
  * a usage error (exit 2) when the binary lacks the feature; an unusable
    device is a runtime error (exit 1).
* **Kernel delivery (HIP): code objects built at build time** (decided
  2026-11-20).
  * **Build.** The build script of `sesmesh-gpu` runs `hipcc --genco`, one
    code object per GPU family, using ROCm's generic targets:
    `gfx10-3-generic` (RDNA 2), `gfx11-generic` (RDNA 3 and 3.5),
    `gfx12-generic` (RDNA 4). Data-centre targets (`gfx90a`,
    `gfx9-4-generic`) are added when those GPUs are tested. The list can be
    overridden with `SESMESH_HIP_ARCHS`; specific targets load before
    generic ones. Without `hipcc` the build fails with a message saying so;
    `SESMESH_GPU_NO_KERNELS=1` builds without kernels for code checks
    (clippy, rustdoc) on machines without ROCm.
  * **Measured.** A `gfx11-generic` object ran the sampling kernel on
    gfx1151 at the same speed as a `gfx1151` object (60 ms on 1AON 0.5 Å)
    and bit-identical to the CPU. Each object is ≈ 17 KB and takes
    ≈ 0.4 s to build.
  * **Flags.** `-O3 -ffp-contract=off -fno-fast-math`; wave64 on RDNA,
    where it was 1.3× faster.
  * **Loading.** At run time `sesmesh` loads the object whose target the
    device supports. A device that no object covers is unsupported, and
    `auto` falls back to the CPU.
  * **Self-test.** When a device opens, a short test (the arithmetic
    agreement of §2.4 on 4 096 values, f32 on device buffers and f64 on
    pinned host memory) confirms that it computes as the CPU does. A device
    that fails is not used.
  * **docs.rs.** The build script skips kernel compilation when `DOCS_RS`
    is set.
  * **Why not hiprtc at run time.** Cold compilation took 200 ms for the
    sampling kernel alone, more for K1–K5; ROCm's comgr disk cache brings
    repeats down to 14 ms. The code would also depend on the ROCm version
    installed, so fast-mode output would be reproducible per binary and
    ROCm version, not per binary. hiprtc's advantages (no ROCm needed to
    build, any architecture) keep it as a later option, if a GPU outside
    the generic targets is needed.
  * CUDA and Metal choose their own delivery in #25/#26. Likely: `nvcc` to
    PTX, embedded and compiled by the driver; Metal compiled from source at
    run time, the platform's usual way.
* **Distribution.**
  * crates.io: `cargo install sesmesh-cli --features hip` on a machine with
    ROCm (`hipcc` at build time).
  * GitHub releases: the tier-1 CPU binaries stay as they are. A further
    `x86_64-unknown-linux-gnu-hip` archive is built in a ROCm container
    (build only, no GPU needed); it also runs on machines without ROCm,
    using the CPU.
  * Later, possibly one Linux binary with `hip` and `cuda`, and macOS
    binaries with `metal`. Metal is part of the OS, and the default back
    end stays the CPU, so output is unchanged.
* **Platforms.** ROCm on Linux x86-64 first. The Windows HIP SDK and HIP on
  NVIDIA are not planned (#25 is the native CUDA back end).
* **Shared with #25/#26.** These parts do not depend on the device:
  * `Backend`;
  * the batch scheduler and the packing;
  * the exactness rules;
  * the fast-mode criteria.

  Metal has no f64, so it gets fast mode or CPU refinement only.

## 10. Plan (Phase 11)

> **Status (2026-11-21).** Done: T11.0, T11.1, T11.2 (committed); spike S1
> (`docs/gpu-prior-art.md` §5.2); T11.3′, exact device classification,
> gathering and sampling (`docs/gpu-prior-art.md` §5.3); spike S2, f32
> refinement on the device (`docs/gpu-prior-art.md` §5.4). Deviations: the
> first T11.3 (CPU-gathered jobs staged for the device) was measured
> 2–27 % slower than the CPU and replaced by T11.3′; T11.4–T11.7 (device
> f64) are dropped; the tasks that follow are those of
> `docs/gpu-prior-art.md` §5.3 (S2, T11.5′, T11.6′).

Every task's gate is checked on this machine. GitHub runners have no GPU,
and the manual CI job only builds the feature in a ROCm container.

| ID | Task | Gate |
|---|---|---|
| T11.0 | Analysis and spike (this document, `tools/hip-spike/`). | Done 2026-11-20. |
| T11.1 | `sesmesh-gpu` skeleton (feature `hip`): `dlopen` FFI, device query, generic-target code objects (`gfx10-3-generic`, `gfx11-generic`, `gfx12-generic`) and their selection, buffers, streams, events, errors, device self-test, `DOCS_RS` skip; the arithmetic-agreement test of §2.4 as a crate test (skips without a device). | Tests pass on gfx1151; a build without `hipcc` fails with a clear message; a binary on a machine without ROCm reports "no device". |
| T11.2 | `Backend` in `SurfaceParams`, CLI flags, stats, CPU fallback, errors; docs for the option. | Default output unchanged (`cmp` on the suite); `--backend hip` without feature or device fails as specified. |
| T11.3 | Batch scheduler plus K2 sampling with CPU-gathered lists packed into device-visible arenas (SES, SAS, vdW). | `cmp` of OBJ/PLY/stats against CPU on the test suite, the validation corpus (53) and `sweep` (200) at 0.25/0.5/1 Å; repeated runs and batch sizes 1…N identical; time 4HHB/1AON/3J3Q. |
| T11.4 | K5 exact (f64) with CPU-gathered `gather_local` lists; K3 edge finding; attributes on the CPU from K5's output. | Byte-identical as in T11.3; f64 refinement throughput vs 16 cores; per-edge results equal `vertex()`. |
| T11.5 | Device gathers (K2, K4) and classification (K1) in f64, LDS lists with global overflow; resident arrangement. | Byte-identical; per-kernel oracle tests (lists equal `gather_local`, states equal `uniform_sign`); largest list sizes recorded. |
| T11.6 | Pipelining and hybrid exact refinement (CPU + device share edges); device memory budget; memory guard; cancellation between batches; progress. | Byte-identical under any thread count and split; cancel < 100 ms; `--max-memory` refuses oversize device runs. |
| T11.7 | Fast mode: anchored coordinates, f32 K5, flag criteria from the exact-vs-fast comparison, f64 fallback on the device. | Same connectivity as CPU; vertex deviation ≤ refinement bound; 1.1.1 outlier cases covered; flagged share; area/volume within 10⁻⁴; deterministic on repeat. |
| T11.8 | Benchmarks (`benches/surface.rs`, `benches/field.rs` with the back end), end-to-end table (4HHB, 1AON at 0.5/0.25 Å, 3J3Q at 1 Å; sampling alone, + refinement, transfers), tuning (LDS staging, wave size, sample order). | Numbers in this document and Design §7.7; both machines' CPU times given. |
| T11.9 | Docs and release: Spec §3.2 (GPU no longer a non-goal), Design §7.5 note, §11, §17, §18, `docs/cli.md`, README, site, CHANGELOG; `docs/release.md` local validation with the feature; 1.2.0 (new opt-in feature, minor). | Local validation list plus GPU tests. |

Order and exits:

* T11.3 is the first useful exit: an exact back end, small gain, all of the
  infrastructure.
* T11.5 completes the exact pipeline.
* T11.7 brings the speed.
* T11.4 can be skipped on GPUs where exact f64 refinement is not wanted,
  but it is the oracle for T11.7 and stays.

Related CPU work, not in this phase:

* **SIMD exact refinement on the CPU.** Lanes are the edges of a sub-block,
  f64, AVX-512/NEON. This is byte-identical by the same argument as §3, and
  vertex refinement is 41–59 % of mesh time.
* **Reuse of the sampling gather for refinement.** This needs the slightly
  larger refinement ball for both, which can change output in rounding
  cases.
* **Arrangement and measure stages** at the scale of 3J3Q.

## 11. Decisions

1. **Default precision.** *Decided 2026-11-20:* `exact` is the default on
   every GPU (byte-identical to the CPU back end, ≈ 1.6× here); `fast`
   (≈ 2.5–3×, positions within the refinement bound) is opt-in with
   `--gpu-refine fast` / `GpuRefine::Fast`. `Backend::Auto` therefore never
   changes output.
2. **Data-centre GPUs.** *Decided 2026-11-20:* not now. Phase 11 targets
   consumer RDNA and is tested on gfx1151. MI250X (gfx90a) and MI300
   (gfx942) code objects, validation and benchmarks come later.
3. **Crates.** *Decided 2026-11-20:* one implementation crate,
   `sesmesh-gpu`, with vendor features. No per-back-end API crates: back
   ends are features of `sesmesh` and `sesmesh-cli` (§9).
4. **Kernel delivery.** *Decided 2026-11-20:* HIP code objects built at
   build time for ROCm's generic targets, plus a self-test when a device
   opens. hiprtc only later, if a GPU outside the generic targets is needed
   (§9).
