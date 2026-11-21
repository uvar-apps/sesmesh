# sesmesh-gpu

GPU back ends of [sesmesh](https://github.com/uvar-apps/sesmesh). This is an
implementation crate: use sesmesh's GPU features and its `Backend` option
instead (Phase 11, `docs/05-gpu-hip.md`; Phase 12, `docs/06-gpu-cuda.md`).
Its API serves sesmesh and may change in minor releases.

All of sesmesh's `unsafe` code lives here (`sesmesh` forbids it). Device
code follows the exactness rules of `docs/05-gpu-hip.md` §3: kernels
reproduce the CPU's floating-point results bit for bit.

## Features

| Feature | GPUs | Build needs | Run needs |
|---|---|---|---|
| `hip` | AMD, through HIP (ROCm) | ROCm's `hipcc` | the HIP runtime (`libamdhip64`), loaded when a device is opened |
| `cuda` | NVIDIA, through the CUDA driver API | CUDA's `nvcc` | the NVIDIA driver (`libcuda`), loaded when a device is opened |

Without features the crate is empty. With `hip` or `cuda`, a binary still
runs where the runtime is missing: opening a device then fails with "GPU
runtime library not found", and sesmesh falls back to the CPU. Both back
ends run the same kernels (`kernels/sesmesh.cuh`, included by
`kernels/hip/sesmesh.hip` and `kernels/cuda/sesmesh.cu`) on the same batch
descriptions (module `batch`, with the traits `DeviceResident` and
`DeviceBatch` that sesmesh's pipeline is written against).

## Building with `hip`

The build script compiles `kernels/hip/sesmesh.hip` with
`hipcc --genco -O3 -ffp-contract=off -fno-fast-math` (wave64 on RDNA) into
one code object per target and embeds them. At run time the first object
the device loads is used.

| Variable | Effect |
|---|---|
| `HIPCC` | `hipcc` to use; otherwise `$ROCM_PATH/bin`, `$HIP_PATH/bin`, `/opt/rocm/bin`, then `PATH` |
| `SESMESH_HIP_ARCHS` | Targets, comma or space separated. Default `gfx10-3-generic,gfx11-generic,gfx12-generic` (RDNA 2, 3/3.5, 4); specific targets (e.g. `gfx1151`) load before generic ones |
| `SESMESH_GPU_NO_KERNELS=1` | Build without `hipcc` and without kernels, for code checks (clippy, rustdoc) on machines without ROCm; opening a device reports it. Also implied under docs.rs (`DOCS_RS`) |

Without `hipcc` the build stops with an error that says so, rather than
producing a binary that silently has no GPU code.

## Building with `cuda`

The build script compiles `kernels/cuda/sesmesh.cu` with
`nvcc -fatbin -O3 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false`
into one fat binary: machine code for each architecture, and PTX for the
lowest, which the driver compiles for GPUs newer than the build knows. The
PTX has only round-to-nearest IEEE operations and no contracted
multiply-adds, as the exactness rules require. The driver API is used
directly (no CUDA runtime library), so running needs only the driver.

| Variable | Effect |
|---|---|
| `NVCC` | `nvcc` to use; otherwise `$CUDA_PATH/bin`, `$CUDA_HOME/bin`, `/usr/local/cuda/bin`, `/opt/cuda/bin`, then `PATH` |
| `SESMESH_CUDA_ARCHS` | Architectures, comma or space separated (`75` or `sm_75`). Default `75,80,86,89,90,100,120` (Turing to Blackwell), cut to those the `nvcc` supports |
| `SESMESH_GPU_NO_KERNELS=1` | As for `hip` |

A machine without an NVIDIA GPU can build it: `tools/gpu/cuda-build.Containerfile`
is an image with Rust and `nvcc` only (`docs/06-gpu-cuda.md` §3).

## Testing

```bash
SESMESH_GPU_REQUIRE=1 cargo test -p sesmesh-gpu --features hip
SESMESH_GPU_REQUIRE=1 cargo test -p sesmesh-gpu --features cuda
```

The device tests skip without a runtime, device or kernels;
`SESMESH_GPU_REQUIRE=1` turns a skip into a failure. They check the device
properties, buffers, events, use from several threads, and that one million
pairs of f32 inputs (including subnormals) give the CPU's results bit for
bit. Device code is f32 and integer only: no f64 runs on the GPU. Tested on gfx1151 (Radeon 8060S, ROCm 7) with the
`gfx11-generic` object, and on sm_75 (GeForce RTX 2070 Max-Q, driver for
CUDA 13.4) with the `sm_75` machine code.

## Kernels

| Kernel | API | Replaces (sesmesh) |
|---|---|---|
| Classification, gathering and field sampling, SES and union of spheres (SAS, vdW) | `hip::Resident`, `hip::Batch`; `cuda::Resident`, `cuda::Batch` | `uniform_sign`, `gather_ses`, `gather_spheres`, `ses_kernel`, `sphere_kernel` |
| Crossing-edge listing and refinement in f32 (fast mode) | `Resident::set_refinement`, `Resident::refine` | edge scan and `vertex()` in `mesh/extract.rs` (f64) |

`Resident` holds the arrangement on the device: the frame-local f32 copies
the CPU kernels use, the blocker tables and the CPU's cell grids, uploaded
once. `Resident::run` takes a batch of bricks into a `Batch` in three
passes. With HIP (an APU) the batch buffers are pinned host memory the
device reads and writes; with CUDA (a discrete GPU) they are device memory,
and the host copies the sub-block records down for its scans between passes
and the results into pinned host buffers. The passes are: count (cheap
classification, list sizes), fill (candidate lists, then the Lipschitz test
at each sub-block centre) and sample (one workgroup per sub-block, one lane
per sample). The list buffer is sized from the counts. Every f32 test that
mirrors an f64 test of the CPU has a slack on its inclusive side, so the
device's uniform sub-blocks are a subset of the CPU's and its lists
supersets. The samples the CPU evaluates get its values bit for bit. The
kernels bound every index by the uploaded lengths and the validated tables,
so bad input gives wrong values, not memory errors. sesmesh's tests compare
values and meshes with the CPU back end
(`SESMESH_GPU_REQUIRE=1 cargo test -p sesmesh --features hip gpu`).

`Resident::refine` (sesmesh's fast mode, `RefineMode::Fast`) refines the
crossing edges of the last `run` in f32 into the same `Batch`: a count pass
per sub-block, a host scan, then one workgroup per sub-block lists its
edges in the CPU's order (a scan in LDS) and runs the CPU's safeguarded
Newton / Illinois / bisection on them, against the sub-block's lists, in
brick-local coordinates: each position is the f32 frame-local copy plus its
f32 residual (`RefineRef`, uploaded by `set_refinement`), so the
frame-local rounding does not enter. It returns the last parameter, feature
and closest point per edge, and whether Newton / Illinois converged; the
CPU places the vertex. Its vertices are as close to the surface as the
CPU's f64 ones, but not bit-identical (`docs/gpu-prior-art.md` §5.4).

## Licence

MIT OR Apache-2.0, like sesmesh. Copyright (c) 2026 Ugo Varetto.
