# GPU back end (CUDA)

Phase 12, issue #25, 2026-11-21, branch `phase-12-gpu-cuda`. The CUDA back
end reuses the design of the HIP back end (`docs/05-gpu-hip.md`,
`docs/gpu-prior-art.md` §5): the same kernels, the same batch descriptions
and the same mesh pipeline in `sesmesh`. This document records what differs
for CUDA and the results on the first NVIDIA GPU tested.

## 1. Summary

* `Backend::Cuda(CudaOptions)` / `--backend cuda` (features `cuda` of
  `sesmesh-gpu`, `sesmesh` and `sesmesh-cli`). `Backend::Auto` tries HIP
  device 0, then CUDA device 0, then the CPU.
* Exact mode writes the CPU back end's files byte for byte; `--refine fast`
  refines the crossing edges in f32 on the device, as with HIP. Device
  code is f32 and integer only.
* One kernel source for both vendors: `kernels/sesmesh.cuh`, included by
  `kernels/hip/sesmesh.hip` and `kernels/cuda/sesmesh.cu`. It uses no
  warp or wave intrinsics, so the CUDA build needed no change to it.
* The CUDA driver API is loaded at run time (`libcuda`, through
  `libloading`), as HIP's runtime is: a binary built with the feature runs,
  and falls back, where the NVIDIA driver is missing. No CUDA runtime
  library (`libcudart`) is needed.
* The difference from HIP is memory. HIP was built on an APU, where the
  device reads pinned host memory at full speed. A discrete GPU reads it
  over PCIe, so CUDA batches live in device memory: the host copies the
  sub-block records down for its scans between passes, and the results
  (sample values; in fast mode the edge results) into pinned host buffers.
* Tested on a laptop with a GeForce RTX 2070 Max-Q (sm_75, 8 GB, PCIe 3
  x16) and a 6-core i7-9750H: §5.

## 2. Design

### 2.1 Kernels and exactness

The HIP kernel file became `kernels/sesmesh.cuh`, with the HIP header
include moved to the HIP entry file. `nvcc` compiles it with

```text
nvcc -fatbin -O3 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false
```

the counterparts of `hipcc`'s `-ffp-contract=off -fno-fast-math`:

* `-fmad=false`: no contraction of `a * b + c` into a fused multiply-add.
  The three `fmaf` calls the kernels make on purpose (error-free products
  of the double-float coordinates in fast mode) stay fused, as on the CPU.
* `-prec-div=true -prec-sqrt=true`: IEEE round-to-nearest division and
  square root instead of the fast approximations.
* `-ftz=false`: subnormals are kept.

The PTX was checked once: every f32 operation is `.rn` (round to nearest
even), the only `fma.rn` are the three explicit ones, and there is no f64
instruction. The self-test at device open (`docs/05-gpu-hip.md` §9: f32
`+ − × ÷ √` and compound expressions on 4 096 random pairs, including
subnormals, on device memory and on pinned host memory) and the crate
test on 2 M pairs give the CPU's bits.

### 2.2 Fat binary

`build.rs` compiles one fat binary with machine code (SASS) for each
architecture of `SESMESH_CUDA_ARCHS`, by default sm_75, 80, 86, 89, 90, 100
and 120 (Turing to Blackwell; those the `nvcc` does not support are left
out), and the PTX of the lowest. `cuModuleLoadData` picks the machine code
of the device's architecture, or JIT-compiles the PTX for a newer one.
PTX keeps the same IEEE semantics, so the JIT path is exact too; the
self-test checks it on the device before use. `DeviceInfo::code` says
which code runs (`sm_75`, or `compute_75 PTX`). The fat binary for the seven
default architectures is 1.6 MB.

### 2.3 Driver API

`sesmesh_gpu::cuda` mirrors `sesmesh_gpu::hip`:

* `Runtime::load()` opens `libcuda.so.1` (`nvcuda.dll` on Windows) and
  resolves the 32 driver functions it uses, with their `_v2` names where
  CUDA versions them. `Runtime::open(index)` reads the device's
  attributes, retains its primary context, loads the fat binary and runs
  the self-test. Devices need unified addressing (every 64-bit CUDA
  device has it).
* The primary context is shared by the `Device` and every buffer, stream
  and event created from it (reference counted), and made current before
  each driver call, so all of them can be used from any thread and may
  outlive the `Device`.
* `DeviceBuffer<T>` (`cuMemAlloc`), `HostBuffer<T>` (pinned,
  `cuMemHostAlloc`), `Stream`, `Event`, and `Resident` / `Batch` with
  HIP's API.

### 2.4 Batches on a discrete GPU

`cuda::Batch` holds every working buffer in device memory, each with a
pinned host mirror where the host needs the contents:

| Buffer | Device | Host copy |
|---|---|---|
| Brick origins | read by every pass | written by the host, copied down once per batch |
| Sub-block records | written by count and sample, read by fill | copied up for the list scan (base offsets), down again; up after sampling |
| Candidate lists | written by fill, read by sample | none (`Batch::id_lists` copies on request) |
| Sample values | written by sample | copied up after sampling |
| Edge counts and offsets (fast) | count pass, refine pass | counts up for the scan, offsets down |
| Edge results (fast) | written by refine | copied up |

Copies are synchronous on the null stream, as the launches are; the ring of
three batches in `sesmesh` (`docs/05-gpu-hip.md` §5) overlaps a batch's
device work and copies with the meshing of earlier batches on the CPU
threads.

### 2.5 One pipeline for both back ends

The batch descriptions, record and edge encodings, validation and self-test
reference moved from `sesmesh_gpu::hip` to the shared module
`sesmesh_gpu::batch` (re-exported by both back ends). Two traits there,
`DeviceResident` and `DeviceBatch`, carry what the mesh pipeline needs;
both back ends implement them by forwarding to their inherent methods.
`sesmesh`'s pipeline (`field/gpu.rs`) is generic over them, and a small
enum (`GpuDevice`) dispatches to the back end that `Backend` resolved. The
device tests run on every back end of the build whose device opens.

## 3. Building

```bash
cargo build --release -p sesmesh-cli --features cuda      # needs nvcc
SESMESH_GPU_NO_KERNELS=1 cargo clippy -p sesmesh-gpu --features cuda   # checks only
```

`nvcc` is found through `NVCC`, `CUDA_PATH`, `CUDA_HOME`,
`/usr/local/cuda/bin`, `/opt/cuda/bin` or `PATH`; without it the build
stops with an error saying so. Building does not need a GPU or the driver.
`tools/gpu/cuda-build.Containerfile` is an image with Rust and `nvcc` only
(Ubuntu 24.04, CUDA 13.3), for building on a machine without CUDA:

```bash
podman build -t localhost/sesmesh-cuda-build -f tools/gpu/cuda-build.Containerfile .
podman run --rm -v "$PWD:$PWD" -w "$PWD" localhost/sesmesh-cuda-build \
  cargo build --release -p sesmesh-cli --features cuda
```

Mount the checkout at the same path inside the container: test binaries
record their source directory (`CARGO_MANIFEST_DIR`) for their fixtures.
The binaries load the host's `libcuda` when they run, so they can be
copied to the GPU machine.

## 4. Testing

```bash
SESMESH_GPU_REQUIRE=1 cargo test -p sesmesh-gpu --features cuda
SESMESH_GPU_REQUIRE=1 cargo test --release -p sesmesh -p sesmesh-cli --features sesmesh-cli/cuda
BACKEND=cuda SIMD=avx2 tools/gpu/gate.sh /tmp/gate /tmp/gate-inputs.txt
```

The gate (`tools/gpu/README.md`) can split its work over two machines:
`MODE=cpu` writes the CPU runs' reference records (digests of the OBJ and
its face lines, and the statistics) on a fast CPU, and `MODE=gpu
REF=…/ref` runs only the GPU and compares with them. Both need the same
binary, inputs and SIMD level; the CPU's output does not depend on the
SIMD level (checked: AVX-512 and AVX2 write the same bytes), but the
statistics name it, so `SIMD` sets one both machines support.

## 5. Results

### 5.1 Speed

Complete CLI runs (`--timings`, OBJ written to `/dev/null`), best of 5
(3J3Q: best of 3), on the laptop: GeForce RTX 2070 with Max-Q Design
(sm_75, 36 SMs, 8 GB, PCIe 3 x16) and an i7-9750H (6 cores, 12 threads),
on mains power, idle before each series (load < 0.3, GPU 0 %). Device
opening (≈ 110 ms per process: driver initialisation, module load,
self-test) is excluded. The Zen 5 and M4 columns are the CPU times of
`docs/05-gpu-hip.md` §12 (same CPU code), for scale. Total time, with
the mesh stage in brackets:

| Input | M4 CPU | Zen 5 CPU | i7 CPU | RTX exact | RTX fast | Exact vs i7 | Fast vs i7 |
|---|---|---|---|---|---|---|---|
| 4HHB 0.25 Å | 378 (298) ms | 243 (169) ms | 977 (766) ms | 651 (326) ms | 470 (152) ms | 1.50× | 2.08× |
| 1AON 0.5 Å | 2 142 (1 779) ms | 1 382 (1 036) ms | 5 481 (4 468) ms | 3 035 (2 089) ms | 1 703 (631) ms | 1.81× | 3.22× |
| 1AON 0.25 Å | 5 619 (4 482) ms | 3 403 (2 504) ms | not run | 6 984 (4 497) ms | 4 192 (1 357) ms | | |
| 3J3Q 1 Å | 47.6 (39.2) s | 28.9 (21.4) s | not run | 83.3 (62.6) s | 35.3 (13.1) s | | |
| 1AON SAS 0.25 Å | 1 208 (410) ms | 932 (256) ms | not run | 2 563 (630) ms | 3 003 (771) ms | | |
| 1AON vdW 0.25 Å | 2 284 (552) ms | 1 758 (356) ms | not run | 5 172 (975) ms | 5 521 (944) ms | | |

The i7's own CPU runs were limited to the two smaller cases (the laptop's
CPU is not the point of the test).

* Against the laptop's own CPU: exact mode 1.50–1.81×, fast mode
  2.08–3.22×.
* The laptop's CPU bounds both modes. In exact mode the f64 vertex
  refinement runs on the 6 cores, as on every GPU back end. In fast mode
  the GPU shortens the SES mesh stage 2.1–4.8× against exact mode, and what
  is left is mostly the stages outside it (arrangement, assembly, measure,
  OBJ formatting), all on the CPU: 2.8 s of 1AON's 4.2 s at 0.25 Å, where
  the 16 Zen 5 cores spend 0.9 s.
* So the laptop in fast mode beats the M4's CPU on the larger inputs (1AON
  0.25 Å 4.2 s vs 5.6 s, 3J3Q 35 s vs 48 s) but not the 16 Zen 5 cores
  (3.4 s, 28.9 s), and stays behind the Radeon 8060S host (1.41 s, 12.3 s
  in fast mode), whose CPU is the Zen 5.
* SAS and vdW gain nothing (75–83 % of their time is outside the mesh
  stage), and in fast mode they are slower than in exact mode (1AON SAS
  0.25 Å: 3.00 vs 2.56 s). Not analysed yet (§6).

### 5.2 Gate

`tools/gpu/gate.sh` with `BACKEND=cuda`, the CPU references made with
`MODE=cpu SIMD=avx2` on the 16-core Zen 5 and the GPU runs with
`MODE=gpu` on the laptop, same binary (built in the container):

* SES at 1, 0.5 and 0.25 Å on the 258 inputs of the `small`, `bench`,
  `validation` and `sweep` sets: 774 cases, all meshed on both back ends,
  **no difference**. Exact mode wrote the CPU's OBJ and statistics byte for
  byte; fast mode wrote the CPU's face lines, the same OBJ when repeated,
  and area and volume within 3.5·10⁻⁸ (relative) of the CPU's; 39 689
  edges were refined on the CPU in fast mode.
* SAS and vdW at 1 and 0.5 Å on the 53 inputs of the validation set: 212
  cases, no difference; fast area and volume within 3.5·10⁻⁷.
* The largest case, 3J3Q at 0.25 Å (800 M triangles, a 72 GB OBJ), ran on
  the laptop's 62 GB with swap and fits the GPU's 8 GB: the device holds
  the arrangement and three batches, not the mesh.
* Run time: the references took 614 s (SES) and 34 s (SAS/vdW) on the
  Zen 5, the GPU side 5 456 s and 465 s on the laptop. Most of the SES
  side is 3J3Q at 0.25 Å: its swap and, before the fix, `compare.py`
  reading its OBJ files into Python to hash them (it now uses `sha256sum`
  and `grep`).

The device tests (`tests/cuda.rs`; in `sesmesh` the device sampling
values, exact meshes and fast meshes for every batch size, cancellation,
the `Auto` order; the CLI's `--backend cuda` and `--refine fast` tests)
pass on the RTX 2070, and the HIP ones still pass on the gfx1151 after the
kernel move.

## 6. Open

* Fast mode on SAS and vdW is slower than exact mode on this GPU, unlike
  on the APU. Candidates: the large fast-mode batches (≥ 8 192 bricks,
  tuned on the APU), whose first batch nothing overlaps, and the extra
  copies of edge results over PCIe. Measure with the batch size and a
  device timeline.
* Whether the RTX 2070 or the i7 bounds fast mode's SES mesh stage (1.36 s
  for 1AON at 0.25 Å, 0.53 s on the Radeon 8060S host): thread scaling and
  a device timeline (CUDA events per pass) would tell. A desktop GPU with a
  fast CPU would give the GPU's own numbers.
* Copies are synchronous on the null stream; the ring overlaps them with
  meshing, but not with the next batch's kernels. Streams and async copies
  would, if the device turns out to bound fast mode.
* Device opening takes ≈ 110 ms (driver initialisation, module load and
  self-test), against ≈ 45 ms for HIP; it is paid once per process.
* GPU buffers are still not in `--max-memory`'s estimate (as for HIP).
* Only sm_75 is tested; the fat binary's other architectures and its PTX
  path are untested on hardware.
