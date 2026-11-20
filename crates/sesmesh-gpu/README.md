# sesmesh-gpu

GPU back ends of [sesmesh](https://github.com/uvar-apps/sesmesh). This is an
implementation crate: use sesmesh's GPU features and its `Backend` option
instead (Phase 11, `docs/05-gpu-hip.md`). Its API serves sesmesh and may
change in minor releases.

All of sesmesh's `unsafe` code lives here (`sesmesh` forbids it). Device
code follows the exactness rules of `docs/05-gpu-hip.md` §3: kernels
reproduce the CPU's floating-point results bit for bit.

## Features

| Feature | GPUs | Build needs | Run needs |
|---|---|---|---|
| `hip` | AMD, through HIP (ROCm) | ROCm's `hipcc` | the HIP runtime (`libamdhip64`), loaded when a device is opened |

Without features the crate is empty. With `hip`, a binary built on a ROCm
machine still runs where ROCm is missing: opening a device then fails with
"GPU runtime library not found", and sesmesh falls back to the CPU.

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

## Testing

```bash
SESMESH_GPU_REQUIRE=1 cargo test -p sesmesh-gpu --features hip
```

The device tests skip without a HIP runtime, device or kernels;
`SESMESH_GPU_REQUIRE=1` turns a skip into a failure. They check the device
properties, buffers, events, use from several threads, and that one million
pairs of f32 and f64 inputs (including subnormals) give the CPU's results
bit for bit. Tested on gfx1151 (Radeon 8060S, ROCm 7) with the
`gfx11-generic` object.

## Licence

MIT OR Apache-2.0, like sesmesh. Copyright (c) 2026 Ugo Varetto.
