// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// HIP kernels of sesmesh (docs/05-gpu-hip.md). Built by build.rs with
// -ffp-contract=off and without fast math: every expression keeps the
// operation order of the Rust code it mirrors (left to right), so results
// equal the CPU's bit for bit (§3).

#include <hip/hip_runtime.h>

// Self-test (§9): IEEE operations whose results the host recomputes. Keep in
// sync with `selftest::reference_f32` / `reference_f64` (SELFTEST_OPS each).
#define SELFTEST_OPS 8

extern "C" __global__ void sesmesh_selftest_f32(const float* a, const float* b, unsigned* out,
                                                unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i], y = b[i];
    const float dx = x - y, dy = y * 0.5f - x, dz = x * y - 1.0f;
    const float d = sqrtf(dx * dx + dy * dy + dz * dz);
    const float r[SELFTEST_OPS] = {x + y,       x * y,       x / y,      sqrtf(fabsf(x)),
                                   d,           (d - y) / d, fminf(x, y), fabsf(d - fabsf(y))};
    for (int k = 0; k < SELFTEST_OPS; k++) out[SELFTEST_OPS * i + k] = __float_as_uint(r[k]);
}

extern "C" __global__ void sesmesh_selftest_f64(const double* a, const double* b,
                                                unsigned long long* out, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = a[i], y = b[i];
    const double dx = x - y, dy = y * 0.5 - x, dz = x * y - 1.0;
    const double d = sqrt(dx * dx + dy * dy + dz * dz);
    const double r[SELFTEST_OPS] = {x + y,       x * y,       x / y,
                                    sqrt(fabs(x)), d,         (d - y) / d,
                                    (double)(float)(x * 1e3), fabs(d - fabs(y))};
    for (int k = 0; k < SELFTEST_OPS; k++)
        out[SELFTEST_OPS * i + k] = (unsigned long long)__double_as_longlong(r[k]);
}
