// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// Feasibility spike kernels (#24): IEEE agreement with the CPU, f32/f64
// throughput, launch latency, memory bandwidth. Compiled with
// -ffp-contract=off and without fast math.

#include <hip/hip_runtime.h>

#define OPS 8

// Same expression order as Rust (left to right).
extern "C" __global__ void arith_f32(const float* a, const float* b, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i];
    float dx = x - y, dy = y * 0.5f - x, dz = x * y - 1.0f;
    float d = sqrtf(dx * dx + dy * dy + dz * dz);
    float r[OPS] = {x + y, x * y, x / y, sqrtf(fabsf(x)), d, (d - y) / d,
                    fminf(x, y), fabsf(d - fabsf(y))};
    for (int k = 0; k < OPS; k++) out[OPS * i + k] = __float_as_uint(r[k]);
}

extern "C" __global__ void arith_f64(const double* a, const double* b, unsigned long long* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = a[i], y = b[i];
    double dx = x - y, dy = y * 0.5 - x, dz = x * y - 1.0;
    double d = sqrt(dx * dx + dy * dy + dz * dz);
    double r[OPS] = {x + y, x * y, x / y, sqrt(fabs(x)), d, (d - y) / d,
                     (double)(float)(x * 1e3), fabs(d - fabs(y))};
    for (int k = 0; k < OPS; k++) out[OPS * i + k] = (unsigned long long)__double_as_longlong(r[k]);
}

// Throughput: 8 independent chains of (mul, add) or fma per thread.
#define CHAINS(T, STEP)                                                     \
    T c0 = threadIdx.x * (T)1e-3, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3,  \
      c4 = c0 + 4, c5 = c0 + 5, c6 = c0 + 6, c7 = c0 + 7;                  \
    for (int k = 0; k < iters; k++) {                                      \
        c0 = STEP(c0); c1 = STEP(c1); c2 = STEP(c2); c3 = STEP(c3);        \
        c4 = STEP(c4); c5 = STEP(c5); c6 = STEP(c6); c7 = STEP(c7);        \
    }                                                                      \
    out[blockIdx.x * blockDim.x + threadIdx.x] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;

#define MULADD32(c) ((c) * s + 0.5f)
#define FMA32(c) fmaf((c), s, 0.5f)
#define MULADD64(c) ((c) * s + 0.5)
#define FMA64(c) fma((c), s, 0.5)
#define SQRT32(c) (sqrtf(c) + s)
#define SQRT64(c) (sqrt(c) + s)
#define DIV32(c) (s / (c) + 1.0f)
#define DIV64(c) (s / (c) + 1.0)

extern "C" __global__ void rate_muladd_f32(float* out, int iters, float s) { CHAINS(float, MULADD32) }
extern "C" __global__ void rate_fma_f32(float* out, int iters, float s) { CHAINS(float, FMA32) }
extern "C" __global__ void rate_muladd_f64(double* out, int iters, double s) { CHAINS(double, MULADD64) }
extern "C" __global__ void rate_fma_f64(double* out, int iters, double s) { CHAINS(double, FMA64) }
extern "C" __global__ void rate_sqrt_f32(float* out, int iters, float s) { CHAINS(float, SQRT32) }
extern "C" __global__ void rate_sqrt_f64(double* out, int iters, double s) { CHAINS(double, SQRT64) }
extern "C" __global__ void rate_div_f32(float* out, int iters, float s) { CHAINS(float, DIV32) }
extern "C" __global__ void rate_div_f64(double* out, int iters, double s) { CHAINS(double, DIV64) }

extern "C" __global__ void empty() {}

extern "C" __global__ void stream(const float4* src, float4* dst, long n) {
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
        float4 v = src[i];
        v.x += 1.0f;
        dst[i] = v;
    }
}
