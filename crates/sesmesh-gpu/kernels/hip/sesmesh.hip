// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// HIP kernels of sesmesh (docs/05-gpu-hip.md). Built by build.rs with
// -ffp-contract=off and without fast math: every expression keeps the
// operation order of the Rust code it mirrors (left to right), so results
// equal the CPU's bit for bit (§3). Device code uses f32 and integer
// arithmetic only: no f64 on the device.

#include <hip/hip_runtime.h>

// Self-test (§9): IEEE operations whose results the host recomputes. Keep in
// sync with `selftest::reference_f32` (SELFTEST_OPS each).
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

// ---------------------------------------------------------------------------
// Field sampling (Plan T11.3′, docs/gpu-prior-art.md §3.2, §5.2):
// classification, candidate gathering and sampling on the device, from the
// arrangement uploaded once (f32 copies the CPU computes) and the CPU's own
// cell grids. The evaluation mirrors `ses_kernel` / `sphere_kernel`
// (crates/sesmesh/src/field/kernel.rs) operation by operation. Decisions the CPU makes in
// f64 are made here in f32 with `slack` added on the inclusive side, so the
// lists are supersets of the CPU's and uniform sub-blocks a subset (see the
// review for why the mesh is unchanged). One workgroup per sub-block; a batch
// holds all 8 sub-blocks of each brick, sub-block j = 8 * brick + sb.
//
// Passes: count (cheap classification, list sizes), host scan (list bases),
// fill (lists, then the Lipschitz test at the centre for SES), sample.
// Record per sub-block (REC_WORDS): state, base, n_in, n_s, n_c, n_v, n_near
// (SES with p.near only: non-contained spheres within reach, the superset of
// the CPU's refinement blockers; measurements).
// States: 0 sample, 1 uniform outside, 2 uniform inside.

#define REC_WORDS 7
#define ST_SAMPLE 0u
#define ST_OUT 1u
#define ST_IN 2u
#define STATUS_EXPOSED 0u
#define STATUS_CONTAINED 2u
#define KIND_SES 0u

struct Grid {
    const unsigned* starts;  // dx * dy * dz + 1
    const unsigned* items;
    float ox, oy, oz, cell;  // frame-local origin, cell size
    unsigned dx, dy, dz, n_items;
};

struct ResidentParams {
    const float *px, *py, *pz, *pr, *pr2, *pr2tol, *pvdw;
    const unsigned* status;
    const unsigned *nb_off, *nb_items;
    const float *cx, *cy, *cz, *ux, *uy, *uz, *ex, *ey, *ez, *crho;
    const unsigned *cb_off, *cb_items;
    const float *vx, *vy, *vz;
    Grid gs, gc, gv;
    const int* bricks;  // 3 per brick: brick origin minus frame (lattice units)
    unsigned* rec;      // REC_WORDS per sub-block
    unsigned* ids;
    float* values;      // samples_per_brick per brick
    unsigned long long ids_len, values_len;
    unsigned n_sub, kind, n_spheres, n_circles, n_vertices, b, sub, near;  // near: SES near lists
    float probe, band, cutoff2, rho, reach, margin, slack, cutoffc2, h, max_expanded, rho_max,
        pad2;
};

#define LANES 64

__device__ static unsigned cell_coord(float p, float o, float cell, unsigned dim) {
    const float c = floorf((p - o) / cell);
    if (!(c > 0.0f)) return 0;
    if (c >= (float)(dim - 1)) return dim - 1;
    return (unsigned)c;
}

// Calls f(item) for the items of the cells overlapping the cube of half-size r
// around (x, y, z), lanes striding over each row's items.
template <typename F>
__device__ static void for_near(const Grid& g, float x, float y, float z, float r, F f) {
    const unsigned x0 = cell_coord(x - r, g.ox, g.cell, g.dx), x1 = cell_coord(x + r, g.ox, g.cell, g.dx);
    const unsigned y0 = cell_coord(y - r, g.oy, g.cell, g.dy), y1 = cell_coord(y + r, g.oy, g.cell, g.dy);
    const unsigned z0 = cell_coord(z - r, g.oz, g.cell, g.dz), z1 = cell_coord(z + r, g.oz, g.cell, g.dz);
    for (unsigned zc = z0; zc <= z1; zc++) {
        for (unsigned yc = y0; yc <= y1; yc++) {
            const unsigned row = (zc * g.dy + yc) * g.dx;
            const unsigned a = g.starts[row + x0], e = min(g.starts[row + x1 + 1], g.n_items);
            for (unsigned i = a + threadIdx.x; i < e; i += LANES) f(g.items[i]);
        }
    }
}

__device__ static void center_of(const ResidentParams& p, unsigned j, float& x, float& y, float& z) {
    const unsigned brick = j / 8, sb = j % 8;
    const float half = (float)p.sub * 0.5f;
    const float ox = (float)((sb & 1) * p.sub) + half, oy = (float)(((sb >> 1) & 1) * p.sub) + half,
                oz = (float)(((sb >> 2) & 1) * p.sub) + half;
    x = ((float)p.bricks[3 * brick] + ox) * p.h;
    y = ((float)p.bricks[3 * brick + 1] + oy) * p.h;
    z = ((float)p.bricks[3 * brick + 2] + oz) * p.h;
}

__device__ static float block_min(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (unsigned o = LANES / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] = fminf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// Circle distance as in gather_ses (f32).
__device__ static float circle_dist(const ResidentParams& p, unsigned ci, float x, float y, float z) {
    const float yx = x - p.cx[ci], yy = y - p.cy[ci], yz = z - p.cz[ci];
    const float along = yx * p.ux[ci] + yy * p.uy[ci] + yz * p.uz[ci];
    const float wx = yx - p.ux[ci] * along, wy = yy - p.uy[ci] * along, wz = yz - p.uz[ci] * along;
    const float wl = sqrtf(wx * wx + wy * wy + wz * wz);
    const float t = wl - p.crho[ci];
    return sqrtf(along * along + t * t);
}

// One scan of the three grids for sub-block j. mode 0: count, 1: fill.
__device__ static void scan(const ResidentParams& p, unsigned j, unsigned mode, unsigned* cnt,
                            unsigned* ids, unsigned base, const unsigned* n, float* s_sas,
                            float* s_vdw) {
    float x, y, z;
    center_of(p, j, x, y, z);
    const bool ses = p.kind == KIND_SES;
    float ls = INFINITY, lv = INFINITY;
    for_near(p.gs, x, y, z, p.reach + p.max_expanded + 2.0f * p.slack, [&](unsigned k) {
        if (k >= p.n_spheres || p.status[k] == STATUS_CONTAINED) return;
        const float dx = x - p.px[k], dy = y - p.py[k], dz = z - p.pz[k];
        const float d = sqrtf(dx * dx + dy * dy + dz * dz);
        const float r = p.pr[k];
        ls = fminf(ls, d - r);
        lv = fminf(lv, d - p.pvdw[k]);
        // Inside group (SES: can contain a sample; spheres: all near).
        if (d < r + (ses ? p.rho : p.reach) + p.slack) {
            const unsigned at = atomicAdd(&cnt[0], 1u);
            if (mode == 1 && at < n[0]) ids[base + at] = k;
        }
        if (ses && p.status[k] == STATUS_EXPOSED && fabsf(d - r) <= p.reach + p.slack) {
            const unsigned at = atomicAdd(&cnt[1], 1u);
            if (mode == 1 && at < n[1]) ids[base + n[0] + at] = k;
        }
        if (ses && p.near && d < r + p.reach + p.slack) {
            const unsigned at = atomicAdd(&cnt[4], 1u);
            if (mode == 1 && at < n[4]) ids[base + n[0] + n[1] + n[2] + n[3] + at] = k;
        }
    });
    if (s_sas) {
        __shared__ float red[LANES];
        *s_sas = block_min(ls, red);
        *s_vdw = block_min(lv, red);
    }
    if (!ses || mode == 2) return;
    if (mode == 0 && s_sas) {
        // Cheap uniform tests decide before circles and vertices are needed.
        if (*s_sas >= p.margin + p.slack || *s_vdw <= -(p.margin + p.slack)) return;
    }
    for_near(p.gc, x, y, z, p.reach + p.rho_max + 2.0f * p.slack, [&](unsigned ci) {
        if (ci >= p.n_circles) return;
        if (circle_dist(p, ci, x, y, z) <= p.reach + p.slack) {
            const unsigned at = atomicAdd(&cnt[2], 1u);
            if (mode == 1 && at < n[2]) ids[base + n[0] + n[1] + at] = ci;
        }
    });
    for_near(p.gv, x, y, z, p.reach + 2.0f * p.slack, [&](unsigned v) {
        if (v >= p.n_vertices) return;
        const float dx = x - p.vx[v], dy = y - p.vy[v], dz = z - p.vz[v];
        if (sqrtf(dx * dx + dy * dy + dz * dz) <= p.reach + p.slack) {
            const unsigned at = atomicAdd(&cnt[3], 1u);
            if (mode == 1 && at < n[3]) ids[base + n[0] + n[1] + n[2] + at] = v;
        }
    });
}

extern "C" __global__ void sesmesh_resident_count(ResidentParams p) {
    const unsigned j = blockIdx.x;
    if (j >= p.n_sub) return;
    __shared__ unsigned cnt[5];
    if (threadIdx.x < 5) cnt[threadIdx.x] = 0;
    __syncthreads();
    float s_sas, s_vdw;
    scan(p, j, 0, cnt, nullptr, 0, nullptr, &s_sas, &s_vdw);
    __syncthreads();
    if (threadIdx.x != 0) return;
    unsigned* r = p.rec + REC_WORDS * j;
    const float m = p.margin + p.slack;
    unsigned state = ST_SAMPLE;
    if (p.kind == KIND_SES) {
        if (s_sas >= m) state = ST_OUT;
        else if (s_vdw <= -m) state = ST_IN;
    } else if (s_sas > m) {
        state = ST_OUT;
    } else if (s_sas < -m) {
        state = ST_IN;
    }
    r[0] = state;
    r[1] = 0;
    for (int k = 0; k < 5; k++) r[2 + k] = state == ST_SAMPLE ? cnt[k] : 0;
}

// Validity of candidate point q against the blockers in rows [b0, b1).
__device__ static bool unblocked(const ResidentParams& p, const unsigned* items, unsigned b0,
                                 unsigned b1, float qx, float qy, float qz) {
    for (unsigned b = b0; b < b1; b++) {
        const unsigned k = items[b];
        if (k >= p.n_spheres) return false;  // invalid input only
        if (p.status[k] == STATUS_CONTAINED) continue;
        const float ex = qx - p.px[k], ey = qy - p.py[k], ez = qz - p.pz[k];
        if (!(ex * ex + ey * ey + ez * ez >= p.pr2tol[k])) return false;
    }
    return true;
}

// Clamped SES field at (x, y, z) from sub-block lists (one lane, the CPU
// kernel's operations in its order). Candidates [lane0, ..) step `step` let
// several lanes share one point (classification); the minimum is the same.
__device__ static float ses_best2(const ResidentParams& p, const unsigned* l, const unsigned* n,
                                  float x, float y, float z, float best2, unsigned first,
                                  unsigned step) {
    const unsigned *s = l + n[0], *c = s + n[1], *v = c + n[2];
    for (unsigned i = first; i < n[3]; i += step) {
        const unsigned k = v[i];
        if (k >= p.n_vertices) continue;
        const float dx = x - p.vx[k], dy = y - p.vy[k], dz = z - p.vz[k];
        best2 = fminf(best2, dx * dx + dy * dy + dz * dz);
    }
    for (unsigned i = first; i < n[1]; i += step) {
        const unsigned k = s[i];
        if (k >= p.n_spheres) continue;
        const float cx = p.px[k], cy = p.py[k], cz = p.pz[k], r = p.pr[k];
        const float dx = x - cx, dy = y - cy, dz = z - cz;
        const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        const float lb = fabsf(dist - r);
        const float lb2 = lb * lb;
        if (!(lb2 < best2 && 1e-6f < dist)) continue;
        const float scale = r / dist;
        const float qx = cx + dx * scale, qy = cy + dy * scale, qz = cz + dz * scale;
        if (unblocked(p, p.nb_items, p.nb_off[k], p.nb_off[k + 1], qx, qy, qz)) best2 = lb2;
    }
    for (unsigned i = first; i < n[2]; i += step) {
        const unsigned ci = c[i];
        if (ci >= p.n_circles) continue;
        const float mx = p.cx[ci], my = p.cy[ci], mz = p.cz[ci];
        const float ux = p.ux[ci], uy = p.uy[ci], uz = p.uz[ci], rho = p.crho[ci];
        const float yx = x - mx, yy = y - my, yz = z - mz;
        const float along = yx * ux + yy * uy + yz * uz;
        const float wx = yx - ux * along, wy = yy - uy * along, wz = yz - uz * along;
        const float wl = sqrtf(wx * wx + wy * wy + wz * wz);
        const float tt = wl - rho;
        const float d2 = along * along + tt * tt;
        if (!(d2 < best2)) continue;
        float qx, qy, qz;
        if (wl < 1e-6f) {
            qx = mx + p.ex[ci] * rho;
            qy = my + p.ey[ci] * rho;
            qz = mz + p.ez[ci] * rho;
        } else {
            const float sc = rho / wl;
            qx = mx + wx * sc;
            qy = my + wy * sc;
            qz = mz + wz * sc;
        }
        if (unblocked(p, p.cb_items, p.cb_off[ci], p.cb_off[ci + 1], qx, qy, qz)) best2 = d2;
    }
    return best2;
}

extern "C" __global__ void sesmesh_resident_fill(ResidentParams p) {
    const unsigned j = blockIdx.x;
    if (j >= p.n_sub) return;
    unsigned* r = p.rec + REC_WORDS * j;
    if (r[0] != ST_SAMPLE) return;
    const unsigned base = r[1];
    const unsigned n[5] = {r[2], r[3], r[4], r[5], r[6]};
    const unsigned long long total = (unsigned long long)n[0] + n[1] + n[2] + n[3] + n[4];
    if ((unsigned long long)base + total > p.ids_len) return;  // invalid input only
    __shared__ unsigned cnt[5];
    if (threadIdx.x < 5) cnt[threadIdx.x] = 0;
    __syncthreads();
    scan(p, j, 1, cnt, p.ids, base, n, nullptr, nullptr);
    __threadfence_block();
    __syncthreads();
    if (p.kind != KIND_SES) return;
    // Lipschitz test at the centre (Design §7.3), lanes sharing candidates.
    float x, y, z;
    center_of(p, j, x, y, z);
    const unsigned* l = p.ids + base;
    __shared__ unsigned inside;
    if (threadIdx.x == 0) inside = 0;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < n[0]; i += LANES) {
        const unsigned k = l[i];
        if (k >= p.n_spheres) continue;
        const float dx = x - p.px[k], dy = y - p.py[k], dz = z - p.pz[k];
        if (dx * dx + dy * dy + dz * dz < p.pr2[k]) atomicOr(&inside, 1u);
    }
    __syncthreads();
    float f = p.probe;
    if (inside) {
        __shared__ float red[LANES];
        const float b2 = block_min(ses_best2(p, l, n, x, y, z, p.cutoffc2, threadIdx.x, LANES), red);
        f = p.probe - sqrtf(b2);
    }
    if (threadIdx.x == 0) {
        const float m = p.margin + p.slack;
        if (f > m) r[0] = ST_OUT;
        else if (f < -m) r[0] = ST_IN;
    }
}

extern "C" __global__ void sesmesh_resident_sample(ResidentParams p) {
    const unsigned j = blockIdx.x;
    if (j >= p.n_sub) return;
    const unsigned brick = j / 8, sb = j % 8;
    unsigned state[8];
    for (int s = 0; s < 8; s++) state[s] = p.rec[REC_WORDS * (8 * brick + s)];
    const unsigned* r = p.rec + REC_WORDS * j;
    const unsigned base = r[1];
    const unsigned n[5] = {r[2], r[3], r[4], r[5], r[6]};
    const unsigned long long total = (unsigned long long)n[0] + n[1] + n[2] + n[3] + n[4];
    const bool lists_ok = (unsigned long long)base + total <= p.ids_len;
    const unsigned* l = p.ids + (lists_ok ? base : 0);
    const unsigned side = p.b + 1, box = p.sub + 1, sub = p.sub;
    const unsigned long long spb = (unsigned long long)side * side * side;
    if ((unsigned long long)(brick + 1) * spb > p.values_len) return;  // invalid input only
    float* out = p.values + brick * spb;
    const unsigned bx = (sb & 1) * sub, by = ((sb >> 1) & 1) * sub, bz = ((sb >> 2) & 1) * sub;
    const int ix = p.bricks[3 * brick], iy = p.bricks[3 * brick + 1], iz = p.bricks[3 * brick + 2];
    for (unsigned t = threadIdx.x; t < box * box * box; t += blockDim.x) {
        const unsigned lx = bx + t % box, ly = by + (t / box) % box, lz = bz + t / (box * box);
        // Owner: the first non-uniform sub-block containing the sample
        // (Engine::assign).
        unsigned lo[3], hi[3];
        const unsigned v[3] = {lx, ly, lz};
        for (int a = 0; a < 3; a++) {
            hi[a] = min(v[a] / sub, 1u);
            lo[a] = (v[a] > 0 && v[a] % sub == 0) ? min(v[a] / sub - 1, 1u) : hi[a];
        }
        int owner = -1;
        bool positive = true;
        for (unsigned sz = lo[2]; sz <= hi[2] && owner < 0; sz++)
            for (unsigned sy = lo[1]; sy <= hi[1] && owner < 0; sy++)
                for (unsigned sx = lo[0]; sx <= hi[0]; sx++) {
                    const unsigned s = sx + 2 * sy + 4 * sz;
                    if (state[s] == ST_SAMPLE) {
                        owner = (int)s;
                        break;
                    }
                    positive = state[s] == ST_OUT;
                }
        const unsigned i = lx + side * (ly + side * lz);
        if (owner < 0) {
            if (sb == lo[0] + 2 * lo[1] + 4 * lo[2]) out[i] = positive ? p.band : -p.band;
            continue;
        }
        if ((unsigned)owner != sb || !lists_ok) continue;
        const float x = (float)(ix + (int)lx) * p.h, y = (float)(iy + (int)ly) * p.h,
                    z = (float)(iz + (int)lz) * p.h;
        if (p.kind == KIND_SES) {
            bool inside = false;
            for (unsigned q = 0; q < n[0]; q++) {
                const unsigned k = l[q];
                if (k >= p.n_spheres) continue;
                const float dx = x - p.px[k], dy = y - p.py[k], dz = z - p.pz[k];
                if (dx * dx + dy * dy + dz * dz < p.pr2[k]) {
                    inside = true;
                    break;
                }
            }
            out[i] = inside ? p.probe - sqrtf(ses_best2(p, l, n, x, y, z, p.cutoff2, 0, 1))
                            : p.probe;
        } else {
            float best = p.band;
            for (unsigned q = 0; q < n[0]; q++) {
                const unsigned k = l[q];
                if (k >= p.n_spheres) continue;
                const float dx = x - p.px[k], dy = y - p.py[k], dz = z - p.pz[k];
                best = fminf(best, sqrtf(dx * dx + dy * dy + dz * dz) - p.pr[k]);
            }
            out[i] = best;
        }
    }
}
