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

// ---------------------------------------------------------------------------
// Vertex refinement in f32 (spike S2, docs/gpu-prior-art.md §3.4): the
// safeguarded Newton / Illinois iteration and bisection of `vertex()`
// (crates/sesmesh/src/mesh/extract.rs) on crossing edges, against the lists
// of the edge's sub-block. One workgroup per group of edges of one
// (brick, sub-block), one lane per edge.
//
// Coordinates are local to the brick origin O. A candidate's offset c − O is
// formed from double-float pairs: its frame-local f32 position plus the f32
// residual of that rounding, and O·h as an exact f32 product plus its error
// (fmaf) plus O·h_lo. So the f32 rounding of frame-local coordinates (up to
// 1.5e-5 Å at 500 Å) does not enter; local values stay below ~10 Å.
//
// Outputs per edge: t (the last evaluated parameter, unclamped), the feature
// of that evaluation (kind << 29 | id), its closest point (local), and the
// evaluation count with bit 31 set when Newton / Illinois converged.

#define FEAT_SOLVENT 0u
#define FEAT_BEYOND 1u
#define FEAT_SPHERE 2u
#define FEAT_CIRCLE 3u
#define FEAT_VERTEX 4u
#define FEAT_SHIFT 29

struct RefineParams {
    const float *plx, *ply, *plz;  // sphere centre residuals
    const float *clx, *cly, *clz;  // circle centre residuals
    const float *vlx, *vly, *vlz;  // vertex residuals
    unsigned* edge_count;          // count pass: crossing edges per sub-block
    const unsigned* edge_off;      // refine pass: first edge per sub-block (n_sub + 1)
    float* t_out;
    unsigned* feat_out;
    float* q_out;                  // 3 per edge
    unsigned* evals_out;
    unsigned n_edges, max_iter, max_bisect, pad0;
    float h_hi, h_lo, tol, pad;
};

// O·h as a double-float pair (hi, lo).
__device__ static void origin_df(int o, float h_hi, float h_lo, float& hi, float& lo) {
    const float of = (float)o;
    hi = of * h_hi;
    lo = fmaf(of, h_hi, -hi) + of * h_lo;
}

struct Local {
    float ohx, olx, ohy, oly, ohz, olz;  // origin pairs (hi, lo) per axis
};

__device__ static float loc(float hi, float lo, float ohi, float olo) {
    return (hi - ohi) + (lo - olo);
}

struct PointEval {
    float value;
    unsigned feat;
    float qx, qy, qz;
};

__device__ static bool unblocked_local(const ResidentParams& p, const RefineParams& r,
                                       const Local& o, const unsigned* items, unsigned b0,
                                       unsigned b1, float qx, float qy, float qz) {
    for (unsigned b = b0; b < b1; b++) {
        const unsigned k = items[b];
        if (k >= p.n_spheres) return false;  // invalid input only
        if (p.status[k] == STATUS_CONTAINED) continue;
        const float ex = qx - loc(p.px[k], r.plx[k], o.ohx, o.olx);
        const float ey = qy - loc(p.py[k], r.ply[k], o.ohy, o.oly);
        const float ez = qz - loc(p.pz[k], r.plz[k], o.ohz, o.olz);
        if (!(ex * ex + ey * ey + ez * ez >= p.pr2tol[k])) return false;
    }
    return true;
}

// eval_local in f32, local coordinates: value, feature and closest point.
__device__ static PointEval eval_point(const ResidentParams& p, const RefineParams& r,
                                       const Local& o, const unsigned* l, const unsigned* n,
                                       float x, float y, float z) {
    PointEval e;
    if (p.kind != KIND_SES) {
        e.value = p.band;
        e.feat = FEAT_BEYOND << FEAT_SHIFT;
        for (unsigned q = 0; q < n[0]; q++) {
            const unsigned k = l[q];
            if (k >= p.n_spheres) continue;
            const float dx = x - loc(p.px[k], r.plx[k], o.ohx, o.olx);
            const float dy = y - loc(p.py[k], r.ply[k], o.ohy, o.oly);
            const float dz = z - loc(p.pz[k], r.plz[k], o.ohz, o.olz);
            const float v = sqrtf(dx * dx + dy * dy + dz * dz) - p.pr[k];
            if (v < e.value) {
                e.value = v;
                e.feat = (FEAT_SPHERE << FEAT_SHIFT) | k;
            }
        }
        e.qx = x;
        e.qy = y;
        e.qz = z;
        return e;
    }
    bool inside = false;
    for (unsigned q = 0; q < n[0] && !inside; q++) {
        const unsigned k = l[q];
        if (k >= p.n_spheres) continue;
        const float dx = x - loc(p.px[k], r.plx[k], o.ohx, o.olx);
        const float dy = y - loc(p.py[k], r.ply[k], o.ohy, o.oly);
        const float dz = z - loc(p.pz[k], r.plz[k], o.ohz, o.olz);
        inside = sqrtf(dx * dx + dy * dy + dz * dz) < p.pr[k];
    }
    e.qx = x;
    e.qy = y;
    e.qz = z;
    if (!inside) {
        e.value = p.probe;
        e.feat = FEAT_SOLVENT << FEAT_SHIFT;
        return e;
    }
    float best = p.probe + p.band;
    e.feat = FEAT_BEYOND << FEAT_SHIFT;
    const unsigned *s = l + n[0], *c = s + n[1], *v = c + n[2];
    for (unsigned i = 0; i < n[1]; i++) {
        const unsigned k = s[i];
        if (k >= p.n_spheres) continue;
        const float cx = loc(p.px[k], r.plx[k], o.ohx, o.olx);
        const float cy = loc(p.py[k], r.ply[k], o.ohy, o.oly);
        const float cz = loc(p.pz[k], r.plz[k], o.ohz, o.olz);
        const float dx = x - cx, dy = y - cy, dz = z - cz;
        const float len = sqrtf(dx * dx + dy * dy + dz * dz);
        const float lb = fabsf(len - p.pr[k]);
        if (lb >= best || len < 1e-12f) continue;
        const float sc = p.pr[k] / len;
        const float qx = cx + dx * sc, qy = cy + dy * sc, qz = cz + dz * sc;
        if (unblocked_local(p, r, o, p.nb_items, p.nb_off[k], p.nb_off[k + 1], qx, qy, qz)) {
            best = lb;
            e.feat = (FEAT_SPHERE << FEAT_SHIFT) | k;
            e.qx = qx;
            e.qy = qy;
            e.qz = qz;
        }
    }
    for (unsigned i = 0; i < n[2]; i++) {
        const unsigned ci = c[i];
        if (ci >= p.n_circles) continue;
        const float mx = loc(p.cx[ci], r.clx[ci], o.ohx, o.olx);
        const float my = loc(p.cy[ci], r.cly[ci], o.ohy, o.oly);
        const float mz = loc(p.cz[ci], r.clz[ci], o.ohz, o.olz);
        const float ux = p.ux[ci], uy = p.uy[ci], uz = p.uz[ci], rho = p.crho[ci];
        const float yx = x - mx, yy = y - my, yz = z - mz;
        const float along = yx * ux + yy * uy + yz * uz;
        const float wx = yx - ux * along, wy = yy - uy * along, wz = yz - uz * along;
        const float wl = sqrtf(wx * wx + wy * wy + wz * wz);
        const float tt = wl - rho;
        const float dist = sqrtf(along * along + tt * tt);
        if (dist >= best) continue;
        float qx, qy, qz;
        if (wl > 1e-12f) {
            qx = mx + (wx / wl) * rho;
            qy = my + (wy / wl) * rho;
            qz = mz + (wz / wl) * rho;
        } else {
            qx = mx + p.ex[ci] * rho;
            qy = my + p.ey[ci] * rho;
            qz = mz + p.ez[ci] * rho;
        }
        if (unblocked_local(p, r, o, p.cb_items, p.cb_off[ci], p.cb_off[ci + 1], qx, qy, qz)) {
            best = dist;
            e.feat = (FEAT_CIRCLE << FEAT_SHIFT) | ci;
            e.qx = qx;
            e.qy = qy;
            e.qz = qz;
        }
    }
    for (unsigned i = 0; i < n[3]; i++) {
        const unsigned k = v[i];
        if (k >= p.n_vertices) continue;
        const float px = loc(p.vx[k], r.vlx[k], o.ohx, o.olx);
        const float py = loc(p.vy[k], r.vly[k], o.ohy, o.oly);
        const float pz = loc(p.vz[k], r.vlz[k], o.ohz, o.olz);
        const float dx = x - px, dy = y - py, dz = z - pz;
        const float d = sqrtf(dx * dx + dy * dy + dz * dz);
        if (d < best) {
            best = d;
            e.feat = (FEAT_VERTEX << FEAT_SHIFT) | k;
            e.qx = px;
            e.qy = py;
            e.qz = pz;
        }
    }
    e.value = p.probe - best;
    return e;
}

// Derivative of the field along the edge (direction `axis`, length h), from
// the evaluation's exact gradient (`gradient()` in extract.rs); 0 without one.
__device__ static float slope_of(const ResidentParams& p, const RefineParams& r, const Local& o,
                                 const PointEval& e, float x, float y, float z, unsigned axis) {
    const unsigned kind = e.feat >> FEAT_SHIFT;
    if (kind == FEAT_SOLVENT || kind == FEAT_BEYOND) return 0.0f;
    float wx, wy, wz;
    if (p.kind == KIND_SES) {
        wx = e.qx - x;
        wy = e.qy - y;
        wz = e.qz - z;
    } else {
        const unsigned k = e.feat & ((1u << FEAT_SHIFT) - 1);
        wx = x - loc(p.px[k], r.plx[k], o.ohx, o.olx);
        wy = y - loc(p.py[k], r.ply[k], o.ohy, o.oly);
        wz = z - loc(p.pz[k], r.plz[k], o.ohz, o.olz);
    }
    const float len = sqrtf(wx * wx + wy * wy + wz * wz);
    if (!(len > 1e-12f)) return 0.0f;
    const float w = axis == 0 ? wx : (axis == 1 ? wy : wz);
    return (w / len) * p.h;
}

// Crossing-edge candidate `c` of sub-block `sb`: the edges whose lower
// sample lies in it (`extract.rs`: sub-block `(v / sub).min(1)` per axis), in
// the CPU's z, y, x, axis order. Writes its lower sample and axis; true if
// the samples at its ends differ in sign.
__device__ static bool crossing_edge(const ResidentParams& p, const float* val, unsigned sb,
                                     unsigned c, unsigned& x, unsigned& y, unsigned& z,
                                     unsigned& axis) {
    const unsigned sub = p.sub, side = p.b + 1;
    axis = c % 3u;
    const unsigned cell = c / 3u;
    x = (sb & 1u) * sub + cell % sub;
    y = ((sb >> 1) & 1u) * sub + (cell / sub) % sub;
    z = (sb >> 2) * sub + cell / (sub * sub);
    const unsigned i0 = x + side * (y + side * z);
    const unsigned st = axis == 0 ? 1u : (axis == 1 ? side : side * side);
    return (val[i0] < 0.0f) != (val[i0 + st] < 0.0f);
}

// Whether sub-block j of the batch is sampled and its brick's values are in
// bounds (invalid input only otherwise).
__device__ static bool sampled_sub(const ResidentParams& p, unsigned j) {
    if (j >= p.n_sub || 2u * p.sub != p.b || p.b > 32u) return false;
    if (p.rec[REC_WORDS * j] != ST_SAMPLE) return false;
    const unsigned long long side = p.b + 1;
    return (unsigned long long)(j / 8 + 1) * side * side * side <= p.values_len;
}

// Count pass: crossing edges per sub-block (0 for uniform sub-blocks, which
// have none). One workgroup per sub-block.
extern "C" __global__ void sesmesh_resident_edge_count(ResidentParams p, RefineParams r) {
    const unsigned j = blockIdx.x;
    if (j >= p.n_sub) return;
    __shared__ unsigned count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    unsigned mine = 0;
    if (sampled_sub(p, j)) {
        const unsigned side = p.b + 1;
        const float* val = p.values + (unsigned long long)(j / 8) * side * side * side;
        const unsigned nc = 3u * p.sub * p.sub * p.sub;
        unsigned x, y, z, axis;
        for (unsigned c = threadIdx.x; c < nc; c += blockDim.x)
            mine += crossing_edge(p, val, j % 8, c, x, y, z, axis) ? 1u : 0u;
    }
    atomicAdd(&count, mine);
    __syncthreads();
    if (threadIdx.x == 0) r.edge_count[j] = count;
}

// Refines one crossing edge (lower sample `code`, packed x | y << 5 |
// z << 10 | axis << 15) into output slot `id`.
__device__ static void refine_edge(const ResidentParams& p, const RefineParams& r,
                                   const Local& o, const unsigned* l, const unsigned* n,
                                   const float* val, unsigned code, unsigned id) {
    const unsigned side = p.b + 1;
    const unsigned ex = code & 31u, ey = (code >> 5) & 31u, ez = (code >> 10) & 31u,
                   axis = (code >> 15) & 3u;
    const unsigned i0 = ex + side * (ey + side * ez);
    const unsigned st = axis == 0 ? 1u : (axis == 1 ? side : side * side);
    const float h = r.h_hi;
    const float fa = val[i0], fb = val[i0 + st];
    // Lower end, local: integer offsets times h (double-float h).
    const float ax = (float)ex * h + (float)ex * r.h_lo;
    const float ay = (float)ey * h + (float)ey * r.h_lo;
    const float az = (float)ez * h + (float)ez * r.h_lo;
    const float hd = h + r.h_lo;  // edge length
    auto at = [&](float t, float& x, float& y, float& z) {
        x = ax + (axis == 0 ? hd * t : 0.0f);
        y = ay + (axis == 1 ? hd * t : 0.0f);
        z = az + (axis == 2 ? hd * t : 0.0f);
    };
    float t = fminf(fmaxf(fa / (fa - fb), 0.0f), 1.0f);
    float t0 = 0.0f, g0 = fa, t1 = 1.0f, g1 = fb;
    int side_ = 0;
    bool converged = false;
    unsigned evals = 0;
    PointEval last;
    last.feat = FEAT_BEYOND << FEAT_SHIFT;
    last.qx = last.qy = last.qz = 0.0f;
    last.value = 0.0f;
    float x, y, z;
    for (unsigned it = 0; it < r.max_iter; it++) {
        at(t, x, y, z);
        const PointEval e = eval_point(p, r, o, l, n, x, y, z);
        evals++;
        last = e;
        const float gv = e.value;
        if (fabsf(gv) <= r.tol) {
            converged = true;
            break;
        }
        if ((gv < 0.0f) == (g0 < 0.0f)) {
            t0 = t;
            g0 = gv;
            if (side_ == -1) g1 *= 0.5f;
            side_ = -1;
        } else {
            t1 = t;
            g1 = gv;
            if (side_ == 1) g0 *= 0.5f;
            side_ = 1;
        }
        const float slope = slope_of(p, r, o, e, x, y, z, axis);
        const float newton = t - gv / slope;
        if (fabsf(slope) > 1e-3f * hd && newton > t0 && newton < t1) {
            t = newton;
        } else {
            const float s = (t0 * g1 - t1 * g0) / (g1 - g0);
            t = (isfinite(s) && s > t0 && s < t1) ? s : 0.5f * (t0 + t1);
        }
    }
    if (!converged) {
        const bool neg0 = g0 < 0.0f;
        for (unsigned it = 0; it < r.max_bisect; it++) {
            t = 0.5f * (t0 + t1);
            at(t, x, y, z);
            const PointEval e = eval_point(p, r, o, l, n, x, y, z);
            evals++;
            last = e;
            if (fabsf(e.value) <= r.tol || (t1 - t0) * hd <= 2.0f * r.tol) break;
            if ((e.value < 0.0f) == neg0) t0 = t;
            else t1 = t;
        }
    }
    r.t_out[id] = t;
    r.feat_out[id] = last.feat;
    r.q_out[3 * id] = last.qx;
    r.q_out[3 * id + 1] = last.qy;
    r.q_out[3 * id + 2] = last.qz;
    r.evals_out[id] = evals | (converged ? 0x80000000u : 0u);
}

// Refine pass: one workgroup (64 lanes) per sub-block. The lanes list the
// sub-block's crossing edges in the CPU's order (a scan in LDS per round of
// 64 candidates, into a buffer of up to 127), and refine 64 at a time, so
// lanes stay busy; edge k of the sub-block goes to slot edge_off[j] + k.
extern "C" __global__ void sesmesh_resident_refine(ResidentParams p, RefineParams r) {
    const unsigned j = blockIdx.x;
    if (j >= p.n_sub || blockDim.x != 64) return;
    const unsigned first = r.edge_off[j], last = r.edge_off[j + 1];
    if (last <= first || last > r.n_edges || !sampled_sub(p, j)) return;
    const unsigned* rec = p.rec + REC_WORDS * j;
    const unsigned base = rec[1];
    const unsigned n[5] = {rec[2], rec[3], rec[4], rec[5], rec[6]};
    const unsigned long long total = (unsigned long long)n[0] + n[1] + n[2] + n[3] + n[4];
    if ((unsigned long long)base + total > p.ids_len) return;
    const unsigned* l = p.ids + base;
    const unsigned brick = j / 8, sb = j % 8;
    const unsigned side = p.b + 1;
    const float* val = p.values + (unsigned long long)brick * side * side * side;
    Local o;
    origin_df(p.bricks[3 * brick], r.h_hi, r.h_lo, o.ohx, o.olx);
    origin_df(p.bricks[3 * brick + 1], r.h_hi, r.h_lo, o.ohy, o.oly);
    origin_df(p.bricks[3 * brick + 2], r.h_hi, r.h_lo, o.ohz, o.olz);

    __shared__ unsigned scan[64];
    __shared__ unsigned buf[128];
    const unsigned lane = threadIdx.x;
    const unsigned nc = 3u * p.sub * p.sub * p.sub;
    unsigned fill = 0, done = 0;
    for (unsigned c0 = 0; c0 < nc; c0 += 64) {
        unsigned x = 0, y = 0, z = 0, axis = 0;
        const unsigned c = c0 + lane;
        const bool f = c < nc && crossing_edge(p, val, sb, c, x, y, z, axis);
        scan[lane] = f ? 1u : 0u;
        __syncthreads();
        for (unsigned off = 1; off < 64; off <<= 1) {
            const unsigned add = lane >= off ? scan[lane - off] : 0u;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        const unsigned added = scan[63];
        if (f) buf[fill + scan[lane] - 1] = x | y << 5 | z << 10 | axis << 15;
        __syncthreads();
        fill += added;
        if (fill >= 64) {
            const unsigned id = first + done + lane;
            if (id < last) refine_edge(p, r, o, l, n, val, buf[lane], id);
            done += 64;
            const unsigned rest = fill - 64;
            const unsigned moved = lane < rest ? buf[64 + lane] : 0u;
            __syncthreads();
            if (lane < rest) buf[lane] = moved;
            __syncthreads();
            fill = rest;
        }
    }
    const unsigned id = first + done + lane;
    if (lane < fill && id < last) refine_edge(p, r, o, l, n, val, buf[lane], id);
}
