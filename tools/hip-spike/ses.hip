// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// Port of `field::kernel::ses_kernel` (Design §7.5) for the HIP spike: one
// workgroup per sub-block, one lane per sample, per-lane control flow (the
// hardware skips a branch when no lane of the wave takes it, which is the
// CPU kernel's "skip if no lane improves"). Same operation order as the
// Rust kernel; compiled with -ffp-contract=off.

#include <hip/hip_runtime.h>

struct Arrays {
    const unsigned* desc;  // 10 words per sub-block
    const float *xs, *ys, *zs;
    const float *nx, *ny, *nz, *nr2, *nr2tol;
    const float *sx, *sy, *sz, *sr;
    const unsigned *s_start, *s_blk;
    const float *cx, *cy, *cz, *ux, *uy, *uz, *ex, *ey, *ez, *rho;
    const unsigned *c_start, *c_blk;
    const float *vx, *vy, *vz;
    float* out;
    float probe, cutoff2;
};

template <typename R>
__device__ void ses_body(const Arrays& a) {
    const unsigned* d = a.desc + 10 * blockIdx.x;
    const unsigned samp0 = d[0], m = d[1], near0 = d[2], nin = d[3], s0 = d[4], ns = d[5],
                   c0 = d[6], nc = d[7], v0 = d[8], nv = d[9];
    for (unsigned t = threadIdx.x; t < m; t += blockDim.x) {
        const unsigned i = samp0 + t;
        const R x = a.xs[i], y = a.ys[i], z = a.zs[i];
        bool inside = false;
        for (unsigned k = near0; k < near0 + nin; k++) {
            R dx = x - a.nx[k], dy = y - a.ny[k], dz = z - a.nz[k];
            if (dx * dx + dy * dy + dz * dz < a.nr2[k]) { inside = true; break; }
        }
        if (!inside) { a.out[i] = a.probe; continue; }
        R best2 = (R)a.cutoff2;
        for (unsigned v = v0; v < v0 + nv; v++) {
            R dx = x - a.vx[v], dy = y - a.vy[v], dz = z - a.vz[v];
            best2 = fmin(best2, dx * dx + dy * dy + dz * dz);
        }
        for (unsigned si = s0; si < s0 + ns; si++) {
            const R cx = a.sx[si], cy = a.sy[si], cz = a.sz[si], r = a.sr[si];
            R dx = x - cx, dy = y - cy, dz = z - cz;
            R dist = sqrt(dx * dx + dy * dy + dz * dz);
            R lb = fabs(dist - r);
            R lb2 = lb * lb;
            bool valid = lb2 < best2 && (R)1e-6f < dist;
            if (!valid) continue;
            R scale = r / dist;
            R qx = cx + dx * scale, qy = cy + dy * scale, qz = cz + dz * scale;
            for (unsigned b = a.s_start[si]; b < a.s_start[si + 1]; b++) {
                unsigned k = a.s_blk[b];
                R ex = qx - a.nx[k], ey = qy - a.ny[k], ez = qz - a.nz[k];
                if (!(ex * ex + ey * ey + ez * ez >= a.nr2tol[k])) { valid = false; break; }
            }
            if (valid) best2 = lb2;
        }
        for (unsigned ci = c0; ci < c0 + nc; ci++) {
            const R mx = a.cx[ci], my = a.cy[ci], mz = a.cz[ci];
            const R ux = a.ux[ci], uy = a.uy[ci], uz = a.uz[ci], rho = a.rho[ci];
            R yx = x - mx, yy = y - my, yz = z - mz;
            R along = yx * ux + yy * uy + yz * uz;
            R wx = yx - ux * along, wy = yy - uy * along, wz = yz - uz * along;
            R wl = sqrt(wx * wx + wy * wy + wz * wz);
            R t2 = wl - rho;
            R d2 = along * along + t2 * t2;
            if (!(d2 < best2)) continue;
            bool valid = true;
            R qx, qy, qz;
            if (wl < (R)1e-6f) {
                qx = mx + a.ex[ci] * rho; qy = my + a.ey[ci] * rho; qz = mz + a.ez[ci] * rho;
            } else {
                R s = rho / wl;
                qx = mx + wx * s; qy = my + wy * s; qz = mz + wz * s;
            }
            for (unsigned b = a.c_start[ci]; b < a.c_start[ci + 1]; b++) {
                unsigned k = a.c_blk[b];
                R ex = qx - a.nx[k], ey = qy - a.ny[k], ez = qz - a.nz[k];
                if (!(ex * ex + ey * ey + ez * ez >= a.nr2tol[k])) { valid = false; break; }
            }
            if (valid) best2 = d2;
        }
        a.out[i] = (float)((R)a.probe - sqrt(best2));
    }
}

extern "C" __global__ void ses_kernel(Arrays a) { ses_body<float>(a); }
extern "C" __global__ void ses_kernel_f64(Arrays a) { ses_body<double>(a); }
