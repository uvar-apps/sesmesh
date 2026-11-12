//! Sampling lattice, bricks and the conservative brick classification
//! (Design §7.1–7.3).

use super::SurfaceKind;
use super::point::SesField;
use crate::error::{Error, Result};
use crate::geom::{Status, Vec3};
use crate::par;

/// Bricks span at most ±2^20 per axis (packed into a `u64` key).
const KEY_BIAS: i64 = 1 << 20;

/// Safety margin (Å) of the classification tests; larger than any rounding
/// error of the `f32` kernel, so samples of uniform bricks always get the
/// right sign (Design §7.3).
pub(crate) const CLASSIFY_EPS: f64 = 2e-3;

/// The global lattice `h · (i, j, k)` (anchored at the input-frame origin)
/// partitioned into bricks of `b³` cells. A brick with key `(bx, by, bz)`
/// owns sample indices `[bx·b, bx·b + b]` per axis (its "+" faces included).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lattice {
    pub(crate) h: f64,
    pub(crate) b: i64,
    /// Lattice index of the `f32` frame origin.
    frame: [i64; 3],
    h32: f32,
}

impl Lattice {
    pub(crate) fn new(h: f64, b: usize, frame_center: Vec3) -> Self {
        let frame =
            [frame_center.x, frame_center.y, frame_center.z].map(|v| (v / h).round() as i64);
        Self {
            h,
            b: b as i64,
            frame,
            h32: h as f32,
        }
    }

    /// Global position of a lattice sample.
    pub(crate) fn point(&self, idx: [i64; 3]) -> Vec3 {
        Vec3::new(
            idx[0] as f64 * self.h,
            idx[1] as f64 * self.h,
            idx[2] as f64 * self.h,
        )
    }

    /// Origin of the `f32` frame (a lattice point).
    pub(crate) fn frame_origin(&self) -> Vec3 {
        self.point(self.frame)
    }

    /// Frame-local `f32` coordinates of a sample. Computed from integers only,
    /// so every brick sharing a sample gets bit-identical coordinates
    /// (Design §7.5, P7).
    pub(crate) fn local(&self, idx: [i64; 3]) -> [f32; 3] {
        [0, 1, 2].map(|a| (idx[a] - self.frame[a]) as f32 * self.h32)
    }

    pub(crate) fn brick_center(&self, key: [i32; 3]) -> Vec3 {
        let c = |k: i32| (f64::from(k) + 0.5) * self.b as f64 * self.h;
        Vec3::new(c(key[0]), c(key[1]), c(key[2]))
    }

    /// Circumradius of a brick.
    pub(crate) fn brick_radius(&self) -> f64 {
        0.5 * 3f64.sqrt() * self.b as f64 * self.h
    }

    /// First sample index of a brick per axis.
    pub(crate) fn brick_origin(&self, key: [i32; 3]) -> [i64; 3] {
        key.map(|k| i64::from(k) * self.b)
    }

    fn brick_of(&self, v: f64) -> i64 {
        (v / (self.b as f64 * self.h)).floor() as i64
    }
}

pub(crate) fn pack(key: [i32; 3]) -> u64 {
    let k = key.map(|v| (i64::from(v) + KEY_BIAS) as u64);
    k[0] | (k[1] << 21) | (k[2] << 42)
}

pub(crate) fn unpack(packed: u64) -> [i32; 3] {
    [0, 21, 42].map(|s| (((packed >> s) & 0x1f_ffff) as i64 - KEY_BIAS) as i32)
}

/// Bricks overlapping any (non-contained) sphere's surface-carrying ball:
/// expanded radius for SES/SAS, vdW radius for vdW. All other bricks lie in
/// the solvent and are uniformly outside. Sorted by packed key.
pub(crate) fn touched_bricks(field: &SesField, lattice: &Lattice) -> Result<Vec<[i32; 3]>> {
    let a = &field.arr;
    let n = a.centers.len();
    let per_sphere: Vec<Result<Vec<u64>>> = par::map(n, |i| {
        if a.status[i] == Status::Contained {
            return Ok(Vec::new());
        }
        let r = a.expanded[i] + CLASSIFY_EPS;
        let c = a.centers[i];
        let lo = [c.x - r, c.y - r, c.z - r].map(|v| lattice.brick_of(v));
        let hi = [c.x + r, c.y + r, c.z + r].map(|v| lattice.brick_of(v));
        if lo.iter().chain(&hi).any(|k| k.abs() >= KEY_BIAS - 1) {
            return Err(Error::InvalidInput(format!(
                "coordinates too far from the origin for grid spacing {} Å",
                lattice.h
            )));
        }
        let mut keys = Vec::new();
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    keys.push(pack([x as i32, y as i32, z as i32]));
                }
            }
        }
        Ok(keys)
    });
    let mut keys = Vec::new();
    for part in per_sphere {
        keys.extend(part?);
    }
    par::sort_unstable(&mut keys);
    keys.dedup();
    Ok(keys.into_iter().map(unpack).collect())
}

/// Keeps the bricks the surface can pass through, using provably
/// conservative tests (Design §7.3): cheap SAS/vdW bounds, then the
/// Lipschitz test on the exact field at the brick centre.
pub(crate) fn surface_bricks(
    field: &SesField,
    lattice: &Lattice,
    touched: &[[i32; 3]],
) -> Vec<[i32; 3]> {
    let rho = lattice.brick_radius();
    let keep: Vec<bool> = par::map(touched.len(), |t| {
        uniform_sign(field, lattice.brick_center(touched[t]), rho).is_none()
    });
    touched
        .iter()
        .zip(keep)
        .filter_map(|(k, keep)| keep.then_some(*k))
        .collect()
}

/// Sign of the field on the whole ball of radius `rho` around `center`, if
/// provably uniform (`Some(true)`: outside, `Some(false)`: inside), else
/// `None`. Conservative tests of Design §7.3: cheap SAS/vdW bounds, then the
/// Lipschitz test on the exact field at the centre, all with margin
/// [`CLASSIFY_EPS`].
pub(crate) fn uniform_sign(field: &SesField, center: Vec3, rho: f64) -> Option<bool> {
    let a = &field.arr;
    let margin = rho + CLASSIFY_EPS;
    // s_sas: min(|b − c| − R); s_vdw: min(|b − c| − r) over nearby spheres.
    let (mut s_sas, mut s_vdw) = (f64::INFINITY, f64::INFINITY);
    a.cells.for_each_near(center, margin + a.max_expanded, |k| {
        let k = k as usize;
        if a.status[k] != Status::Contained {
            let d = center.dist(a.centers[k]);
            s_sas = s_sas.min(d - a.expanded[k]);
            s_vdw = s_vdw.min(d - a.radii[k]);
        }
    });
    let sign = |f: f64| {
        if f > margin {
            Some(true)
        } else if f < -margin {
            Some(false)
        } else {
            None
        }
    };
    match field.kind {
        // `expanded` is the SAS radius for SAS and the vdW radius for the
        // vdW field (built with probe 0).
        SurfaceKind::Sas | SurfaceKind::Vdw => sign(s_sas),
        SurfaceKind::Ses => {
            if s_sas >= margin {
                return Some(true); // inside P: solvent
            }
            if s_vdw <= -margin {
                return Some(false); // deep inside an atom
            }
            let cutoff = field.probe + margin + CLASSIFY_EPS;
            sign(field.probe - field.ses_eval(center, cutoff).distance)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip() {
        for key in [[0, 0, 0], [-1, 2, -3], [1_000_000, -1_000_000, 17]] {
            assert_eq!(unpack(pack(key)), key);
        }
        let mut sorted = [pack([1, 0, 0]), pack([0, 1, 0]), pack([0, 0, 1])];
        sorted.sort_unstable();
        assert_eq!(unpack(sorted[0]), [1, 0, 0], "x varies fastest");
    }

    #[test]
    fn lattice_coordinates() {
        let l = Lattice::new(0.5, 8, Vec3::new(10.2, -3.9, 0.0));
        assert_eq!(l.frame_origin(), Vec3::new(10.0, -4.0, 0.0));
        assert_eq!(l.local([20, -8, 0]), [0.0, 0.0, 0.0]);
        assert_eq!(l.local([21, -8, 1]), [0.5, 0.0, 0.5]);
        assert_eq!(l.brick_origin([1, -1, 0]), [8, -8, 0]);
        assert_eq!(l.brick_center([0, 0, 0]), Vec3::new(2.0, 2.0, 2.0));
    }
}
