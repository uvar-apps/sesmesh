//! Reference SES area and volume for one and two spheres (Design §15).
//!
//! Two spheres give an axially symmetric SES: two spherical zones joined by
//! the arc of the probe circle swept around the axis (a toroidal saddle). In
//! the meridian plane the probe centre sits at `(x_p, y_p)`; arc points are
//! `(x_p + r_p sin φ, y_p − r_p cos φ)` between the two contact angles. When
//! `y_p < r_p` the arc crosses the axis (self-intersecting torus) and the part
//! below the axis is clipped: the SES then splits into two components meeting
//! at cusps. Areas and volumes of revolution are integrated with composite
//! Gauss–Legendre quadrature (machine precision for these smooth integrands),
//! split at the clipping points.

use std::f64::consts::PI;

/// Area and volume of a surface of revolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaVolume {
    /// Surface area, Å².
    pub area: f64,
    /// Enclosed volume, Å³.
    pub volume: f64,
    /// Number of closed components.
    pub components: usize,
}

/// SES of a single sphere: the sphere itself.
pub fn single_sphere(r: f64) -> AreaVolume {
    AreaVolume {
        area: 4.0 * PI * r * r,
        volume: 4.0 / 3.0 * PI * r * r * r,
        components: 1,
    }
}

/// Area of the zone of a sphere of radius `r` between heights `a <= b`
/// (measured from the centre along the axis).
fn zone_area(r: f64, a: f64, b: f64) -> f64 {
    2.0 * PI * r * (b - a)
}

/// Volume of the slab of a ball of radius `r` between heights `a <= b`.
fn zone_volume(r: f64, a: f64, b: f64) -> f64 {
    let f = |h: f64| PI * (r * r * h - h * h * h / 3.0);
    f(b) - f(a)
}

/// 16-point Gauss–Legendre nodes and weights on [-1, 1].
const GL_X: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_4,
    0.617_876_244_402_643_7,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];
const GL_W: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_6,
    0.169_156_519_395_002_5,
    0.149_595_988_816_576_7,
    0.124_628_971_255_533_9,
    0.095_158_511_682_492_78,
    0.062_253_523_938_647_89,
    0.027_152_459_411_754_09,
];

/// Composite 16-point Gauss–Legendre integral of `f` on `[a, b]`.
fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    const PANELS: usize = 64;
    let h = (b - a) / PANELS as f64;
    let mut total = 0.0;
    for p in 0..PANELS {
        let mid = a + (p as f64 + 0.5) * h;
        let half = 0.5 * h;
        for k in 0..8 {
            total += GL_W[k] * (f(mid - half * GL_X[k]) + f(mid + half * GL_X[k])) * half;
        }
    }
    total
}

/// SES of two spheres with vdW radii `r1`, `r2`, centre distance `d` and
/// probe radius `rp`.
///
/// # Panics
/// If one expanded sphere contains the other (not a two-sphere SES).
pub fn two_spheres(r1: f64, r2: f64, d: f64, rp: f64) -> AreaVolume {
    let (big1, big2) = (r1 + rp, r2 + rp);
    assert!(d > (big1 - big2).abs(), "nested spheres");
    if d >= big1 + big2 {
        let (a, b) = (single_sphere(r1), single_sphere(r2));
        return AreaVolume {
            area: a.area + b.area,
            volume: a.volume + b.volume,
            components: 2,
        };
    }
    // Meridian plane: c1 = (0, 0), c2 = (d, 0); probe centre (xp, yp).
    let xp = (d * d + big1 * big1 - big2 * big2) / (2.0 * d);
    let yp = (big1 * big1 - xp * xp).sqrt();
    // Contact points and their angles on the probe circle.
    let xc1 = xp * r1 / big1;
    let xc2 = d + (xp - d) * r2 / big2;
    let phi1 = (-xp / big1).asin();
    let phi2 = ((d - xp) / big2).asin();
    assert!(phi1 < phi2 && xc1 < xc2);

    // Arc height above the axis; clipped where negative.
    let y = |phi: f64| yp - rp * phi.cos();
    let (lo, hi) = if yp >= rp {
        (phi2, phi2) // never below the axis
    } else {
        let a = (yp / rp).acos(); // y(±a) = 0, negative for |phi| < a
        (a.mul_add(-1.0, 0.0).max(phi1), a.min(phi2))
    };
    let pieces: Vec<(f64, f64)> = if lo < hi {
        vec![(phi1, lo), (hi, phi2)]
    } else {
        vec![(phi1, phi2)]
    };
    let mut area = zone_area(r1, -r1, xc1) + zone_area(r2, xc2 - d, r2);
    let mut volume = zone_volume(r1, -r1, xc1) + zone_volume(r2, xc2 - d, r2);
    for &(a, b) in &pieces {
        // dA = 2π y r_p dφ ; dV = π y² dx with dx = r_p cos φ dφ
        area += integrate(|phi| 2.0 * PI * y(phi).max(0.0) * rp, a, b);
        volume += integrate(|phi| PI * y(phi).max(0.0).powi(2) * rp * phi.cos(), a, b);
    }
    AreaVolume {
        area,
        volume,
        components: if lo < hi { 2 } else { 1 },
    }
}

/// Closed-form SES area of two equal spheres (radius `r`, distance `d`,
/// probe `rp`), Design §15. Independent of [`two_spheres`]' quadrature.
pub fn two_equal_spheres_area(r: f64, d: f64, rp: f64) -> f64 {
    let big = r + rp;
    if d >= 2.0 * big {
        return 8.0 * PI * r * r;
    }
    let sin_theta = d / (2.0 * big);
    let theta = sin_theta.asin();
    let rho = (big * big - d * d / 4.0).sqrt();
    let phi0 = (rho / rp).min(1.0).acos();
    let convex = 2.0 * 2.0 * PI * r * r * (1.0 + sin_theta);
    let saddle = 4.0 * PI * rp * (rho * (theta - phi0) - rp * (sin_theta - phi0.sin()));
    convex + saddle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_matches_quadrature() {
        for &d in &[0.5, 1.0, 2.0, 3.0, 4.5, 5.6, 6.0, 6.19] {
            let q = two_spheres(1.7, 1.7, d, 1.4);
            let c = two_equal_spheres_area(1.7, d, 1.4);
            assert!((q.area - c).abs() < 1e-9 * c, "d = {d}: {} vs {c}", q.area);
        }
    }

    #[test]
    fn limits() {
        // Far apart: two separate spheres.
        let far = two_spheres(1.5, 1.2, 10.0, 1.4);
        assert!((far.area - (single_sphere(1.5).area + single_sphere(1.2).area)).abs() < 1e-12);
        // Probe -> 0: union of two balls (lens removed).
        let (r, d) = (1.7, 2.0);
        let union = two_spheres(r, r, d, 1e-9);
        let h = r - d / 2.0;
        let cap_area = 2.0 * PI * r * h;
        let cap_volume = PI * h * h * (3.0 * r - h) / 3.0;
        assert!((union.area - 2.0 * (4.0 * PI * r * r - cap_area)).abs() < 1e-6);
        assert!((union.volume - 2.0 * (4.0 / 3.0 * PI * r.powi(3) - cap_volume)).abs() < 1e-6);
        // Self-intersecting torus gives two components.
        assert_eq!(two_spheres(1.7, 1.7, 5.6, 1.4).components, 2);
        assert_eq!(two_spheres(1.7, 1.7, 3.0, 1.4).components, 1);
    }

    #[test]
    fn unequal_spheres_are_continuous_and_symmetric() {
        let a = two_spheres(1.2, 2.0, 3.0, 1.4);
        let b = two_spheres(2.0, 1.2, 3.0, 1.4);
        assert!((a.area - b.area).abs() < 1e-9 && (a.volume - b.volume).abs() < 1e-9);
        let near = two_spheres(1.2, 2.0, 2.0 * 1.4 + 3.2 - 1e-7, 1.4);
        let apart = two_spheres(1.2, 2.0, 2.0 * 1.4 + 3.2 + 1e-7, 1.4);
        assert!(
            (near.area - apart.area).abs() < 1e-3,
            "{} vs {}",
            near.area,
            apart.area
        );
    }
}
