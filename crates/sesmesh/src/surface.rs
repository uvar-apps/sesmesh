//! Surface computation parameters and the main entry point
//! [`compute_surface`] (Design §3.3, §4.2, §13).

use std::time::{Duration, Instant};

use crate::diag::Diagnostics;
use crate::error::{Error, Result};
use crate::field::{Engine, SesField, SimdLevel, SurfaceKind};
use crate::geom::Sphere;
use crate::mesh::{self, Mesh, extract};

/// Resolution presets (Spec FR-LOD-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Quality {
    /// 1.0 Å grid spacing.
    Draft,
    /// 0.7 Å.
    Low,
    /// 0.5 Å (default).
    #[default]
    Medium,
    /// 0.25 Å.
    High,
    /// 0.1 Å.
    Ultra,
}

impl Quality {
    /// All presets, coarse to fine.
    pub const ALL: [Quality; 5] = [
        Quality::Draft,
        Quality::Low,
        Quality::Medium,
        Quality::High,
        Quality::Ultra,
    ];

    /// Grid spacing in Å.
    pub fn spacing(self) -> f64 {
        match self {
            Self::Draft => 1.0,
            Self::Low => 0.7,
            Self::Medium => 0.5,
            Self::High => 0.25,
            Self::Ultra => 0.1,
        }
    }

    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }

    /// Parses a preset name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|q| q.name().eq_ignore_ascii_case(name.trim()))
    }
}

/// How finely the surface is sampled.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Resolution {
    /// Grid spacing in Å (Spec FR-LOD-1).
    Spacing(f64),
    /// A preset.
    Quality(Quality),
}

impl Resolution {
    /// Grid spacing in Å.
    pub fn spacing(self) -> f64 {
        match self {
            Self::Spacing(h) => h,
            Self::Quality(q) => q.spacing(),
        }
    }
}

/// How vertices are placed on crossing grid edges (Design §8.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum RefineMode {
    /// Exact root of the analytic field on the edge (vertices on the surface
    /// to about 10⁻⁶ Å). Default.
    #[default]
    Exact,
    /// Linear interpolation of the sampled values (faster, vertices within
    /// about `h²/8r` of the surface).
    Linear,
}

impl RefineMode {
    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Linear => "linear",
        }
    }
}

/// Parameters of [`compute_surface`].
///
/// ```
/// use sesmesh::{Quality, SurfaceParams};
///
/// let params = SurfaceParams::ses(1.4).with_quality(Quality::High);
/// assert_eq!(params.spacing(), 0.25);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SurfaceParams {
    /// Surface kind. Default SES.
    pub kind: SurfaceKind,
    /// Probe radius in Å (ignored for vdW). Default 1.4.
    pub probe_radius: f64,
    /// Grid resolution. Default [`Quality::Medium`] (0.5 Å).
    pub resolution: Resolution,
    /// Vertex placement. Default [`RefineMode::Exact`].
    pub refine: RefineMode,
    /// Brick edge in cells (expert; 2..=32). Default 8.
    pub brick_size: usize,
    /// Kernel instruction set (expert). Default [`SimdLevel::Auto`].
    pub simd: SimdLevel,
}

impl Default for SurfaceParams {
    fn default() -> Self {
        Self::ses(1.4)
    }
}

impl SurfaceParams {
    /// Solvent excluded surface with the given probe radius.
    pub fn ses(probe_radius: f64) -> Self {
        Self {
            kind: SurfaceKind::Ses,
            probe_radius,
            resolution: Resolution::Quality(Quality::Medium),
            refine: RefineMode::Exact,
            brick_size: 8,
            simd: SimdLevel::Auto,
        }
    }

    /// Solvent accessible surface with the given probe radius.
    pub fn sas(probe_radius: f64) -> Self {
        Self {
            kind: SurfaceKind::Sas,
            ..Self::ses(probe_radius)
        }
    }

    /// Van der Waals surface.
    pub fn vdw() -> Self {
        Self {
            kind: SurfaceKind::Vdw,
            ..Self::ses(0.0)
        }
    }

    /// Sets the grid spacing (Å).
    #[must_use]
    pub fn with_spacing(mut self, spacing: f64) -> Self {
        self.resolution = Resolution::Spacing(spacing);
        self
    }

    /// Sets a quality preset.
    #[must_use]
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.resolution = Resolution::Quality(quality);
        self
    }

    /// Sets the vertex placement mode.
    #[must_use]
    pub fn with_refine(mut self, refine: RefineMode) -> Self {
        self.refine = refine;
        self
    }

    /// Sets the brick size in cells (expert).
    #[must_use]
    pub fn with_brick_size(mut self, cells: usize) -> Self {
        self.brick_size = cells;
        self
    }

    /// Sets the kernel instruction set (expert).
    #[must_use]
    pub fn with_simd(mut self, simd: SimdLevel) -> Self {
        self.simd = simd;
        self
    }

    /// Grid spacing in Å.
    pub fn spacing(&self) -> f64 {
        self.resolution.spacing()
    }

    /// Probe radius actually used (0 for vdW).
    pub fn effective_probe(&self) -> f64 {
        if self.kind == SurfaceKind::Vdw {
            0.0
        } else {
            self.probe_radius
        }
    }

    /// Checks the parameter ranges.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] describing the first invalid value.
    pub fn validate(&self) -> Result<()> {
        let h = self.spacing();
        if !(h.is_finite() && (0.01..=10.0).contains(&h)) {
            return Err(Error::InvalidInput(format!(
                "grid spacing must be within 0.01..=10 Å, got {h}"
            )));
        }
        let p = self.probe_radius;
        if self.kind != SurfaceKind::Vdw && !(p.is_finite() && (0.0..=100.0).contains(&p)) {
            return Err(Error::InvalidInput(format!(
                "probe radius must be within 0..=100 Å, got {p}"
            )));
        }
        if !(2..=32).contains(&self.brick_size) {
            return Err(Error::InvalidInput(format!(
                "brick size must be 2..=32 cells, got {}",
                self.brick_size
            )));
        }
        Ok(())
    }
}

/// Wall-clock time of each stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Timings {
    /// SAS arrangement (neighbours, circles, vertices).
    pub geometry: Duration,
    /// Brick classification.
    pub classify: Duration,
    /// Field sampling and per-brick meshing.
    pub mesh: Duration,
    /// Global assembly.
    pub assemble: Duration,
    /// Validation, components and measurements.
    pub measure: Duration,
}

impl Timings {
    /// Sum of all stages.
    pub fn total(&self) -> Duration {
        self.geometry + self.classify + self.mesh + self.assemble + self.measure
    }
}

/// Statistics of a computed surface.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SurfaceStats {
    /// Input spheres.
    pub spheres: usize,
    /// Surface kind actually computed (an SES with probe 0 is a vdW surface).
    pub kind: SurfaceKind,
    /// Grid spacing used, Å.
    pub spacing: f64,
    /// Bricks overlapping the molecule.
    pub touched_bricks: usize,
    /// Bricks the surface can pass through (sampled and meshed).
    pub surface_bricks: usize,
    /// Mesh vertices.
    pub vertices: usize,
    /// Mesh triangles.
    pub triangles: usize,
    /// Total area, Å².
    pub area: f64,
    /// Total signed volume, Å³ (outer surfaces minus cavities).
    pub volume: f64,
    /// Connected components.
    pub components: usize,
    /// Components that are internal cavities (negative volume).
    pub cavities: usize,
    /// Euler characteristic of the whole mesh.
    pub euler: i64,
    /// Exact field evaluations spent on vertex placement.
    pub refine_evaluations: u64,
    /// Kernel instruction set used.
    pub simd: &'static str,
    /// Stage timings.
    pub timings: Timings,
}

/// Result of [`compute_surface`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SurfaceOutput {
    /// The surface mesh.
    pub mesh: Mesh,
    /// Per-component properties, in component order (see
    /// [`mesh::components`]).
    pub components: Vec<mesh::Component>,
    /// Statistics.
    pub stats: SurfaceStats,
    /// Warnings.
    pub diagnostics: Diagnostics,
}

/// Computes the surface of a set of spheres (Spec FR-LIB-1).
///
/// Runs on all cores of the current rayon thread pool (with the `parallel`
/// feature); the output does not depend on the number of threads.
///
/// ```
/// use sesmesh::{Sphere, SurfaceParams, compute_surface, mesh};
///
/// let atoms = [Sphere::new([0.0, 0.0, 0.0], 1.7), Sphere::new([2.5, 0.0, 0.0], 1.5)];
/// let out = compute_surface(&atoms, &SurfaceParams::ses(1.4).with_spacing(0.4))?;
/// assert!(mesh::validate(&out.mesh).is_ok());
/// assert_eq!(out.stats.components, 1);
/// assert!(out.stats.area > 50.0 && out.stats.volume > 30.0);
/// # Ok::<(), sesmesh::Error>(())
/// ```
///
/// # Errors
/// [`Error::InvalidInput`] for invalid spheres or parameters,
/// [`Error::ResourceLimit`] for meshes beyond `u32` indices, and
/// [`Error::Internal`] if the mesh fails validation (a bug).
pub fn compute_surface(spheres: &[Sphere], params: &SurfaceParams) -> Result<SurfaceOutput> {
    params.validate()?;
    let mut timings = Timings::default();
    let mut clock = Instant::now();
    let mut lap = |slot: &mut Duration| {
        let now = Instant::now();
        *slot = now - clock;
        clock = now;
    };

    let field = SesField::new(spheres, params.effective_probe(), params.kind)?;
    let kind = field.kind();
    let engine = Engine::new(field, params.spacing(), params.brick_size, params.simd)?;
    lap(&mut timings.geometry);

    let (touched, bricks) = engine.surface_bricks()?;
    lap(&mut timings.classify);

    let parts = engine.map_bricks(&bricks, |key, values| {
        extract::mesh_brick(&engine, key, values, params.refine)
    });
    lap(&mut timings.mesh);

    let refine_evaluations = parts.iter().map(|p| p.evals).sum();
    let mesh = extract::assemble(&bricks, &parts)?;
    drop(parts);
    lap(&mut timings.assemble);

    let check = mesh::validate(&mesh);
    if !check.is_ok() {
        return Err(Error::Internal(format!(
            "mesh failed validation: {check:?}"
        )));
    }
    let (_, components) = mesh::components(&mesh);
    lap(&mut timings.measure);

    let stats = SurfaceStats {
        spheres: spheres.len(),
        kind,
        spacing: params.spacing(),
        touched_bricks: touched,
        surface_bricks: bricks.len(),
        vertices: mesh.vertex_count(),
        triangles: mesh.triangle_count(),
        area: components.iter().map(|c| c.area).sum(),
        volume: components.iter().map(|c| c.volume).sum(),
        components: components.len(),
        cavities: components.iter().filter(|c| c.is_cavity()).count(),
        euler: components.iter().map(|c| c.euler).sum(),
        refine_evaluations,
        simd: engine.simd_name(),
        timings,
    };
    Ok(SurfaceOutput {
        mesh,
        components,
        stats,
        diagnostics: Diagnostics::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "parallel")]
    use crate::export::obj::{ObjOptions, write_obj};
    use proptest::prelude::*;
    use sesmesh_testkit::clusters::{
        Cluster, Rng, cage, cluster_strategy, collinear, cubic_lattice, random_box, random_chain,
        with_duplicates,
    };

    fn spheres(c: &Cluster) -> Vec<Sphere> {
        c.centers
            .iter()
            .zip(&c.radii)
            .map(|(&center, &radius)| Sphere::new(center, radius))
            .collect()
    }

    #[cfg(feature = "parallel")]
    fn obj_bytes(out: &SurfaceOutput) -> Vec<u8> {
        let mut buf = Vec::new();
        write_obj(&mut buf, &out.mesh, &ObjOptions::default()).unwrap();
        buf
    }

    /// NFR-DET: identical mesh and OBJ bytes for any thread count.
    #[cfg(feature = "parallel")]
    #[test]
    fn thread_count_does_not_change_output() {
        let c = random_chain(&mut Rng::new(11), 60);
        let s = spheres(&c);
        let params = SurfaceParams::ses(c.probe).with_spacing(0.3);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| compute_surface(&s, &params).unwrap())
        };
        let one = run(1);
        assert!(one.stats.triangles > 1000);
        let bytes = obj_bytes(&one);
        for threads in [2, 4, 7] {
            let other = run(threads);
            assert_eq!(other.mesh, one.mesh, "{threads} threads");
            assert_eq!(obj_bytes(&other), bytes, "{threads} threads");
        }
    }

    /// Degenerate geometries of the robustness corpus (Design §15) give
    /// closed, valid meshes (`compute_surface` validates every mesh).
    #[test]
    fn robustness_corpus() {
        let cases = [
            cubic_lattice(4, 3.0, 1.6, 1.4),
            cage(6.0, 1.5, 1.4),
            collinear(8, 2.0, 1.5, 1.4),
            with_duplicates(random_box(&mut Rng::new(3), 30, 10.0), 4),
            random_chain(&mut Rng::new(9), 50),
        ];
        for (k, c) in cases.iter().enumerate() {
            for kind in [SurfaceKind::Ses, SurfaceKind::Sas, SurfaceKind::Vdw] {
                let mut p = SurfaceParams::ses(c.probe).with_spacing(0.4);
                p.kind = kind;
                let out = compute_surface(&spheres(c), &p)
                    .unwrap_or_else(|e| panic!("case {k} {kind:?}: {e}"));
                assert!(out.stats.triangles > 0, "case {k} {kind:?}");
            }
        }
        // Far from the origin, tiny and huge probes.
        let mut c = random_chain(&mut Rng::new(4), 20);
        for p in &mut c.centers {
            p[0] += 1.0e4;
        }
        compute_surface(&spheres(&c), &SurfaceParams::ses(1.4).with_spacing(0.5)).unwrap();
        for probe in [0.1, 10.0] {
            compute_surface(&spheres(&c), &SurfaceParams::ses(probe).with_spacing(0.5)).unwrap();
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        /// Random clusters at random spacings and brick sizes: always a
        /// valid closed mesh.
        #[test]
        fn random_clusters_mesh_cleanly(
            c in cluster_strategy(25),
            h in 0.25f64..0.9,
            brick in 2usize..12,
        ) {
            let p = SurfaceParams::ses(c.probe).with_spacing(h).with_brick_size(brick);
            let out = compute_surface(&spheres(&c), &p);
            prop_assert!(out.is_ok(), "{:?}", out.err());
        }
    }
}
