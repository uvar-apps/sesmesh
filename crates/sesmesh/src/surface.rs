//! Surface computation parameters and the main entry point
//! [`compute_surface`] (Design §3.3, §4.2, §13).

use std::time::{Duration, Instant};

use crate::control::{Control, Stage};
use crate::diag::Diagnostics;
use crate::error::{Error, Result};
use crate::field::{Engine, ScratchPool, SesField, SimdLevel, SurfaceKind};
use crate::geom::Sphere;
use crate::mesh::{self, Mesh, extract, post};

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
    /// Target vertex density in vertices per Å² (Spec FR-LOD-3).
    Density(f64),
}

/// Mesh vertices per Å² times h²: crossing grid edges per unit area of a
/// surface average `1.5 / h²` over orientations (Design §8.6); measured
/// 1.50 on crambin at 0.25 Å. Calibrated on the validation set in Phase 9.
pub const VERTICES_PER_AREA_H2: f64 = 1.5;

impl Resolution {
    /// Grid spacing in Å (for a density, `√(1.5 / density)`).
    pub fn spacing(self) -> f64 {
        match self {
            Self::Spacing(h) => h,
            Self::Quality(q) => q.spacing(),
            Self::Density(d) => (VERTICES_PER_AREA_H2 / d).sqrt(),
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

/// Which components to keep (Spec FR-SURF-4, Design §9.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CavityMode {
    /// Keep every component, including internal cavities (the exact SES).
    /// Default.
    #[default]
    Keep,
    /// Drop internal cavities and anything inside them (MSMS-like output).
    Remove,
    /// Keep internal cavities only.
    Only,
}

impl CavityMode {
    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Remove => "remove",
            Self::Only => "only",
        }
    }
}

/// Which vertex normals the mesh carries (Spec FR-MESH-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum NormalMode {
    /// Exact normals of the analytic surface (Design §8.2). Default.
    #[default]
    Analytic,
    /// Area-weighted averages of the incident triangle normals.
    FaceAverage,
    /// No normals (`Mesh::normals` is empty).
    None,
}

impl NormalMode {
    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Analytic => "analytic",
            Self::FaceAverage => "face-average",
            Self::None => "none",
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
    /// Brick edge in cells (expert; 2..=32). Default 10 (measured fastest
    /// among 4–16 on 4HHB and 1AON, Design §7.7).
    pub brick_size: usize,
    /// Kernel instruction set (expert). Default [`SimdLevel::Auto`].
    pub simd: SimdLevel,
    /// Component selection. Default [`CavityMode::Keep`].
    pub cavities: CavityMode,
    /// Drop cavities with |volume| below this (Å³). Default 0.
    pub min_cavity_volume: f64,
    /// Drop components with area below this (Å²). Default 0.
    pub min_component_area: f64,
    /// Vertex normals. Default [`NormalMode::Analytic`].
    pub normals: NormalMode,
    /// Limit on the estimated peak memory of the mesh stages (bytes), checked
    /// after brick classification. Default `None`.
    pub max_memory: Option<u64>,
    /// With `max_memory`: coarsen the grid until the estimate fits instead of
    /// refusing (Spec FR-LOD-6); `SurfaceStats::spacing` reports the spacing
    /// used. Default `false`.
    pub auto_coarsen: bool,
    /// Taubin smoothing rounds followed by exact re-projection (Spec
    /// FR-LOD-5). Default 0 (off).
    pub smoothing: u32,
    /// Decimation after smoothing (Spec FR-LOD-4); retained vertices stay
    /// on the exact surface. Default `None`.
    pub decimation: Option<mesh::Decimation>,
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
            brick_size: 10,
            simd: SimdLevel::Auto,
            cavities: CavityMode::Keep,
            min_cavity_volume: 0.0,
            min_component_area: 0.0,
            normals: NormalMode::Analytic,
            max_memory: None,
            auto_coarsen: false,
            smoothing: 0,
            decimation: None,
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

    /// Sets a target vertex density (vertices per Å²).
    #[must_use]
    pub fn with_density(mut self, density: f64) -> Self {
        self.resolution = Resolution::Density(density);
        self
    }

    /// Refuses meshes whose estimated peak memory exceeds `bytes`
    /// (Spec FR-LOD-6; [`Error::ResourceLimit`]).
    #[must_use]
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory = Some(bytes);
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

    /// Sets the component selection.
    #[must_use]
    pub fn with_cavities(mut self, mode: CavityMode) -> Self {
        self.cavities = mode;
        self
    }

    /// Drops cavities with |volume| below `volume` Å³.
    #[must_use]
    pub fn with_min_cavity_volume(mut self, volume: f64) -> Self {
        self.min_cavity_volume = volume;
        self
    }

    /// Drops components with area below `area` Å².
    #[must_use]
    pub fn with_min_component_area(mut self, area: f64) -> Self {
        self.min_component_area = area;
        self
    }

    /// With [`SurfaceParams::with_max_memory`]: coarsen the grid until the
    /// memory estimate fits instead of refusing.
    #[must_use]
    pub fn with_auto_coarsen(mut self, on: bool) -> Self {
        self.auto_coarsen = on;
        self
    }

    /// Smooths the mesh for `rounds` Taubin rounds and re-projects every
    /// vertex onto the exact surface (better-shaped triangles, vertices still
    /// on the surface).
    #[must_use]
    pub fn with_smoothing(mut self, rounds: u32) -> Self {
        self.smoothing = rounds;
        self
    }

    /// Decimates the mesh (quadric-error half-edge collapses; vertices stay
    /// on the exact surface).
    #[must_use]
    pub fn with_decimation(mut self, decimation: mesh::Decimation) -> Self {
        self.decimation = Some(decimation);
        self
    }

    /// Sets the vertex normal mode.
    #[must_use]
    pub fn with_normals(mut self, normals: NormalMode) -> Self {
        self.normals = normals;
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
        if let Resolution::Density(d) = self.resolution
            && !(d.is_finite() && d > 0.0)
        {
            return Err(Error::InvalidInput(format!(
                "vertex density must be finite and > 0, got {d}"
            )));
        }
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
        if let Some(d) = &self.decimation {
            if !(d.ratio.is_finite() && d.ratio > 0.0 && d.ratio <= 1.0) {
                return Err(Error::InvalidInput(format!(
                    "decimation ratio must be within (0, 1], got {}",
                    d.ratio
                )));
            }
            if d.max_error.is_nan() || d.max_error < 0.0 {
                return Err(Error::InvalidInput(format!(
                    "decimation error bound must be >= 0, got {}",
                    d.max_error
                )));
            }
        }
        for (name, v) in [
            ("minimum cavity volume", self.min_cavity_volume),
            ("minimum component area", self.min_component_area),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::InvalidInput(format!(
                    "{name} must be finite and >= 0, got {v}"
                )));
            }
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
    /// Grid spacing used, Å (larger than requested after auto-coarsening).
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
    /// Components dropped by the cavity mode and filters.
    pub removed_components: usize,
    /// Decimation result, when decimation was requested.
    pub decimation: Option<mesh::DecimationStats>,
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
    compute_surface_with(spheres, params, &Control::new())
}

/// [`compute_surface`] with cancellation and progress reporting
/// (Spec FR-LIB-5).
///
/// # Errors
/// As [`compute_surface`], and [`Error::Cancelled`] when `control`'s flag is
/// set.
pub fn compute_surface_with(
    spheres: &[Sphere],
    params: &SurfaceParams,
    control: &Control<'_>,
) -> Result<SurfaceOutput> {
    run(spheres, params, control, &ScratchPool::default())
}

/// A reusable surface computation (Spec FR-LIB-3), e.g. for the frames of a
/// trajectory: parameters are validated once and the per-thread field
/// sampling buffers are kept between calls.
///
/// ```
/// use sesmesh::{Sphere, SurfaceEngine, SurfaceParams};
///
/// let mut engine = SurfaceEngine::new(SurfaceParams::ses(1.4).with_spacing(0.5))?;
/// for x in [2.5, 3.0, 3.5] {
///     let frame = [Sphere::new([0.0; 3], 1.7), Sphere::new([x, 0.0, 0.0], 1.7)];
///     let out = engine.compute(&frame)?;
///     assert_eq!(out.stats.components, 1);
/// }
/// # Ok::<(), sesmesh::Error>(())
/// ```
#[derive(Debug)]
pub struct SurfaceEngine {
    params: SurfaceParams,
    pool: ScratchPool,
}

impl SurfaceEngine {
    /// Creates an engine.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for invalid parameters.
    pub fn new(params: SurfaceParams) -> Result<Self> {
        params.validate()?;
        Ok(Self {
            params,
            pool: ScratchPool::default(),
        })
    }

    /// The parameters.
    pub fn params(&self) -> &SurfaceParams {
        &self.params
    }

    /// Computes the surface of `spheres`.
    ///
    /// # Errors
    /// As [`compute_surface`].
    pub fn compute(&mut self, spheres: &[Sphere]) -> Result<SurfaceOutput> {
        self.compute_with(spheres, &Control::new())
    }

    /// Computes the surface of `spheres` with cancellation and progress.
    ///
    /// # Errors
    /// As [`compute_surface_with`].
    pub fn compute_with(
        &mut self,
        spheres: &[Sphere],
        control: &Control<'_>,
    ) -> Result<SurfaceOutput> {
        run(spheres, &self.params, control, &self.pool)
    }
}

/// Estimated triangle count and peak bytes for `surface_bricks` bricks and
/// `spheres` input spheres (Design §8.4, §8.6): area ≤ 0.7·(B·h)² per
/// surface brick (measured 0.38–0.68 on 1CRN, 4HHB, 1AON), `2 · 1.5 / h²`
/// triangles per Å², and peak memory of 150 bytes per triangle plus 800 bytes
/// per sphere (Phase 9 measurements: 110–150 B per triangle on 1AON, 4V6X,
/// 3J3Q; ≈ 600–800 B per atom for the SAS arrangement and structure data,
/// docs/validation-report.md).
pub fn estimate_memory(
    surface_bricks: usize,
    brick_size: usize,
    spacing: f64,
    spheres: usize,
) -> (u64, u64) {
    let side = brick_size as f64 * spacing;
    let area = surface_bricks as f64 * 0.7 * side * side;
    let triangles = 2.0 * VERTICES_PER_AREA_H2 * area / (spacing * spacing);
    (
        triangles as u64,
        (triangles * 150.0 + spheres as f64 * 800.0) as u64,
    )
}

/// Applies the cavity mode and filters (Design §9.2); returns the number of
/// removed components.
fn select_components(
    mesh: &mut Mesh,
    vertex_component: &mut Vec<u32>,
    components: &mut Vec<mesh::Component>,
    params: &SurfaceParams,
) -> usize {
    let mut keep: Vec<bool> = components
        .iter()
        .map(|c| {
            let by_mode = match params.cavities {
                CavityMode::Keep => true,
                CavityMode::Remove => !c.is_cavity(),
                CavityMode::Only => c.is_cavity(),
            };
            by_mode
                && !(c.is_cavity() && -c.volume < params.min_cavity_volume)
                && c.area >= params.min_component_area
        })
        .collect();
    // Whatever lies inside a removed cavity goes with it.
    let removed: Vec<bool> = keep.iter().map(|k| !k).collect();
    if removed.iter().any(|&r| r) {
        let inner = post::enclosed_by(mesh, vertex_component, components, &removed);
        for (k, i) in keep.iter_mut().zip(inner) {
            *k &= !i;
        }
    }
    let n = keep.iter().filter(|k| !**k).count();
    post::retain_components(mesh, vertex_component, components, &keep);
    n
}

fn run(
    spheres: &[Sphere],
    params: &SurfaceParams,
    control: &Control<'_>,
    pool: &ScratchPool,
) -> Result<SurfaceOutput> {
    params.validate()?;
    control.check()?;
    let mut timings = Timings::default();
    let mut clock = Instant::now();
    let mut lap = |slot: &mut Duration| {
        let now = Instant::now();
        *slot = now - clock;
        clock = now;
    };

    control.report(Stage::Geometry, 0, 1);
    let field = SesField::new_with(spheres, params.effective_probe(), params.kind, control)?;
    let kind = field.kind();
    let mut engine = Engine::new(field, params.spacing(), params.brick_size, params.simd)?;
    control.report(Stage::Geometry, 1, 1);
    lap(&mut timings.geometry);

    control.report(Stage::Classify, 0, 1);
    let mut spacing = params.spacing();
    let (mut touched, mut bricks) = engine.surface_bricks()?;
    control.check()?;
    if let Some(limit) = params.max_memory {
        // Refuse, or (auto_coarsen) grow the spacing until the estimate fits:
        // triangles scale as 1/h², so one corrected pass usually suffices.
        let mut attempts = 0;
        loop {
            let (triangles, bytes) =
                estimate_memory(bricks.len(), params.brick_size, spacing, spheres.len());
            if bytes <= limit {
                break;
            }
            let atom_bytes = spheres.len() as u64 * 800;
            if !params.auto_coarsen || attempts == 4 || limit <= atom_bytes || spacing >= 10.0 {
                return Err(Error::ResourceLimit {
                    what: format!(
                        "estimated mesh memory ({bytes} bytes for about {triangles} triangles at {spacing} Å)"
                    ),
                    limit,
                });
            }
            let factor = ((bytes - atom_bytes) as f64 / (limit - atom_bytes) as f64).sqrt();
            spacing = (spacing * factor * 1.05).min(10.0);
            let field = engine.field;
            engine = Engine::new(field, spacing, params.brick_size, params.simd)?;
            (touched, bricks) = engine.surface_bricks()?;
            control.check()?;
            attempts += 1;
        }
    }
    control.report(Stage::Classify, 1, 1);
    lap(&mut timings.classify);

    let chunk = crate::par::chunk_len(bricks.len());
    let parts = engine.fold_brick_chunks_with(
        &bricks,
        chunk,
        control,
        Stage::Mesh,
        pool,
        extract::ChunkMesh::default,
        |part, key, values| part.push(&engine, key, values, params.refine),
    )?;
    lap(&mut timings.mesh);

    control.report(Stage::Assemble, 0, 1);
    let refine_evaluations = parts.iter().map(extract::ChunkMesh::evals).sum();
    let mesh = extract::assemble(&bricks, chunk, parts)?;
    control.check()?;
    control.report(Stage::Assemble, 1, 1);
    lap(&mut timings.assemble);

    control.report(Stage::Measure, 0, 1);
    let (check, edges) = mesh::validate_with(&mesh, None);
    let edges = match edges {
        Some(edges) if check.is_ok() => edges,
        _ => {
            return Err(Error::Internal(format!(
                "mesh failed validation: {check:?}"
            )));
        }
    };
    control.check()?;
    let (mut vertex_component, mut components) = mesh::components_with(&mesh, &edges);
    drop(edges);
    let mut mesh = mesh;
    post::group_by_component(&mut mesh, &vertex_component, &mut components);
    let removed_components =
        select_components(&mut mesh, &mut vertex_component, &mut components, params);
    if params.smoothing > 0 {
        post::smooth_and_project(&mut mesh, &engine.field, spheres, params.smoothing);
        let check = mesh::validate(&mesh);
        if !check.is_ok() {
            return Err(Error::Internal(format!(
                "smoothed mesh failed validation: {check:?}"
            )));
        }
        // Areas and volumes of the smoothed surface.
        let origin = mesh
            .positions
            .first()
            .map_or([0.0; 3], |p| p.map(f64::from));
        for c in &mut components {
            let (mut area, mut volume) = (0.0, 0.0);
            for t in &mesh.triangles[c.first_triangle..c.first_triangle + c.triangles] {
                let (a, v) = mesh::triangle_area_volume(&mesh, *t, origin);
                area += a;
                volume += v;
            }
            c.area = area;
            c.volume = volume;
        }
    }
    let mut decimation = None;
    if let Some(d) = &params.decimation {
        mesh.vertex_component = vertex_component;
        decimation = Some(mesh::decimate(&mut mesh, d));
        let check = mesh::validate(&mesh);
        if !check.is_ok() {
            return Err(Error::Internal(format!(
                "decimated mesh failed validation: {check:?}"
            )));
        }
        let (vc, comps) = mesh::components(&mesh);
        vertex_component = vc;
        components = comps;
        post::group_by_component(&mut mesh, &vertex_component, &mut components);
    }
    match params.normals {
        NormalMode::Analytic => {}
        NormalMode::FaceAverage => post::face_average_normals(&mut mesh),
        NormalMode::None => mesh.normals = Vec::new(),
    }
    mesh.vertex_component = vertex_component;
    control.report(Stage::Measure, 1, 1);
    lap(&mut timings.measure);

    let stats = SurfaceStats {
        spheres: spheres.len(),
        kind,
        spacing,
        touched_bricks: touched,
        surface_bricks: bricks.len(),
        vertices: mesh.vertex_count(),
        triangles: mesh.triangle_count(),
        area: components.iter().map(|c| c.area).sum(),
        volume: components.iter().map(|c| c.volume).sum(),
        components: components.len(),
        cavities: components.iter().filter(|c| c.is_cavity()).count(),
        removed_components,
        decimation,
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
    use crate::control::Progress;
    #[cfg(feature = "parallel")]
    use crate::export::obj::{ObjOptions, write_obj};
    use proptest::prelude::*;
    use sesmesh_testkit::clusters::{
        Cluster, Rng, cage, cluster_strategy, collinear, cubic_lattice, random_box, random_chain,
        with_duplicates,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

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

    /// Coordinates 10⁴ Å from the origin give the same surface as near it
    /// (the `f32` kernel frame is centred on the molecule); an extent beyond
    /// the `f32` limit selects the exact `f64` sampling path automatically
    /// (Plan T6.5).
    #[test]
    fn far_coordinates_and_huge_extents() {
        let c = random_chain(&mut Rng::new(31), 30);
        let near = spheres(&c);
        // A multiple of the spacing, so both lattices align.
        let offset = [1.0e4, -2.0e4, 1.5e4];
        let far: Vec<Sphere> = near
            .iter()
            .map(|s| Sphere::new([0, 1, 2].map(|k| s.center[k] + offset[k]), s.radius))
            .collect();
        let params = SurfaceParams::ses(c.probe).with_spacing(0.5);
        let a = compute_surface(&near, &params).unwrap();
        let b = compute_surface(&far, &params).unwrap();
        assert_ne!(b.stats.simd, "scalar-f64");
        assert!(a.stats.vertices.abs_diff(b.stats.vertices) <= 4);
        assert!((a.stats.area - b.stats.area).abs() < 1e-4 * a.stats.area);
        assert!((a.stats.volume - b.stats.volume).abs() < 1e-4 * a.stats.volume);
        // Two copies 5 000 Å apart: f64 path, twice the surface.
        let mut pair = near.clone();
        pair.extend(
            near.iter()
                .map(|s| Sphere::new([s.center[0] + 5000.0, s.center[1], s.center[2]], s.radius)),
        );
        let p = compute_surface(&pair, &params).unwrap();
        assert_eq!(p.stats.simd, "scalar-f64");
        assert_eq!(p.stats.components, 2 * a.stats.components);
        assert!((p.stats.area - 2.0 * a.stats.area).abs() < 1e-4 * a.stats.area);
    }

    /// Setting the flag from the progress callback at the given stage stops
    /// the computation with `Error::Cancelled`, promptly.
    #[test]
    fn cancellation_is_prompt() {
        let c = random_chain(&mut Rng::new(21), 400);
        let s = spheres(&c);
        let params = SurfaceParams::ses(c.probe).with_spacing(0.25);
        for (stage, after) in [(Stage::Geometry, 0), (Stage::Classify, 0), (Stage::Mesh, 1)] {
            let flag = AtomicBool::new(false);
            let set_at = Mutex::new(None::<Instant>);
            let cb = |p: Progress| {
                if p.stage == stage && p.done >= after && !flag.load(Ordering::Relaxed) {
                    *set_at.lock().unwrap() = Some(Instant::now());
                    flag.store(true, Ordering::Relaxed);
                }
            };
            let control = Control::new().with_cancel(&flag).with_progress(&cb);
            let r = compute_surface_with(&s, &params, &control);
            let latency = set_at.lock().unwrap().expect("flag set").elapsed();
            assert!(
                matches!(r, Err(Error::Cancelled)),
                "{stage:?}: {:?}",
                r.map(|o| o.stats)
            );
            let bound = if cfg!(debug_assertions) { 2.0 } else { 0.1 };
            assert!(latency.as_secs_f64() < bound, "{stage:?}: {latency:?}");
        }
    }

    #[test]
    fn progress_covers_every_stage() {
        let c = random_chain(&mut Rng::new(22), 120);
        let reports = Mutex::new(Vec::new());
        let cb = |p: Progress| reports.lock().unwrap().push(p);
        let params = SurfaceParams::ses(c.probe).with_spacing(0.3);
        let out = compute_surface_with(&spheres(&c), &params, &Control::new().with_progress(&cb))
            .unwrap();
        let reports = reports.into_inner().unwrap();
        let mut order: Vec<Stage> = Vec::new();
        for p in &reports {
            assert!(p.done <= p.total, "{p:?}");
            if order.last() != Some(&p.stage) {
                order.push(p.stage);
            }
        }
        use Stage::*;
        assert_eq!(order, [Geometry, Classify, Mesh, Assemble, Measure]);
        for stage in order {
            let total = reports.iter().find(|p| p.stage == stage).unwrap().total;
            assert!(reports.iter().any(|p| p.stage == stage && p.done == 0));
            assert!(reports.iter().any(|p| p.stage == stage && p.done == total));
        }
        let mesh_total = reports.iter().find(|p| p.stage == Mesh).unwrap().total;
        assert_eq!(mesh_total, out.stats.surface_bricks as u64);
        assert!(reports.iter().filter(|p| p.stage == Mesh).count() > 10);
    }

    /// The engine reuses buffers across frames of different sizes and
    /// matches `compute_surface`.
    #[test]
    fn engine_matches_one_shot() {
        let params = SurfaceParams::ses(1.4).with_spacing(0.35);
        let mut engine = SurfaceEngine::new(params.clone()).unwrap();
        for (seed, n) in [(1, 40), (2, 15), (3, 70), (4, 40)] {
            let c = random_chain(&mut Rng::new(seed), n);
            let s = spheres(&c);
            let a = engine.compute(&s).unwrap();
            let b = compute_surface(&s, &params).unwrap();
            assert_eq!(a.mesh, b.mesh, "frame {seed}");
        }
        assert!(SurfaceEngine::new(SurfaceParams::ses(1.4).with_spacing(-1.0)).is_err());
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
