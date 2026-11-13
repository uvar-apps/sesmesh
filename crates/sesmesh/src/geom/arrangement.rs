//! The SAS arrangement (Design §6): spatially ordered spheres, neighbour
//! lists, intersection circles with their exposed arcs and blocker lists,
//! exposed triple points (probe positions touching three atoms), and sphere
//! exposure flags. Everything the field engine needs to evaluate the exact
//! SES distance (Design §2.3) without searching all spheres.
//!
//! All constructions use `f64`. Spheres are stored in Morton order of their
//! cells for memory locality; `orig` maps back to input indices.

use crate::control::Control;
use std::f64::consts::TAU;

use super::Sphere;
use super::cells::{CellGrid, morton};
use super::vec::{Vec3, orthonormal_basis};
use crate::error::{Error, Result};
use crate::par;

/// Absolute geometric tolerance (Å) for containment, intersection and
/// "strictly inside" tests. Wrong decisions within this tolerance change the
/// SES field by O(`EPS`) (Design §2.3).
pub(crate) const EPS: f64 = 1e-9;

/// Angular tolerance (rad) below which exposed arcs are dropped.
const ANG_EPS: f64 = 1e-12;

/// Role of a sphere in the SAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    /// Part of its expanded surface is on the SAS.
    Exposed,
    /// Intersects other spheres but no part of it is on the SAS.
    Buried,
    /// Inside another expanded sphere (or an exact duplicate of an earlier
    /// one); ignored by every later stage.
    Contained,
}

/// Compressed sparse rows of `u32` indices.
#[derive(Debug, Clone, Default)]
pub(crate) struct Csr {
    offsets: Vec<u32>,
    items: Vec<u32>,
}

impl Csr {
    /// An empty table with room for `rows` rows of `items` entries in all.
    fn with_capacity(rows: usize, items: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0);
        Self {
            offsets,
            items: Vec::with_capacity(items),
        }
    }

    fn push_row(&mut self, row: impl IntoIterator<Item = u32>) {
        self.items.extend(row);
        self.close_row();
    }

    /// Ends the current row: the entries pushed to `items` since the last
    /// row end.
    fn close_row(&mut self) {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        self.offsets
            .push(u32::try_from(self.items.len()).expect("fewer than 2^32 entries"));
    }

    /// Appends the rows of `other`.
    fn append(&mut self, other: &Csr) {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        let base = *self.offsets.last().expect("offsets start with 0");
        self.items.extend_from_slice(&other.items);
        u32::try_from(self.items.len()).expect("fewer than 2^32 entries");
        self.offsets
            .extend(other.offsets.iter().skip(1).map(|&o| base + o));
    }

    /// Number of rows.
    fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub(crate) fn row(&self, i: usize) -> &[u32] {
        &self.items[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    pub(crate) fn total(&self) -> usize {
        self.items.len()
    }

    /// Empties the rows for which `keep` is `false`, in place, and releases
    /// the freed capacity.
    pub(crate) fn retain_rows(&mut self, keep: impl Fn(usize) -> bool) {
        let (mut start, mut write) = (0, 0);
        for i in 0..self.offsets.len().saturating_sub(1) {
            // `start` is the row's original start: `offsets[i]` was rewritten.
            let end = self.offsets[i + 1] as usize;
            if keep(i) {
                self.items.copy_within(start..end, write);
                write += end - start;
            }
            self.offsets[i + 1] = u32::try_from(write).expect("fewer than 2^32 entries");
            start = end;
        }
        self.items.truncate(write);
        self.items.shrink_to_fit();
    }

    /// Heap bytes in use.
    pub(crate) fn heap_bytes(&self) -> usize {
        (self.offsets.capacity() + self.items.capacity()) * size_of::<u32>()
    }
}

/// An exposed (not fully buried) intersection circle of expanded spheres
/// `i < j` (internal indices).
#[derive(Debug, Clone)]
pub(crate) struct Circle {
    pub(crate) i: u32,
    pub(crate) j: u32,
    pub(crate) center: Vec3,
    /// Unit normal of the circle plane, from sphere `i` to sphere `j`.
    pub(crate) axis: Vec3,
    /// In-plane unit vector; `e2() = axis × e1` completes the frame.
    pub(crate) e1: Vec3,
    pub(crate) radius: f64,
    /// Range into [`Arrangement::arcs`]; empty when the whole circle is
    /// exposed.
    pub(crate) arcs: (u32, u32),
}

impl Circle {
    /// Second in-plane unit vector, `axis × e1` (recomputed rather than
    /// stored: it is bit-identical to the one [`orthonormal_basis`] returns).
    #[inline]
    pub(crate) fn e2(&self) -> Vec3 {
        self.axis.cross(self.e1)
    }

    /// Point at angle `theta` (`center + radius (cos θ e1 + sin θ e2)`).
    pub(crate) fn point(&self, theta: f64) -> Vec3 {
        self.center + (self.e1 * theta.cos() + self.e2() * theta.sin()) * self.radius
    }

    /// `true` if no other sphere cuts the circle.
    pub(crate) fn is_full(&self) -> bool {
        self.arcs.0 == self.arcs.1
    }
}

/// An exposed arc `[start, end]` (radians, `start < end <= start + 2π`)
/// between the buried intervals of spheres `from` and `to`; its endpoints are
/// SAS vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Arc {
    pub(crate) start: f64,
    pub(crate) end: f64,
    pub(crate) from: u32,
    pub(crate) to: u32,
}

/// An exposed triple point of expanded spheres `atoms` (internal indices,
/// ascending).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SasVertex {
    pub(crate) position: Vec3,
    pub(crate) atoms: [u32; 3],
}

/// Counts describing an arrangement (diagnostics and benchmarks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ArrangementStats {
    /// Spheres given.
    pub input: usize,
    /// Spheres ignored because their radius is 0.
    pub zero_radius: usize,
    /// Spheres inside another expanded sphere or exact duplicates.
    pub contained: usize,
    /// Spheres with part of their surface on the SAS.
    pub exposed: usize,
    /// Intersecting sphere pairs (undirected).
    pub neighbour_pairs: usize,
    /// Intersection circles not fully buried.
    pub exposed_circles: usize,
    /// Exposed arcs (whole exposed circles not counted).
    pub arcs: usize,
    /// Exposed triple points.
    pub vertices: usize,
}

/// The SAS arrangement of a sphere set for one probe radius.
#[derive(Debug, Clone)]
pub(crate) struct Arrangement {
    pub(crate) probe: f64,
    pub(crate) centers: Vec<Vec3>,
    pub(crate) radii: Vec<f64>,
    pub(crate) expanded: Vec<f64>,
    /// Internal index → input index.
    pub(crate) orig: Vec<u32>,
    pub(crate) status: Vec<Status>,
    /// `N(i)`: non-contained spheres whose expanded sphere intersects `i`'s,
    /// ascending internal index.
    pub(crate) neighbors: Csr,
    pub(crate) circles: Vec<Circle>,
    pub(crate) arcs: Vec<Arc>,
    /// Per circle: spheres that cut it (`K(i, j)`), ascending.
    pub(crate) circle_blockers: Csr,
    pub(crate) vertices: Vec<SasVertex>,
    pub(crate) cells: CellGrid,
    pub(crate) max_expanded: f64,
    pub(crate) stats: ArrangementStats,
}

/// How a sphere meets a circle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cut {
    /// No circle point is strictly inside the sphere.
    Empty,
    /// Every circle point is inside the sphere.
    Full,
    /// Circle points with angle in the open interval `(φ + α, φ + 2π − α)`
    /// are inside. `enter` and `leave` are the unit vectors `(cos, sin)` of
    /// the interval's start `φ + α` and end `φ − α` (the crossing points),
    /// obtained without trigonometric calls.
    Interval { enter: [f64; 2], leave: [f64; 2] },
}

fn cut(circle: &Circle, center: Vec3, expanded: f64) -> Cut {
    // |q(θ) − c|² = |g|² + ρ² + 2ρ (cos θ e1·g + sin θ e2·g), g = m − c:
    // inside iff a cos θ + b sin θ < rhs, i.e. cos(θ − φ) < rhs / amp.
    let g = circle.center - center;
    let a = 2.0 * circle.radius * circle.e1.dot(g);
    let b = 2.0 * circle.radius * circle.e2().dot(g);
    let rhs = expanded * expanded - g.norm2() - circle.radius * circle.radius;
    let amp = (a * a + b * b).sqrt();
    if amp <= rhs {
        Cut::Full
    } else if amp <= -rhs || amp == 0.0 {
        Cut::Empty
    } else {
        let (cp, sp) = (a / amp, b / amp); // cos φ, sin φ
        let ca = rhs / amp; // cos α
        let sa = (1.0 - ca * ca).max(0.0).sqrt(); // sin α
        Cut::Interval {
            enter: [cp * ca - sp * sa, sp * ca + cp * sa],
            leave: [cp * ca + sp * sa, sp * ca - cp * sa],
        }
    }
}

/// Period of [`pseudo_angle`].
const PSEUDO_PERIOD: f64 = 4.0;
/// Pseudo-angle gaps below this are treated as covered.
const PSEUDO_EPS: f64 = 1e-12;

/// A monotone, trig-free substitute for the angle of a unit vector, in
/// `[0, 4)` (the "diamond angle"); order-preserving, so it can replace
/// `atan2` for sorting and merging intervals.
fn pseudo_angle(v: [f64; 2]) -> f64 {
    let [x, y] = v;
    if y >= 0.0 {
        if x >= 0.0 {
            y / (x + y)
        } else {
            1.0 - x / (y - x)
        }
    } else if x < 0.0 {
        2.0 - y / (-x - y)
    } else {
        3.0 + x / (x - y)
    }
}

/// A buried interval in pseudo-angle space (`end > start`, `end` may exceed
/// the period) with its endpoint vectors and the sphere causing it.
#[derive(Debug, Clone, Copy)]
struct Buried {
    start: f64,
    end: f64,
    sphere: u32,
    enter: [f64; 2],
    leave: [f64; 2],
}

impl Buried {
    fn new(enter: [f64; 2], leave: [f64; 2], sphere: u32) -> Self {
        let start = pseudo_angle(enter);
        let mut end = pseudo_angle(leave);
        if end <= start {
            end += PSEUDO_PERIOD;
        }
        Self {
            start,
            end,
            sphere,
            enter,
            leave,
        }
    }
}

/// A merged run of buried intervals: spheres and endpoint vectors at both
/// ends.
#[derive(Debug, Clone, Copy)]
struct Merged {
    start: f64,
    end: f64,
    first: u32,
    last: u32,
    enter: [f64; 2],
    leave: [f64; 2],
}

/// Appends to `out` the exposed arcs left by `intervals` and returns `true`,
/// or returns `false` (appending nothing) if the circle is fully covered.
/// `merged` is scratch space.
fn exposed_arcs(intervals: &mut [Buried], merged: &mut Vec<Merged>, out: &mut Vec<Arc>) -> bool {
    if intervals.is_empty() {
        return true;
    }
    intervals.sort_unstable_by(|x, y| {
        x.start
            .total_cmp(&y.start)
            .then(y.end.total_cmp(&x.end))
            .then(x.sphere.cmp(&y.sphere))
    });
    merged.clear();
    for b in intervals.iter() {
        match merged.last_mut() {
            Some(last) if b.start <= last.end => {
                if b.end > last.end {
                    last.end = b.end;
                    last.last = b.sphere;
                    last.leave = b.leave;
                }
            }
            _ => merged.push(Merged {
                start: b.start,
                end: b.end,
                first: b.sphere,
                last: b.sphere,
                enter: b.enter,
                leave: b.leave,
            }),
        }
    }
    // The last run may wrap past the period over the first ones.
    while merged.len() > 1 {
        let first = merged[0];
        let last = merged.last_mut().expect("len > 1");
        if last.end - PSEUDO_PERIOD < first.start {
            break;
        }
        if first.end + PSEUDO_PERIOD > last.end {
            last.end = first.end + PSEUDO_PERIOD;
            last.last = first.last;
            last.leave = first.leave;
        }
        merged.remove(0);
    }
    let m = merged.len();
    let before = out.len();
    for idx in 0..m {
        let cur = merged[idx];
        let next = merged[(idx + 1) % m];
        let next_start = if idx + 1 < m {
            next.start
        } else {
            next.start + PSEUDO_PERIOD
        };
        if next_start - cur.end > PSEUDO_EPS {
            // Real angles only for exposed arcs (rare compared with circles).
            let a0 = cur.leave[1].atan2(cur.leave[0]).rem_euclid(TAU);
            let a1 = next.enter[1].atan2(next.enter[0]);
            let len = (a1 - a0).rem_euclid(TAU);
            if len > ANG_EPS {
                out.push(Arc {
                    start: a0,
                    end: a0 + len,
                    from: cur.last,
                    to: next.first,
                });
            }
        }
    }
    out.len() > before
}

impl Arrangement {
    /// Builds the arrangement of `spheres` (input order) for probe radius
    /// `probe`. Spheres with radius 0 are ignored.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for non-finite coordinates, negative or
    /// non-finite radii, or an invalid probe radius.
    pub(crate) fn build(spheres: &[Sphere], probe: f64) -> Result<Self> {
        Self::build_with(spheres, probe, true, &Control::new())
    }

    /// Like [`Arrangement::build`] but without circles, arcs and triple
    /// points: enough for the SAS and vdW fields.
    pub(crate) fn build_spheres(spheres: &[Sphere], probe: f64) -> Result<Self> {
        Self::build_with(spheres, probe, false, &Control::new())
    }

    /// [`Arrangement::build`] (`with_circles`) or
    /// [`Arrangement::build_spheres`], checking `control` for cancellation
    /// between stages and per sphere.
    ///
    /// # Errors
    /// As [`Arrangement::build`], and [`Error::Cancelled`].
    pub(crate) fn build_with(
        spheres: &[Sphere],
        probe: f64,
        with_circles: bool,
        control: &Control<'_>,
    ) -> Result<Self> {
        if !(probe.is_finite() && probe >= 0.0) {
            return Err(Error::InvalidInput(format!(
                "probe radius must be finite and >= 0, got {probe}"
            )));
        }
        for (i, s) in spheres.iter().enumerate() {
            if !Vec3::from_array(s.center).is_finite() {
                return Err(Error::InvalidInput(format!(
                    "sphere {i} has a non-finite centre {:?}",
                    s.center
                )));
            }
            if !(s.radius.is_finite() && s.radius >= 0.0) {
                return Err(Error::InvalidInput(format!(
                    "sphere {i} has an invalid radius {}",
                    s.radius
                )));
            }
        }
        let mut stats = ArrangementStats {
            input: spheres.len(),
            ..ArrangementStats::default()
        };
        let mut active: Vec<u32> = (0..spheres.len())
            .filter(|&i| spheres[i].radius > 0.0)
            .map(|i| u32::try_from(i).expect("fewer than 2^32 spheres"))
            .collect();
        stats.zero_radius = spheres.len() - active.len();
        let max_expanded = active
            .iter()
            .map(|&i| spheres[i as usize].radius + probe)
            .fold(0.0, f64::max);
        let cell = (2.0 * max_expanded).max(1e-6);

        // Morton order of cells (ties: input order) for locality.
        if let Some(&first) = active.first() {
            let lo = active.iter().fold(
                Vec3::from_array(spheres[first as usize].center),
                |lo, &i| lo.min(Vec3::from_array(spheres[i as usize].center)),
            );
            let key = |i: u32| {
                let p = (Vec3::from_array(spheres[i as usize].center) - lo) / cell;
                morton([p.x, p.y, p.z].map(|v| (v as u64).min(0x1f_ffff) as u32))
            };
            active.sort_by_key(|&i| (key(i), i));
        }
        let centers: Vec<Vec3> = active
            .iter()
            .map(|&i| Vec3::from_array(spheres[i as usize].center))
            .collect();
        let radii: Vec<f64> = active.iter().map(|&i| spheres[i as usize].radius).collect();
        let expanded: Vec<f64> = radii.iter().map(|r| r + probe).collect();
        let n = centers.len();
        let cells = CellGrid::new(&centers, cell);

        // Containment: j contains i if i's expanded sphere is inside j's;
        // exact duplicates keep the one with the lower input index.
        let contains = |j: usize, i: usize| {
            let d = centers[i].dist(centers[j]);
            d + expanded[i] <= expanded[j] + EPS
                && (expanded[j] - expanded[i] > EPS || d > EPS || active[j] < active[i])
        };
        let contained: Vec<bool> = par::map(n, |i| {
            let mut found = false;
            cells.for_each_near(centers[i], expanded[i] + max_expanded, |j| {
                let j = j as usize;
                if j != i && !found && contains(j, i) {
                    found = true;
                }
            });
            found
        });
        stats.contained = contained.iter().filter(|&&c| c).count();
        control.check()?;

        // Both stages below run over chunks of spheres with flat per-chunk
        // arrays (`par::chunk_len`), concatenated in sphere order.
        let chunk = par::chunk_len(n);
        let chunks = n.div_ceil(chunk);
        let range = |c: usize| par::chunk_range(n, chunk, c);
        let parts: Vec<Csr> = par::map(chunks, |c| {
            let mut part = Csr::with_capacity(chunk, 0);
            for i in range(c) {
                let start = part.items.len();
                if !contained[i] {
                    cells.for_each_near(centers[i], expanded[i] + max_expanded, |j| {
                        let ju = j as usize;
                        if ju != i
                            && !contained[ju]
                            && centers[i].dist(centers[ju]) < expanded[i] + expanded[ju] - EPS
                        {
                            part.items.push(j);
                        }
                    });
                    part.items[start..].sort_unstable();
                }
                part.close_row();
            }
            part
        });
        let mut neighbors = Csr::with_capacity(n, parts.iter().map(Csr::total).sum());
        for part in parts {
            neighbors.append(&part);
        }
        debug_assert_eq!(neighbors.rows(), n);
        stats.neighbour_pairs = neighbors.total() / 2;
        control.check()?;

        // Circles (i, j > i), their exposed arcs and the triple points they
        // carry, per sphere i in parallel; concatenated in sphere order.
        let geometry = SphereGeometry {
            centers: &centers,
            expanded: &expanded,
            neighbors: &neighbors,
        };
        let parts: Vec<Part> = par::map_init(chunks, Scratch::default, |scratch, c| {
            let mut part = Part::default();
            if with_circles {
                for i in range(c) {
                    if !contained[i] && !control.is_cancelled() {
                        geometry.circles_of(i, scratch, &mut part);
                    }
                }
            }
            part
        });
        control.check()?;
        let mut exposed: Vec<bool> = (0..n)
            .map(|i| !contained[i] && (!with_circles || neighbors.row(i).is_empty()))
            .collect();
        // Exact capacities: these arrays live as long as the field.
        let total = |f: fn(&Part) -> usize| parts.iter().map(f).sum::<usize>();
        let mut circles = Vec::with_capacity(total(|p| p.circles.len()));
        let mut arcs = Vec::with_capacity(total(|p| p.arcs.len()));
        let mut circle_blockers =
            Csr::with_capacity(circles.capacity(), total(|p| p.blockers.total()));
        let mut vertices = Vec::with_capacity(total(|p| p.vertices.len()));
        for part in parts {
            let base = u32::try_from(arcs.len()).expect("fewer than 2^32 arcs");
            for mut circle in part.circles {
                exposed[circle.i as usize] = true;
                exposed[circle.j as usize] = true;
                circle.arcs = (circle.arcs.0 + base, circle.arcs.1 + base);
                circles.push(circle);
            }
            circle_blockers.append(&part.blockers);
            arcs.extend(part.arcs);
            vertices.extend(part.vertices);
        }
        debug_assert_eq!(circle_blockers.rows(), circles.len());

        let status: Vec<Status> = (0..n)
            .map(|i| {
                if contained[i] {
                    Status::Contained
                } else if exposed[i] {
                    Status::Exposed
                } else {
                    Status::Buried
                }
            })
            .collect();
        stats.exposed = status.iter().filter(|&&s| s == Status::Exposed).count();
        stats.exposed_circles = circles.len();
        stats.arcs = arcs.len();
        stats.vertices = vertices.len();
        Ok(Self {
            probe,
            centers,
            radii,
            expanded,
            orig: active,
            status,
            neighbors,
            circles,
            arcs,
            circle_blockers,
            vertices,
            cells,
            max_expanded,
            stats,
        })
    }
}

/// Read-only inputs of the per-sphere circle stage.
struct SphereGeometry<'a> {
    centers: &'a [Vec3],
    expanded: &'a [f64],
    neighbors: &'a Csr,
}

/// Per-thread scratch buffers of the circle stage.
#[derive(Default)]
struct Scratch {
    intervals: Vec<Buried>,
    merged: Vec<Merged>,
    blockers: Vec<u32>,
}

/// Circles of a chunk of spheres with arcs (ranges relative to `arcs`),
/// blocker lists (one row per circle) and triple points.
#[derive(Default)]
struct Part {
    circles: Vec<Circle>,
    arcs: Vec<Arc>,
    blockers: Csr,
    vertices: Vec<SasVertex>,
}

impl SphereGeometry<'_> {
    /// Appends the circles of sphere `i` (with `j > i`) to `out`.
    fn circles_of(&self, i: usize, scratch: &mut Scratch, out: &mut Part) {
        let (centers, expanded) = (self.centers, self.expanded);
        for &j in self.neighbors.row(i).iter().filter(|&&j| j as usize > i) {
            let ju = j as usize;
            let d = centers[i].dist(centers[ju]);
            let axis = (centers[ju] - centers[i]) / d;
            let t = (d * d + expanded[i] * expanded[i] - expanded[ju] * expanded[ju]) / (2.0 * d);
            let rho2 = expanded[i] * expanded[i] - t * t;
            if rho2 <= 0.0 {
                continue;
            }
            let (e1, e2) = orthonormal_basis(axis);
            let mut circle = Circle {
                i: i as u32,
                j,
                center: centers[i] + axis * t,
                axis,
                e1,
                radius: rho2.sqrt(),
                arcs: (0, 0),
            };
            scratch.intervals.clear();
            scratch.blockers.clear();
            let mut buried = false;
            for &k in sorted_intersection(self.neighbors.row(i), self.neighbors.row(ju)) {
                match cut(&circle, centers[k as usize], expanded[k as usize]) {
                    Cut::Empty => {}
                    Cut::Full => {
                        buried = true;
                        break;
                    }
                    Cut::Interval { enter, leave } => {
                        scratch.intervals.push(Buried::new(enter, leave, k));
                        scratch.blockers.push(k);
                    }
                }
            }
            if buried {
                continue;
            }
            let first = out.arcs.len() as u32;
            if !exposed_arcs(&mut scratch.intervals, &mut scratch.merged, &mut out.arcs) {
                continue;
            }
            circle.arcs = (first, out.arcs.len() as u32);
            // Triple points: each triple i < j < k comes from circle (i, j)
            // and a blocker k > j; it is kept if no other neighbour of i
            // contains it (any sphere containing a point of sphere i
            // intersects sphere i).
            for &k in scratch.blockers.iter().filter(|&&k| k > j) {
                let Cut::Interval { enter, leave } =
                    cut(&circle, centers[k as usize], expanded[k as usize])
                else {
                    continue;
                };
                for [c, s] in [leave, enter] {
                    let q = circle.center + (circle.e1 * c + e2 * s) * circle.radius;
                    let free = self.neighbors.row(i).iter().all(|&l| {
                        l == j
                            || l == k
                            || q.dist(centers[l as usize]) >= expanded[l as usize] - EPS
                    });
                    if free {
                        out.vertices.push(SasVertex {
                            position: q,
                            atoms: [i as u32, j, k],
                        });
                    }
                }
            }
            out.circles.push(circle);
            out.blockers.push_row(scratch.blockers.iter().copied());
        }
    }
}

/// Elements common to two ascending slices.
fn sorted_intersection<'a>(a: &'a [u32], b: &'a [u32]) -> impl Iterator<Item = &'a u32> + 'a {
    let mut j = 0;
    a.iter().filter(move |&&x| {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        j < b.len() && b[j] == x
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use sesmesh_testkit::clusters::{
        Cluster, Rng, cage, cluster_strategy, collinear, cubic_lattice, random_box, random_chain,
        with_duplicates,
    };
    use sesmesh_testkit::oracle::{Oracle, fibonacci_sphere};

    fn spheres(c: &Cluster) -> Vec<Sphere> {
        c.centers
            .iter()
            .zip(&c.radii)
            .map(|(&center, &radius)| Sphere { center, radius })
            .collect()
    }

    fn pos(v: Vec3) -> [f64; 3] {
        v.to_array()
    }

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
    }

    /// Compares an arrangement with the brute-force oracle and with dense
    /// sampling. Near-degenerate features (margins below `slack`) are not
    /// required to match: there the decision is tolerance-dependent and the
    /// field is unaffected to O(tolerance).
    fn check(
        c: &Cluster,
        samples: usize,
        circle_samples: usize,
    ) -> std::result::Result<(), String> {
        let slack = 1e-6;
        let a = Arrangement::build(&spheres(c), c.probe).map_err(|e| e.to_string())?;
        let o = Oracle::new(&c.centers, &c.radii, c.probe);
        let mut internal = vec![usize::MAX; c.centers.len()];
        for (k, &orig) in a.orig.iter().enumerate() {
            internal[orig as usize] = k;
        }

        // Neighbours: every intersecting non-contained pair, symmetric.
        for i in 0..a.centers.len() {
            for &j in a.neighbors.row(i) {
                if !a.neighbors.row(j as usize).contains(&(i as u32)) {
                    return Err(format!("neighbour lists not symmetric for {i}, {j}"));
                }
            }
        }

        // Vertices against the oracle's exhaustive triple enumeration.
        let lib: Vec<[f64; 3]> = a.vertices.iter().map(|v| pos(v.position)).collect();
        for v in o.vertices().iter().filter(|v| v.margin > slack) {
            if !lib.iter().any(|&p| close(p, v.position, 1e-7)) {
                return Err(format!("missing vertex {:?} at {:?}", v.atoms, v.position));
            }
        }
        for v in &a.vertices {
            let atoms = v.atoms.map(|k| a.orig[k as usize] as usize);
            let p = pos(v.position);
            let margin = o.margin(p, &atoms);
            if margin < -slack {
                return Err(format!(
                    "vertex {atoms:?} at {p:?} is buried (margin {margin})"
                ));
            }
            if margin > slack && !o.vertices().iter().any(|w| close(w.position, p, 1e-7)) {
                return Err(format!("spurious vertex {atoms:?} at {p:?}"));
            }
        }

        // Exposed circles: arc midpoints (or samples of full circles) must be
        // accessible.
        for (ci, circle) in a.circles.iter().enumerate() {
            let (i, j) = (
                a.orig[circle.i as usize] as usize,
                a.orig[circle.j as usize] as usize,
            );
            let probes: Vec<Vec3> = if circle.is_full() {
                (0..16)
                    .map(|k| circle.point(f64::from(k) * TAU / 16.0))
                    .collect()
            } else {
                a.arcs[circle.arcs.0 as usize..circle.arcs.1 as usize]
                    .iter()
                    .map(|arc| circle.point(0.5 * (arc.start + arc.end)))
                    .collect()
            };
            for q in probes {
                if o.margin(pos(q), &[i, j]) < -slack {
                    return Err(format!(
                        "circle {ci} ({i},{j}) has a buried exposed-arc point"
                    ));
                }
            }
        }
        // Circles the library dropped as buried must have no exposed sample.
        for oc in o.circles() {
            let (ii, jj) = (internal[oc.i], internal[oc.j]);
            let in_lib = a
                .circles
                .iter()
                .any(|c| (c.i as usize, c.j as usize) == (ii.min(jj), ii.max(jj)));
            if in_lib {
                continue;
            }
            for k in 0..circle_samples {
                let q = oc.point(k as f64 * TAU / circle_samples as f64);
                if o.margin(q, &[oc.i, oc.j]) > slack {
                    return Err(format!("circle ({},{}) is exposed but missing", oc.i, oc.j));
                }
            }
        }

        // Sphere exposure against dense sampling.
        for (orig, &center) in c.centers.iter().enumerate() {
            if c.radii[orig] <= 0.0 {
                continue;
            }
            let k = internal[orig];
            let r = a.expanded[k];
            let sampled_exposed = fibonacci_sphere(center, r, samples)
                .into_iter()
                .any(|q| o.margin(q, &[orig]) > slack);
            if sampled_exposed && a.status[k] != Status::Exposed {
                return Err(format!(
                    "sphere {orig} has exposed samples but is {:?}",
                    a.status[k]
                ));
            }
        }
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]
        #[test]
        fn arrangement_matches_oracle(c in cluster_strategy(24)) {
            check(&c, 600, 720).map_err(TestCaseError::fail)?;
        }
    }

    #[test]
    fn csr_retain_rows_compacts_in_place() {
        let rows: [&[u32]; 5] = [&[1, 2], &[], &[3], &[4, 5, 6], &[7]];
        let mut csr = Csr::with_capacity(rows.len(), 7);
        for row in rows {
            csr.push_row(row.iter().copied());
        }
        csr.retain_rows(|i| i != 0 && i != 3);
        let got: Vec<&[u32]> = (0..rows.len()).map(|i| csr.row(i)).collect();
        assert_eq!(got, [&[][..], &[], &[3], &[], &[7]]);
        assert_eq!(csr.total(), 2);
        csr.retain_rows(|_| false);
        assert!((0..rows.len()).all(|i| csr.row(i).is_empty()));
    }

    #[test]
    fn circle_e2_matches_the_basis() {
        for axis in [
            Vec3::X,
            Vec3::new(0.3, -0.8, 0.52).normalized(),
            Vec3::Z * -1.0,
        ] {
            let (e1, e2) = orthonormal_basis(axis);
            let c = Circle {
                i: 0,
                j: 1,
                center: Vec3::new(0.0, 0.0, 0.0),
                axis,
                e1,
                radius: 1.0,
                arcs: (0, 0),
            };
            assert_eq!(c.e2(), e2);
        }
    }

    /// The Phase 2 exit criterion: 10 000 random clusters (release build).
    #[test]
    #[ignore = "slow; run with --release -- --include-ignored"]
    fn arrangement_matches_oracle_10k() {
        let mut rng = Rng::new(2026);
        for case in 0..10_000 {
            let n = 1 + rng.below(30);
            let c = match case % 4 {
                0 | 1 => random_chain(&mut rng, n),
                2 => random_box(&mut rng, n, 2.0 + (n as f64).cbrt() * 3.0),
                _ => with_duplicates(random_chain(&mut rng, n), 3),
            };
            if let Err(e) = check(&c, 800, 720) {
                panic!("case {case}: {e}\n{c:?}");
            }
        }
    }

    #[test]
    fn symmetric_configurations_are_deterministic_and_consistent() {
        for c in [
            cubic_lattice(3, 3.0, 1.7, 1.4),
            cubic_lattice(3, 2.4, 1.6, 0.4),
            cage(3.5, 1.6, 1.4),
            cage(4.0, 1.7, 0.6),
            collinear(8, 2.0, 1.5, 1.0),
            with_duplicates(cage(3.5, 1.6, 1.4), 4),
        ] {
            let x = Arrangement::build(&spheres(&c), c.probe).unwrap();
            let y = Arrangement::build(&spheres(&c), c.probe).unwrap();
            assert_eq!(x.stats, y.stats);
            assert_eq!(x.vertices, y.vertices);
            check(&c, 2000, 720).unwrap();
        }
    }

    #[test]
    fn simple_cases() {
        let one = Arrangement::build(
            &[Sphere {
                center: [0.0; 3],
                radius: 1.7,
            }],
            1.4,
        )
        .unwrap();
        assert_eq!(one.status, [Status::Exposed]);
        assert!(one.circles.is_empty() && one.vertices.is_empty());

        let two = Arrangement::build(
            &[
                Sphere {
                    center: [0.0; 3],
                    radius: 1.7,
                },
                Sphere {
                    center: [3.0, 0.0, 0.0],
                    radius: 1.7,
                },
            ],
            1.4,
        )
        .unwrap();
        assert_eq!(two.circles.len(), 1);
        assert!(two.circles[0].is_full());
        assert_eq!(two.status, [Status::Exposed, Status::Exposed]);

        // Inner sphere and exact duplicate are contained; zero radius ignored.
        let nested = Arrangement::build(
            &[
                Sphere {
                    center: [0.0; 3],
                    radius: 2.0,
                },
                Sphere {
                    center: [0.2, 0.0, 0.0],
                    radius: 0.5,
                },
                Sphere {
                    center: [0.0; 3],
                    radius: 2.0,
                },
                Sphere {
                    center: [9.0, 0.0, 0.0],
                    radius: 0.0,
                },
            ],
            1.4,
        )
        .unwrap();
        assert_eq!(nested.stats.zero_radius, 1);
        assert_eq!(nested.stats.contained, 2);
        assert_eq!(nested.stats.exposed, 1);
        let survivor = nested
            .status
            .iter()
            .position(|s| *s == Status::Exposed)
            .unwrap();
        assert_eq!(nested.orig[survivor], 0, "the earlier duplicate survives");

        let empty = Arrangement::build(&[], 1.4).unwrap();
        assert_eq!(empty.stats, ArrangementStats::default());
    }

    #[test]
    fn invalid_input() {
        let bad = |s: Sphere, p: f64| Arrangement::build(&[s], p).is_err();
        assert!(bad(
            Sphere {
                center: [f64::NAN, 0.0, 0.0],
                radius: 1.0
            },
            1.4
        ));
        assert!(bad(
            Sphere {
                center: [0.0; 3],
                radius: -1.0
            },
            1.4
        ));
        assert!(bad(
            Sphere {
                center: [0.0; 3],
                radius: f64::INFINITY
            },
            1.4
        ));
        assert!(bad(
            Sphere {
                center: [0.0; 3],
                radius: 1.0
            },
            -0.1
        ));
        assert!(!bad(
            Sphere {
                center: [0.0; 3],
                radius: 1.0
            },
            0.0
        ));
    }

    /// Exposed arcs for buried intervals given as `(start, end, sphere)`
    /// angles in radians.
    fn union(iv: Vec<(f64, f64, u32)>) -> Option<Vec<Arc>> {
        let mut buried: Vec<Buried> = iv
            .into_iter()
            .map(|(s, e, k)| Buried::new([s.cos(), s.sin()], [e.cos(), e.sin()], k))
            .collect();
        let mut out = Vec::new();
        exposed_arcs(&mut buried, &mut Vec::new(), &mut out).then_some(out)
    }

    #[test]
    fn pseudo_angle_is_monotone_and_periodic() {
        let mut prev = -1.0;
        for k in 0..10_000 {
            let t = f64::from(k) * TAU / 10_000.0;
            let p = pseudo_angle([t.cos(), t.sin()]);
            assert!(p > prev && (0.0..4.0).contains(&p), "t = {t}");
            prev = p;
        }
    }

    #[test]
    fn arc_union() {
        // Two overlapping intervals and one disjoint: two gaps.
        let arcs = union(vec![(0.5, 1.5, 1), (1.0, 2.0, 2), (3.0, 4.0, 3)]).unwrap();
        assert_eq!(arcs.len(), 2);
        assert_eq!((arcs[0].from, arcs[0].to), (2, 3));
        assert!((arcs[0].start - 2.0).abs() < 1e-12 && (arcs[0].end - 3.0).abs() < 1e-12);
        assert!((arcs[1].start - 4.0).abs() < 1e-12 && (arcs[1].end - (0.5 + TAU)).abs() < 1e-12);
        // An interval wrapping past 2π swallowing the first one.
        let arcs = union(vec![(0.1, 0.4, 1), (6.0, 6.0 + 1.0, 2)]).unwrap();
        assert_eq!(arcs.len(), 1);
        assert!((arcs[0].start - (1.0 - (TAU - 6.0))).abs() < 1e-12);
        // Full coverage.
        assert!(union(vec![(0.0, 4.0, 1), (3.5, 3.5 + 3.0, 2)]).is_none());
        assert_eq!(union(Vec::new()), Some(Vec::new()));
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn crambin_matches_oracle() {
        use crate::io::{ReadOptions, read_path};
        use crate::radii::{RadiusOptions, assign};
        use crate::{Diagnostics, Selection};
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/fixtures/1CRN.cif.gz"
        );
        let s = read_path(path, &ReadOptions::default()).unwrap();
        let mut d = Diagnostics::new();
        let atoms = s.select(&Selection::default(), &mut d).unwrap();
        let radii = assign(&atoms, &RadiusOptions::default(), &mut d).unwrap();
        let c = Cluster {
            centers: atoms.positions(),
            radii,
            probe: 1.4,
        };
        check(&c, 300, 48).unwrap();
        let a = Arrangement::build(&spheres(&c), 1.4).unwrap();
        assert_eq!(a.stats.input, 327);
        assert!(a.stats.vertices > 0 && a.stats.exposed < 327);
    }
}
