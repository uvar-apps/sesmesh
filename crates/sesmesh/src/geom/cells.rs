//! Uniform cell lists in CSR form (Design §6.2) and Morton ordering.

use super::vec::Vec3;

/// Points binned into a uniform grid. Items of a cell are stored
/// contiguously, in ascending point index (deterministic).
#[derive(Debug, Clone)]
pub(crate) struct CellGrid {
    origin: Vec3,
    cell: f64,
    dims: [usize; 3],
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl CellGrid {
    /// Bins `points` into cells of edge `cell`. If the bounding box would
    /// need more than `max(65 536, 8 × points)` cells (very sparse inputs),
    /// the cell edge is doubled until it fits, so memory stays bounded.
    ///
    /// # Panics
    /// If `cell` is not positive and finite.
    pub(crate) fn new(points: &[Vec3], cell: f64) -> Self {
        assert!(cell > 0.0 && cell.is_finite(), "cell size must be positive");
        let Some(&first) = points.first() else {
            return Self {
                origin: Vec3::default(),
                cell,
                dims: [1, 1, 1],
                starts: vec![0, 0],
                items: Vec::new(),
            };
        };
        let (lo, hi) = points
            .iter()
            .fold((first, first), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        let extent = hi - lo;
        let max_cells = (8 * points.len()).max(1 << 16);
        let mut cell = cell;
        let dims = loop {
            let d = [extent.x, extent.y, extent.z].map(|e| (e / cell).floor() as usize + 1);
            let total = d[0].checked_mul(d[1]).and_then(|v| v.checked_mul(d[2]));
            match total {
                Some(t) if t <= max_cells => break d,
                _ => cell *= 2.0,
            }
        };
        let mut grid = Self {
            origin: lo,
            cell,
            dims,
            starts: Vec::new(),
            items: Vec::new(),
        };
        let keys: Vec<usize> = points.iter().map(|&p| grid.flat(grid.coord(p))).collect();
        let total = dims[0] * dims[1] * dims[2];
        let mut starts = vec![0u32; total + 1];
        for &k in &keys {
            starts[k + 1] += 1;
        }
        for i in 0..total {
            starts[i + 1] += starts[i];
        }
        let mut cursor = starts.clone();
        let mut items = vec![0u32; points.len()];
        for (i, &k) in keys.iter().enumerate() {
            items[cursor[k] as usize] = u32::try_from(i).expect("fewer than 2^32 points");
            cursor[k] += 1;
        }
        grid.starts = starts;
        grid.items = items;
        grid
    }

    /// Cell edge actually used.
    pub(crate) fn cell(&self) -> f64 {
        self.cell
    }

    fn coord(&self, p: Vec3) -> [usize; 3] {
        let rel = [
            p.x - self.origin.x,
            p.y - self.origin.y,
            p.z - self.origin.z,
        ];
        [0, 1, 2].map(|a| {
            let c = (rel[a] / self.cell).floor();
            if c <= 0.0 {
                0
            } else {
                (c as usize).min(self.dims[a] - 1)
            }
        })
    }

    fn flat(&self, c: [usize; 3]) -> usize {
        (c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]
    }

    /// Calls `f` for every point in the cells overlapping the axis-aligned box
    /// around the ball (a superset of the points within `radius`).
    pub(crate) fn for_each_near(&self, center: Vec3, radius: f64, mut f: impl FnMut(u32)) {
        let r = Vec3::new(radius, radius, radius);
        let lo = self.coord(center - r);
        let hi = self.coord(center + r);
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                let row = self.flat([0, y, z]);
                let (a, b) = (
                    self.starts[row + lo[0]] as usize,
                    self.starts[row + hi[0] + 1] as usize,
                );
                for &i in &self.items[a..b] {
                    f(i);
                }
            }
        }
    }
}

/// Spreads the low 21 bits of `v` so that two zero bits separate each bit.
fn spread(v: u32) -> u64 {
    let mut x = u64::from(v) & 0x1f_ffff;
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Morton (Z-order) code of 21-bit cell coordinates.
pub(crate) fn morton(c: [u32; 3]) -> u64 {
    spread(c[0]) | (spread(c[1]) << 1) | (spread(c[2]) << 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn neighbourhood_is_a_superset(
            pts in prop::collection::vec((-20.0..20.0f64, -20.0..20.0f64, -5.0..5.0f64), 0..200),
            q in (-25.0..25.0f64, -25.0..25.0f64, -8.0..8.0f64),
            radius in 0.0..9.0f64,
            cell in 0.5..6.0f64,
        ) {
            let pts: Vec<Vec3> = pts.into_iter().map(|(x, y, z)| Vec3::new(x, y, z)).collect();
            let q = Vec3::new(q.0, q.1, q.2);
            let grid = CellGrid::new(&pts, cell);
            let mut found = Vec::new();
            grid.for_each_near(q, radius, |i| found.push(i as usize));
            let unique = found.len();
            found.sort_unstable();
            found.dedup();
            prop_assert_eq!(unique, found.len(), "a point was visited twice");
            for (i, p) in pts.iter().enumerate() {
                if p.dist(q) <= radius {
                    prop_assert!(found.binary_search(&i).is_ok(), "missed point {}", i);
                }
            }
        }
    }

    #[test]
    fn sparse_inputs_stay_bounded() {
        let pts = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1e7, -1e7, 1e7)];
        let grid = CellGrid::new(&pts, 1.0);
        assert!(grid.starts.len() <= (1 << 16) + 1);
        assert!(grid.cell() > 1.0);
        let mut seen = Vec::new();
        grid.for_each_near(Vec3::new(1e7, -1e7, 1e7), 0.5, |i| seen.push(i));
        assert!(seen.contains(&1));
        let empty = CellGrid::new(&[], 2.0);
        empty.for_each_near(Vec3::default(), 10.0, |_| panic!("no points"));
    }

    #[test]
    fn morton_interleaves() {
        assert_eq!(morton([1, 0, 0]), 0b001);
        assert_eq!(morton([0, 1, 0]), 0b010);
        assert_eq!(morton([0, 0, 1]), 0b100);
        assert_eq!(morton([3, 0, 0]), 0b001_001);
        assert_eq!(morton([0x1f_ffff, 0x1f_ffff, 0x1f_ffff]), (1u64 << 63) - 1);
    }
}
