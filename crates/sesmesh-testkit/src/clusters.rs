//! Deterministic sphere clusters for tests: random (seeded) and symmetric
//! configurations that stress degenerate geometry, plus `proptest`
//! strategies.

use proptest::prelude::*;

use crate::V3;
use crate::v::{add, mul, normalize};

/// Spheres plus a probe radius.
#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    /// Centres, Å.
    pub centers: Vec<V3>,
    /// vdW radii, Å.
    pub radii: Vec<f64>,
    /// Probe radius, Å.
    pub probe: f64,
}

/// SplitMix64 generator (deterministic, dependency-free).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform random unit vector.
    pub fn unit(&mut self) -> V3 {
        loop {
            let v = [
                self.uniform(-1.0, 1.0),
                self.uniform(-1.0, 1.0),
                self.uniform(-1.0, 1.0),
            ];
            let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (0.01..=1.0).contains(&n2) {
                return normalize(v);
            }
        }
    }
}

/// Protein-like random walk: bonds 1.3–1.6 Å from a random earlier atom,
/// radii 1.2–2.0 Å, probe 0.6–1.8 Å. Produces many overlapping spheres,
/// buried atoms, saddles and concave pockets.
pub fn random_chain(rng: &mut Rng, n: usize) -> Cluster {
    let mut centers = vec![[0.0; 3]];
    while centers.len() < n {
        let from = centers[rng.below(centers.len())];
        let step = rng.uniform(1.3, 1.6);
        centers.push(add(from, mul(rng.unit(), step)));
    }
    let radii = (0..n).map(|_| rng.uniform(1.2, 2.0)).collect();
    Cluster {
        centers,
        radii,
        probe: rng.uniform(0.6, 1.8),
    }
}

/// Spheres scattered uniformly in a cube of side `side`.
pub fn random_box(rng: &mut Rng, n: usize, side: f64) -> Cluster {
    let centers = (0..n)
        .map(|_| {
            [
                rng.uniform(0.0, side),
                rng.uniform(0.0, side),
                rng.uniform(0.0, side),
            ]
        })
        .collect();
    let radii = (0..n).map(|_| rng.uniform(1.0, 2.2)).collect();
    Cluster {
        centers,
        radii,
        probe: rng.uniform(0.3, 2.0),
    }
}

/// `n × n × n` cubic lattice (highly symmetric: many exactly equal
/// distances and four-sphere coincidences).
pub fn cubic_lattice(n: usize, spacing: f64, radius: f64, probe: f64) -> Cluster {
    let mut centers = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                centers.push([i as f64 * spacing, j as f64 * spacing, k as f64 * spacing]);
            }
        }
    }
    let radii = vec![radius; centers.len()];
    Cluster {
        centers,
        radii,
        probe,
    }
}

/// Vertices of a regular icosahedron (12) and dodecahedron (20), scaled to
/// `cage_radius`: a symmetric cage.
pub fn cage(cage_radius: f64, radius: f64, probe: f64) -> Cluster {
    let g = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut pts: Vec<V3> = Vec::new();
    for &(a, b) in &[(1.0, g), (-1.0, g), (1.0, -g), (-1.0, -g)] {
        pts.push([0.0, a, b]);
        pts.push([a, b, 0.0]);
        pts.push([b, 0.0, a]);
    }
    for &x in &[-1.0, 1.0] {
        for &y in &[-1.0, 1.0] {
            for &z in &[-1.0, 1.0] {
                pts.push([x, y, z]);
            }
        }
    }
    let ig = 1.0 / g;
    for &(a, b) in &[(ig, g), (-ig, g), (ig, -g), (-ig, -g)] {
        pts.push([0.0, a, b]);
        pts.push([a, b, 0.0]);
        pts.push([b, 0.0, a]);
    }
    let centers: Vec<V3> = pts
        .into_iter()
        .map(|p| mul(normalize(p), cage_radius))
        .collect();
    let radii = vec![radius; centers.len()];
    Cluster {
        centers,
        radii,
        probe,
    }
}

/// Collinear chain along x.
pub fn collinear(n: usize, spacing: f64, radius: f64, probe: f64) -> Cluster {
    Cluster {
        centers: (0..n).map(|i| [i as f64 * spacing, 0.0, 0.0]).collect(),
        radii: vec![radius; n],
        probe,
    }
}

/// `cluster` with every `every`-th sphere duplicated exactly and one sphere
/// placed inside another.
pub fn with_duplicates(mut cluster: Cluster, every: usize) -> Cluster {
    let n = cluster.centers.len();
    for i in (0..n).step_by(every.max(1)) {
        cluster.centers.push(cluster.centers[i]);
        cluster.radii.push(cluster.radii[i]);
    }
    if n > 0 {
        cluster
            .centers
            .push(add(cluster.centers[0], [0.1, 0.0, 0.0]));
        cluster.radii.push(cluster.radii[0] * 0.3);
    }
    cluster
}

/// `proptest` strategy: random chains and boxes of 1..=`max_n` spheres.
pub fn cluster_strategy(max_n: usize) -> impl Strategy<Value = Cluster> {
    (any::<u64>(), 1..=max_n, 0..4u8).prop_map(|(seed, n, kind)| {
        let mut rng = Rng::new(seed);
        match kind {
            0 | 1 => random_chain(&mut rng, n),
            2 => random_box(&mut rng, n, 2.0 + (n as f64).cbrt() * 3.0),
            _ => with_duplicates(random_chain(&mut rng, n), 3),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = random_chain(&mut Rng::new(42), 20);
        let b = random_chain(&mut Rng::new(42), 20);
        assert_eq!(a, b);
        assert_eq!(cage(3.0, 1.0, 1.0).centers.len(), 32);
        assert_eq!(cubic_lattice(3, 3.0, 1.7, 1.4).centers.len(), 27);
        let d = with_duplicates(collinear(6, 2.0, 1.5, 1.0), 2);
        assert_eq!(d.centers.len(), 6 + 3 + 1);
    }
}
