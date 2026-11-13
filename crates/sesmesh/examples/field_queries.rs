// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Point queries on the exact surface field: value, closest feature, normal.
//!
//! ```text
//! cargo run --release -p sesmesh --example field_queries
//! ```

use sesmesh::field::Patch;
use sesmesh::{SesField, Sphere, SurfaceKind};

fn main() -> sesmesh::Result<()> {
    let spheres = [
        Sphere::new([0.0, 0.0, 0.0], 1.7),
        Sphere::new([3.0, 0.0, 0.0], 1.7),
    ];
    let field = SesField::new(&spheres, 1.4, SurfaceKind::Ses)?;
    // Negative inside the molecule, 0 on the SES, positive in the solvent.
    for point in [
        [-1.7, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [1.5, 1.5, 0.0],
        [0.0, 3.0, 0.0],
    ] {
        let s = field.sample(point);
        let patch = match s.patch {
            Patch::Convex { atom } => format!("convex (atom {atom})"),
            Patch::Saddle { atoms } => format!("saddle (atoms {atoms:?})"),
            Patch::Concave { atoms } => format!("concave (atoms {atoms:?})"),
            _ => "solvent".to_string(),
        };
        println!(
            "{point:?}: F = {:+.4} Å, {patch}, normal {:?}",
            s.value,
            s.normal.map(|n| n.map(|c| (c * 1000.0).round() / 1000.0))
        );
    }
    Ok(())
}
