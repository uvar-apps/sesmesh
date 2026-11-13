// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plain spheres → mesh (no file I/O), with the three surface kinds.
//!
//! ```text
//! cargo run --release -p sesmesh --example spheres_to_mesh
//! ```

use sesmesh::{Sphere, SurfaceParams, compute_surface, mesh};

fn main() -> sesmesh::Result<()> {
    // A small "molecule": four overlapping spheres.
    let spheres = [
        Sphere::new([0.0, 0.0, 0.0], 1.7),
        Sphere::new([1.5, 0.0, 0.0], 1.52),
        Sphere::new([2.3, 1.2, 0.0], 1.55),
        Sphere::new([0.5, -1.4, 0.6], 1.8),
    ];
    for (name, params) in [
        ("SES", SurfaceParams::ses(1.4)),
        ("SAS", SurfaceParams::sas(1.4)),
        ("vdW", SurfaceParams::vdw()),
    ] {
        let out = compute_surface(&spheres, &params.with_spacing(0.2))?;
        // Every mesh is closed, consistently oriented and edge-manifold.
        assert!(mesh::validate(&out.mesh).is_ok());
        println!(
            "{name}: {:6} vertices {:6} triangles  area {:7.2} Å²  volume {:7.2} Å³  genus {}",
            out.stats.vertices,
            out.stats.triangles,
            out.stats.area,
            out.stats.volume,
            out.components[0].genus()
        );
    }
    Ok(())
}
