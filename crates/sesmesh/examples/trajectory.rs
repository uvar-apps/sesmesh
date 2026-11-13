// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Many frames with one reusable engine, with progress and cancellation.
//!
//! ```text
//! cargo run --release -p sesmesh --example trajectory
//! ```

use std::sync::atomic::AtomicBool;

use sesmesh::{Control, Progress, Sphere, Stage, SurfaceEngine, SurfaceParams};

fn main() -> sesmesh::Result<()> {
    // Parameters are validated once; per-thread buffers are kept between
    // frames.
    let mut engine = SurfaceEngine::new(SurfaceParams::ses(1.4).with_spacing(0.4))?;
    let cancel = AtomicBool::new(false);
    let progress = |p: Progress| {
        if p.stage == Stage::Mesh && p.done == p.total {
            eprint!(".");
        }
    };
    let control = Control::new().with_cancel(&cancel).with_progress(&progress);
    for frame in 0..10 {
        // Two atoms moving apart: one component, then two.
        let d = 2.5 + 0.6 * f64::from(frame);
        let spheres = [Sphere::new([0.0; 3], 1.7), Sphere::new([d, 0.0, 0.0], 1.7)];
        let out = engine.compute_with(&spheres, &control)?;
        println!(
            "frame {frame}: distance {d:.1} Å → {} component(s), area {:.2} Å²",
            out.stats.components, out.stats.area
        );
    }
    eprintln!();
    Ok(())
}
