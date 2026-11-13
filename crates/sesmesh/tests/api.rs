// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public API guarantees (Plan T10.3): thread safety of the main types.

use sesmesh::export::obj::ObjOptions;
use sesmesh::export::ply::PlyOptions;
use sesmesh::io::ReadOptions;
use sesmesh::radii::{RadiusOptions, RadiusRules};
use sesmesh::{
    Control, Diagnostics, Error, Mesh, SesField, Sphere, Structure, SurfaceEngine, SurfaceOutput,
    SurfaceParams,
};

fn send_sync<T: Send + Sync>() {}

#[test]
fn main_types_are_send_and_sync() {
    send_sync::<Sphere>();
    send_sync::<SurfaceParams>();
    send_sync::<SurfaceEngine>();
    send_sync::<SurfaceOutput>();
    send_sync::<Mesh>();
    send_sync::<SesField>();
    send_sync::<Structure>();
    send_sync::<Diagnostics>();
    send_sync::<Error>();
    send_sync::<Control<'static>>();
    send_sync::<ReadOptions>();
    send_sync::<RadiusOptions>();
    send_sync::<RadiusRules>();
    send_sync::<ObjOptions>();
    send_sync::<PlyOptions>();
}

#[test]
fn errors_are_std_errors() {
    fn std_error<E: std::error::Error + Send + Sync + 'static>() {}
    std_error::<Error>();
}
