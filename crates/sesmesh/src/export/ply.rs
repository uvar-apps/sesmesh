// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Binary PLY output (Spec FR-OUT-3, Design §10.2): little-endian, with
//! normals, optional colours and the per-vertex attributes.
//!
//! ```text
//! ply
//! format binary_little_endian 1.0
//! comment sesmesh 1.0.0
//! element vertex N
//! property float x            (y, z)
//! property float nx           (ny, nz; when the mesh has normals)
//! property uchar red          (green, blue; with vertex colours)
//! property uint atom          nearest atom, 0-based input index
//! property uchar patch        0 convex, 1 saddle, 2 concave
//! property uint component     (when the mesh has component ids)
//! element face M
//! property list uchar uint vertex_indices
//! end_header
//! ```

use std::io::Write;
use std::path::Path;

use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// PLY writer options.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct PlyOptions {
    /// Header comments (without `comment `), after `comment sesmesh <version>`.
    pub comments: Vec<String>,
    /// Write normals (when the mesh has them). Default `true` via
    /// [`PlyOptions::new`].
    pub normals: bool,
    /// Per-vertex colours (0..=1), one per vertex; empty: none.
    pub vertex_colors: Vec<[f32; 3]>,
}

impl PlyOptions {
    /// Default options (normals on).
    pub fn new() -> Self {
        Self {
            normals: true,
            ..Self::default()
        }
    }
}

/// Writes `mesh` as binary little-endian PLY.
///
/// # Errors
/// [`Error::InvalidInput`] for inconsistent attribute lengths, [`Error::Io`]
/// if writing fails.
pub fn write_ply<W: Write>(w: W, mesh: &Mesh, options: &PlyOptions) -> Result<()> {
    write_impl(w, mesh, options).map_err(|e| match e {
        Failure::Io(source) => Error::Io { path: None, source },
        Failure::Lib(e) => e,
    })
}

/// Writes `mesh` as PLY to `path` (gzip-compressed if the path ends in
/// `.gz` and the `gzip` feature is enabled).
///
/// # Errors
/// As [`write_ply`], with the path in I/O errors; [`Error::Unsupported`] for
/// a `.gz` path without the `gzip` feature.
pub fn write_ply_path(path: impl AsRef<Path>, mesh: &Mesh, options: &PlyOptions) -> Result<()> {
    let path = path.as_ref();
    let io = |source| Error::Io {
        path: Some(path.to_path_buf()),
        source,
    };
    let mut w = super::Sink::create(path)?;
    write_impl(&mut w, mesh, options)
        .and_then(|()| w.finish().map_err(Failure::Io))
        .map_err(|e| match e {
            Failure::Io(source) => io(source),
            Failure::Lib(e) => e,
        })
}

enum Failure {
    Io(std::io::Error),
    Lib(Error),
}

impl From<std::io::Error> for Failure {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn write_impl<W: Write>(mut w: W, m: &Mesh, o: &PlyOptions) -> Result<(), Failure> {
    let n = m.positions.len();
    let normals = o.normals && !m.normals.is_empty();
    let colors = !o.vertex_colors.is_empty();
    let components = !m.vertex_component.is_empty();
    let lengths = [
        (normals, m.normals.len(), "normals"),
        (colors, o.vertex_colors.len(), "vertex colours"),
        (true, m.vertex_atom.len(), "vertex atoms"),
        (true, m.vertex_patch.len(), "vertex patches"),
        (components, m.vertex_component.len(), "vertex components"),
    ];
    for (used, len, what) in lengths {
        if used && len != n {
            return Err(Failure::Lib(Error::InvalidInput(format!(
                "{len} {what} for {n} vertices"
            ))));
        }
    }
    let clean = |s: &str| s.replace(['\n', '\r'], " ");
    let mut h = String::from("ply\nformat binary_little_endian 1.0\n");
    h += &format!("comment sesmesh {}\n", crate::VERSION);
    for c in &o.comments {
        h += &format!("comment {}\n", clean(c));
    }
    h += &format!("element vertex {n}\n");
    for p in ["x", "y", "z"] {
        h += &format!("property float {p}\n");
    }
    if normals {
        for p in ["nx", "ny", "nz"] {
            h += &format!("property float {p}\n");
        }
    }
    if colors {
        for p in ["red", "green", "blue"] {
            h += &format!("property uchar {p}\n");
        }
    }
    h += "property uint atom\nproperty uchar patch\n";
    if components {
        h += "property uint component\n";
    }
    h += &format!(
        "element face {}\nproperty list uchar uint vertex_indices\nend_header\n",
        m.triangles.len()
    );
    w.write_all(h.as_bytes())?;
    let mut buf = Vec::with_capacity(1 << 16);
    for i in 0..n {
        for c in m.positions[i] {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        if normals {
            for c in m.normals[i] {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }
        if colors {
            for c in o.vertex_colors[i] {
                buf.push((c.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        buf.extend_from_slice(&m.vertex_atom[i].to_le_bytes());
        buf.push(m.vertex_patch[i] as u8);
        if components {
            buf.extend_from_slice(&m.vertex_component[i].to_le_bytes());
        }
        if buf.len() >= 1 << 16 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    for t in &m.triangles {
        buf.push(3);
        for v in t {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if buf.len() >= 1 << 16 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)?;
    Ok(())
}
