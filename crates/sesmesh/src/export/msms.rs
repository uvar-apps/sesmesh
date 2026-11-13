//! MSMS `.vert` / `.face` output (Spec FR-OUT-4, Design §10.2), the text
//! format of M. Sanner's MSMS, read by many docking and electrostatics
//! tools.
//!
//! `.vert`: three header lines, then per vertex
//! `%9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %7d %7d %2d` — position, normal,
//! face number (here: 1-based component), closest sphere (1-based input
//! atom) and type. `.face`: three header lines, then per triangle
//! `%6d %6d %6d %2d %6d` — 1-based vertices, type and face number. Types
//! follow the MSMS manual: 1 toric reentrant (saddle), 2 spheric reentrant
//! (concave), 3 contact (convex); a triangle takes the type its vertices
//! share, or 1 when they differ.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::mesh::{Mesh, PatchKind};

/// MSMS type code of a patch kind.
pub fn msms_type(patch: PatchKind) -> u8 {
    match patch {
        PatchKind::Saddle => 1,
        PatchKind::Concave => 2,
        PatchKind::Convex => 3,
    }
}

/// Header values of the MSMS files.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MsmsHeader {
    /// Description for the first comment line (e.g. the input name).
    pub description: String,
    /// Number of spheres (atoms) of the input.
    pub spheres: usize,
    /// Vertex density (vertices per Å²) reported in the header.
    pub density: f64,
    /// Probe radius in Å.
    pub probe_radius: f64,
}

impl MsmsHeader {
    /// A header.
    pub fn new(
        description: impl Into<String>,
        spheres: usize,
        density: f64,
        probe_radius: f64,
    ) -> Self {
        Self {
            description: description.into(),
            spheres,
            density,
            probe_radius,
        }
    }
}

/// Writes the `.vert` and `.face` contents to `vert` and `face`.
///
/// # Errors
/// [`Error::InvalidInput`] if the mesh has no normals or inconsistent
/// attributes, [`Error::Io`] if writing fails.
pub fn write_msms<V: Write, F: Write>(
    mut vert: V,
    mut face: F,
    mesh: &Mesh,
    header: &MsmsHeader,
) -> Result<()> {
    let n = mesh.positions.len();
    if mesh.normals.len() != n || mesh.vertex_atom.len() != n || mesh.vertex_patch.len() != n {
        return Err(Error::InvalidInput(
            "MSMS output needs normals, atoms and patches for every vertex".into(),
        ));
    }
    let io = |source| Error::Io { path: None, source };
    let comp = |v: usize| mesh.vertex_component.get(v).map_or(1, |&c| c as usize + 1);
    let head = |what: &str, count: usize| {
        format!(
            "# MSMS solvent excluded surface {what} for {}\n#{what} #sphere density probe_r\n{count:7} {:7} {:5.2} {:5.2}\n",
            header.description.replace(['\n', '\r'], " "),
            header.spheres,
            header.density,
            header.probe_radius
        )
    };
    vert.write_all(head("vertices", n).as_bytes()).map_err(io)?;
    let mut buf = String::with_capacity(1 << 16);
    for v in 0..n {
        let p = mesh.positions[v];
        let q = mesh.normals[v];
        let _ = writeln!(
            buf,
            "{:9.3} {:9.3} {:9.3} {:9.3} {:9.3} {:9.3} {:7} {:7} {:2}",
            p[0],
            p[1],
            p[2],
            q[0],
            q[1],
            q[2],
            comp(v),
            mesh.vertex_atom[v] + 1,
            msms_type(mesh.vertex_patch[v])
        );
        if buf.len() >= 1 << 16 {
            vert.write_all(buf.as_bytes()).map_err(io)?;
            buf.clear();
        }
    }
    vert.write_all(buf.as_bytes()).map_err(io)?;
    buf.clear();
    face.write_all(head("faces", mesh.triangles.len()).as_bytes())
        .map_err(io)?;
    for t in &mesh.triangles {
        let types = t.map(|v| msms_type(mesh.vertex_patch[v as usize]));
        let ty = if types[0] == types[1] && types[1] == types[2] {
            types[0]
        } else {
            1
        };
        let _ = writeln!(
            buf,
            "{:6} {:6} {:6} {:2} {:6}",
            t[0] + 1,
            t[1] + 1,
            t[2] + 1,
            ty,
            comp(t[0] as usize)
        );
        if buf.len() >= 1 << 16 {
            face.write_all(buf.as_bytes()).map_err(io)?;
            buf.clear();
        }
    }
    face.write_all(buf.as_bytes()).map_err(io)?;
    Ok(())
}

/// Writes `<stem>.vert` and `<stem>.face`.
///
/// # Errors
/// As [`write_msms`], with the path in I/O errors.
pub fn write_msms_files(stem: impl AsRef<Path>, mesh: &Mesh, header: &MsmsHeader) -> Result<()> {
    let stem = stem.as_ref();
    let (vp, fp) = (stem.with_extension("vert"), stem.with_extension("face"));
    let open = |p: &Path| {
        File::create(p)
            .map(|f| BufWriter::with_capacity(1 << 20, f))
            .map_err(|source| Error::Io {
                path: Some(p.to_path_buf()),
                source,
            })
    };
    let (mut v, mut f) = (open(&vp)?, open(&fp)?);
    write_msms(&mut v, &mut f, mesh, header)?;
    for (w, p) in [(&mut v, &vp), (&mut f, &fp)] {
        w.flush().map_err(|source| Error::Io {
            path: Some(p.clone()),
            source,
        })?;
    }
    Ok(())
}
