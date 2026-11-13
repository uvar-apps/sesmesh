//! Wavefront OBJ output (Spec FR-OUT-1, Design §10.1).
//!
//! ```
//! use sesmesh::export::obj::{ObjOptions, write_obj};
//! use sesmesh::{Sphere, SurfaceParams, compute_surface};
//!
//! let out = compute_surface(&[Sphere::new([0.0; 3], 1.5)], &SurfaceParams::ses(1.4))?;
//! let mut text = Vec::new();
//! write_obj(&mut text, &out.mesh, &ObjOptions::default())?;
//! let text = String::from_utf8(text).unwrap();
//! assert!(text.starts_with("# sesmesh "));
//! assert_eq!(text.lines().filter(|l| l.starts_with("f ")).count(), out.mesh.triangle_count());
//! # Ok::<(), sesmesh::Error>(())
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::fmt::{MAX_PRECISION, push_fixed, push_u64};
use crate::error::{Error, Result};
use crate::mesh::Mesh;
use crate::par;

/// Elements formatted per parallel task.
const CHUNK: usize = 1 << 15;
/// Chunks formatted concurrently (bounds the memory held in buffers).
const WINDOW: usize = 32;

/// OBJ writer options.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ObjOptions {
    /// Decimals of coordinates and normals (0..=9). Default 4.
    pub precision: usize,
    /// Write `vn` records and `f a//a …` faces. Default `true`.
    pub normals: bool,
    /// Extra header comment lines (without the leading `# `), written after
    /// the `# sesmesh <version>` line.
    pub comments: Vec<String>,
    /// Object name (`o` record). Default `ses`.
    pub object_name: String,
    /// Per-vertex colours written as `v x y z r g b` (a common extension
    /// read by MeshLab, Blender and others); empty: none. Must have one entry
    /// per vertex when set.
    pub vertex_colors: Vec<[f32; 3]>,
    /// `g <name>` records: `(first triangle, name)`, ascending; each group
    /// runs to the next one (e.g. one group per component).
    pub groups: Vec<(usize, String)>,
}

impl Default for ObjOptions {
    fn default() -> Self {
        Self {
            precision: 4,
            normals: true,
            comments: Vec::new(),
            object_name: "ses".to_string(),
            vertex_colors: Vec::new(),
            groups: Vec::new(),
        }
    }
}

/// Writes `mesh` as OBJ to `w`. Formatting runs in parallel; output is
/// identical for any thread count.
///
/// # Errors
/// [`Error::InvalidInput`] for a precision above 9; [`Error::Io`] if writing
/// fails.
pub fn write_obj<W: Write>(w: W, mesh: &Mesh, options: &ObjOptions) -> Result<()> {
    write_impl(w, mesh, options).map_err(|e| match e {
        WriteError::Io(source) => Error::Io { path: None, source },
        WriteError::Lib(e) => e,
    })
}

/// Writes `mesh` as OBJ to `path` (gzip-compressed if the path ends in
/// `.gz` and the `gzip` feature is enabled).
///
/// # Errors
/// As [`write_obj`], with the path in I/O errors; [`Error::Unsupported`] for
/// a `.gz` path without the `gzip` feature.
pub fn write_obj_path(path: impl AsRef<Path>, mesh: &Mesh, options: &ObjOptions) -> Result<()> {
    let path = path.as_ref();
    let io = |source| Error::Io {
        path: Some(path.to_path_buf()),
        source,
    };
    let file = File::create(path).map_err(io)?;
    let gz = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"));
    let result = if gz {
        #[cfg(feature = "gzip")]
        {
            let mut enc = flate2::write::GzEncoder::new(
                BufWriter::with_capacity(1 << 20, file),
                flate2::Compression::fast(),
            );
            write_impl(&mut enc, mesh, options).and_then(|()| {
                enc.finish()
                    .and_then(|mut b| b.flush())
                    .map_err(WriteError::Io)
            })
        }
        #[cfg(not(feature = "gzip"))]
        {
            return Err(Error::Unsupported(
                "gzip output requires the `gzip` feature".to_string(),
            ));
        }
    } else {
        let mut b = BufWriter::with_capacity(1 << 20, file);
        write_impl(&mut b, mesh, options).and_then(|()| b.flush().map_err(WriteError::Io))
    };
    result.map_err(|e| match e {
        WriteError::Io(source) => io(source),
        WriteError::Lib(e) => e,
    })
}

enum WriteError {
    Io(std::io::Error),
    Lib(Error),
}

impl From<std::io::Error> for WriteError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn write_impl<W: Write>(mut w: W, mesh: &Mesh, o: &ObjOptions) -> Result<(), WriteError> {
    if o.precision > MAX_PRECISION {
        return Err(WriteError::Lib(Error::InvalidInput(format!(
            "OBJ precision must be 0..={MAX_PRECISION}, got {}",
            o.precision
        ))));
    }
    // A mesh without normals (NormalMode::None) is written without them.
    let normals = o.normals && !mesh.normals.is_empty();
    if normals && mesh.normals.len() != mesh.positions.len() {
        return Err(WriteError::Lib(Error::InvalidInput(
            "mesh has no normal for every vertex".to_string(),
        )));
    }
    if !o.vertex_colors.is_empty() && o.vertex_colors.len() != mesh.positions.len() {
        return Err(WriteError::Lib(Error::InvalidInput(format!(
            "{} vertex colours for {} vertices",
            o.vertex_colors.len(),
            mesh.positions.len()
        ))));
    }
    if o.groups.windows(2).any(|w| w[0].0 > w[1].0) {
        return Err(WriteError::Lib(Error::InvalidInput(
            "OBJ groups must be sorted by first triangle".to_string(),
        )));
    }
    let clean = |s: &str| s.replace(['\n', '\r'], " ");
    writeln!(w, "# sesmesh {}", crate::VERSION)?;
    for c in &o.comments {
        writeln!(w, "# {}", clean(c))?;
    }
    writeln!(w, "o {}", clean(&o.object_name))?;
    let prec = o.precision;
    fn vec3<'a>(
        tag: &'a [u8],
        data: &'a [[f32; 3]],
        colors: &'a [[f32; 3]],
        prec: usize,
    ) -> impl Fn(&mut Vec<u8>, usize) + Sync + 'a {
        move |buf: &mut Vec<u8>, i: usize| {
            buf.extend_from_slice(tag);
            for (k, &c) in data[i].iter().enumerate() {
                if k > 0 {
                    buf.push(b' ');
                }
                push_fixed(buf, c, prec);
            }
            if let Some(rgb) = colors.get(i) {
                for &c in rgb {
                    buf.push(b' ');
                    push_fixed(buf, c.clamp(0.0, 1.0), 4);
                }
            }
            buf.push(b'\n');
        }
    }
    emit(
        &mut w,
        mesh.positions.len(),
        vec3(b"v ", &mesh.positions, &o.vertex_colors, prec),
    )?;
    if normals {
        emit(
            &mut w,
            mesh.normals.len(),
            vec3(b"vn ", &mesh.normals, &[], prec),
        )?;
    }
    let tris = &mesh.triangles;
    let face = |buf: &mut Vec<u8>, i: usize| {
        buf.extend_from_slice(b"f");
        for &v in &tris[i] {
            let n = u64::from(v) + 1;
            buf.push(b' ');
            push_u64(buf, n);
            if normals {
                buf.extend_from_slice(b"//");
                push_u64(buf, n);
            }
        }
        buf.push(b'\n');
    };
    // Faces in group ranges (one range when there are no groups).
    let mut bounds: Vec<(usize, Option<&str>)> = vec![(0, None)];
    for (first, name) in &o.groups {
        let first = (*first).min(tris.len());
        if bounds.last().is_some_and(|b| b.0 == first) {
            bounds.pop();
        }
        bounds.push((first, Some(name.as_str())));
    }
    for (k, &(start, name)) in bounds.iter().enumerate() {
        let end = bounds.get(k + 1).map_or(tris.len(), |b| b.0);
        if let Some(name) = name {
            writeln!(w, "g {}", clean(name))?;
        }
        emit_range(&mut w, start, end, &face)?;
    }
    Ok(())
}

/// Formats `n` records with `f` in parallel chunks and writes them in order.
fn emit<W, F>(w: &mut W, n: usize, f: F) -> std::io::Result<()>
where
    W: Write,
    F: Fn(&mut Vec<u8>, usize) + Sync,
{
    emit_range(w, 0, n, &f)
}

/// [`emit`] for records `begin..end`.
fn emit_range<W, F>(w: &mut W, begin: usize, end_all: usize, f: &F) -> std::io::Result<()>
where
    W: Write,
    F: Fn(&mut Vec<u8>, usize) + Sync,
{
    let n = end_all - begin;
    let chunks = n.div_ceil(CHUNK);
    let mut first = 0;
    while first < chunks {
        let count = WINDOW.min(chunks - first);
        let bufs = par::map(count, |c| {
            let start = begin + (first + c) * CHUNK;
            let end = (start + CHUNK).min(end_all);
            let mut buf = Vec::with_capacity((end - start) * 40);
            for i in start..end {
                f(&mut buf, i);
            }
            buf
        });
        for b in bufs {
            w.write_all(&b)?;
        }
        first += count;
    }
    Ok(())
}
