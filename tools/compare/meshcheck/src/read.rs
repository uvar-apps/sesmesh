// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Readers for the mesh formats written by the compared programs: Wavefront
//! OBJ (sesmesh, ChimeraX), OFF (NanoShaper), PLY ASCII/binary (EDTSurf,
//! sesmesh), MSMS `.vert`/`.face` (MSMS, NanoShaper, sesmesh) and STL.

use std::fs;
use std::path::{Path, PathBuf};

/// A triangle mesh as read from a file (positions in Å).
#[derive(Default)]
pub struct RawMesh {
    pub positions: Vec<[f64; 3]>,
    /// Per-vertex normals when the file has them.
    pub normals: Option<Vec<[f64; 3]>>,
    pub triangles: Vec<[u32; 3]>,
    /// Polygons with more than three vertices (fan-triangulated).
    pub polygons: usize,
}

pub type Res<T> = Result<T, String>;

fn num<T: std::str::FromStr>(s: &str, what: &str) -> Res<T> {
    s.parse::<T>().map_err(|_| format!("bad {what}: {s:?}"))
}

/// Detects the format from the file name.
pub fn detect(path: &Path) -> Res<&'static str> {
    let name = path.to_string_lossy().to_lowercase();
    for (ext, f) in [
        (".obj", "obj"),
        (".off", "off"),
        (".ply", "ply"),
        (".vert", "msms"),
        (".face", "msms"),
        (".stl", "stl"),
    ] {
        if name.ends_with(ext) {
            return Ok(f);
        }
    }
    Err(format!("unknown mesh format: {}", path.display()))
}

pub fn read(path: &Path, format: &str) -> Res<RawMesh> {
    match format {
        "obj" => read_obj(path),
        "off" => read_off(path),
        "ply" => read_ply(path),
        "msms" => read_msms(path),
        "stl" => read_stl(path),
        _ => Err(format!("unsupported format {format}")),
    }
}

fn text(path: &Path) -> Res<String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn fan(m: &mut RawMesh, poly: &[u32]) {
    if poly.len() > 3 {
        m.polygons += 1;
    }
    for k in 1..poly.len().saturating_sub(1) {
        m.triangles.push([poly[0], poly[k], poly[k + 1]]);
    }
}

fn read_obj(path: &Path) -> Res<RawMesh> {
    let src = text(path)?;
    let mut m = RawMesh::default();
    let mut vn: Vec<[f64; 3]> = Vec::new();
    let mut vnorm: Vec<Option<[f64; 3]>> = Vec::new();
    let mut poly = Vec::new();
    for line in src.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let p: Vec<f64> = it
                    .take(3)
                    .map(|s| num(s, "coordinate"))
                    .collect::<Res<_>>()?;
                if p.len() < 3 {
                    return Err(format!("short vertex line: {line}"));
                }
                m.positions.push([p[0], p[1], p[2]]);
                vnorm.push(None);
            }
            Some("vn") => {
                let p: Vec<f64> = it.take(3).map(|s| num(s, "normal")).collect::<Res<_>>()?;
                vn.push([p[0], p[1], p[2]]);
            }
            Some("f") => {
                poly.clear();
                let n = m.positions.len() as i64;
                for tok in it {
                    let mut parts = tok.split('/');
                    let vi: i64 = num(parts.next().unwrap_or(""), "face index")?;
                    let vi = if vi < 0 { n + vi } else { vi - 1 };
                    if vi < 0 || vi >= n {
                        return Err(format!("face index out of range: {tok}"));
                    }
                    let _tex = parts.next();
                    if let Some(ni) = parts.next().filter(|s| !s.is_empty()) {
                        let ni: i64 = num(ni, "normal index")?;
                        let ni = if ni < 0 { vn.len() as i64 + ni } else { ni - 1 };
                        if let Some(nv) = vn.get(ni as usize) {
                            vnorm[vi as usize] = Some(*nv);
                        }
                    }
                    poly.push(vi as u32);
                }
                fan(&mut m, &poly);
            }
            _ => {}
        }
    }
    if !vn.is_empty() {
        // Vertices without an explicit normal reference: same index, as
        // written by programs that emit one `vn` per `v`.
        let same = vn.len() == m.positions.len();
        m.normals = Some(
            vnorm
                .iter()
                .enumerate()
                .map(|(i, n)| n.unwrap_or(if same { vn[i] } else { [0.0; 3] }))
                .collect(),
        );
    }
    Ok(m)
}

/// Non-comment, non-empty lines, comments (`#`) stripped.
fn data_lines(src: &str) -> impl Iterator<Item = &str> {
    src.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
}

fn read_off(path: &Path) -> Res<RawMesh> {
    let src = text(path)?;
    let mut lines = data_lines(&src);
    let head = lines.next().ok_or("empty OFF file")?;
    let mut toks: Vec<&str> = head.split_whitespace().collect();
    let kind = toks.remove(0);
    // OFF, NOFF, COFF, CNOFF; NanoShaper writes "OFF+N" for OFF with normals.
    if !kind.contains("OFF") {
        return Err(format!("not an OFF file: {head}"));
    }
    let has_normals = kind.contains('N');
    let counts: Vec<usize> = if toks.len() >= 2 {
        toks.iter().map(|s| num(s, "count")).collect::<Res<_>>()?
    } else {
        let l = lines.next().ok_or("missing OFF counts")?;
        l.split_whitespace()
            .map(|s| num(s, "count"))
            .collect::<Res<_>>()?
    };
    let (nv, nf) = (counts[0], counts[1]);
    let mut m = RawMesh::default();
    let mut normals = Vec::new();
    for _ in 0..nv {
        let l = lines.next().ok_or("truncated OFF vertices")?;
        let v: Vec<f64> = l
            .split_whitespace()
            .take(6)
            .map(|s| num(s, "coordinate"))
            .collect::<Res<_>>()?;
        m.positions.push([v[0], v[1], v[2]]);
        if has_normals && v.len() >= 6 {
            normals.push([v[3], v[4], v[5]]);
        }
    }
    let mut poly = Vec::new();
    for _ in 0..nf {
        let l = lines.next().ok_or("truncated OFF faces")?;
        let mut it = l.split_whitespace();
        let k: usize = num(it.next().unwrap_or(""), "face size")?;
        poly.clear();
        for _ in 0..k {
            let i: u32 = num(it.next().unwrap_or(""), "face index")?;
            if i as usize >= nv {
                return Err(format!("face index out of range: {l}"));
            }
            poly.push(i);
        }
        fan(&mut m, &poly);
    }
    if normals.len() == nv && nv > 0 {
        m.normals = Some(normals);
    }
    Ok(m)
}

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Res<Ty> {
        Ok(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            _ => return Err(format!("unknown PLY type {s}")),
        })
    }
    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
}

struct Prop {
    name: String,
    ty: Ty,
    list: Option<Ty>,
}

struct Elem {
    name: String,
    count: usize,
    props: Vec<Prop>,
}

/// Reads scalar values from ASCII tokens or binary bytes.
enum Src<'a> {
    Ascii(std::str::SplitWhitespace<'a>),
    Bin {
        data: &'a [u8],
        pos: usize,
        big: bool,
    },
}

impl Src<'_> {
    fn get(&mut self, ty: Ty) -> Res<f64> {
        match self {
            Src::Ascii(it) => num(it.next().ok_or("truncated PLY data")?, "PLY value"),
            Src::Bin { data, pos, big } => {
                let n = ty.size();
                let b = data.get(*pos..*pos + n).ok_or("truncated PLY data")?;
                *pos += n;
                let mut a = [0u8; 8];
                a[..n].copy_from_slice(b);
                if *big {
                    a[..n].reverse();
                }
                Ok(match ty {
                    Ty::I8 => f64::from(a[0] as i8),
                    Ty::U8 => f64::from(a[0]),
                    Ty::I16 => f64::from(i16::from_le_bytes([a[0], a[1]])),
                    Ty::U16 => f64::from(u16::from_le_bytes([a[0], a[1]])),
                    Ty::I32 => f64::from(i32::from_le_bytes([a[0], a[1], a[2], a[3]])),
                    Ty::U32 => f64::from(u32::from_le_bytes([a[0], a[1], a[2], a[3]])),
                    Ty::F32 => f64::from(f32::from_le_bytes([a[0], a[1], a[2], a[3]])),
                    Ty::F64 => f64::from_le_bytes(a),
                })
            }
        }
    }
}

fn read_ply(path: &Path) -> Res<RawMesh> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let end = bytes
        .windows(10)
        .position(|w| w == b"end_header")
        .ok_or("PLY without end_header")?;
    let mut body = end + 10;
    while body < bytes.len() && bytes[body] != b'\n' {
        body += 1;
    }
    body += 1;
    let header = String::from_utf8_lossy(&bytes[..end]);
    let mut format = "";
    let mut elems: Vec<Elem> = Vec::new();
    for line in header.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("format") => {
                format = if t[1] == "ascii" {
                    "ascii"
                } else if t[1].contains("big") {
                    "big"
                } else {
                    "little"
                }
            }
            Some("element") => elems.push(Elem {
                name: t[1].to_string(),
                count: num(t[2], "element count")?,
                props: Vec::new(),
            }),
            Some("property") => {
                let e = elems.last_mut().ok_or("property before element")?;
                if t[1] == "list" {
                    e.props.push(Prop {
                        name: t[4].to_string(),
                        ty: Ty::parse(t[3])?,
                        list: Some(Ty::parse(t[2])?),
                    });
                } else {
                    e.props.push(Prop {
                        name: t[2].to_string(),
                        ty: Ty::parse(t[1])?,
                        list: None,
                    });
                }
            }
            _ => {}
        }
    }
    let ascii_body;
    let mut src = if format == "ascii" {
        ascii_body = String::from_utf8_lossy(&bytes[body..]).into_owned();
        Src::Ascii(ascii_body.split_whitespace())
    } else {
        Src::Bin {
            data: &bytes[body..],
            pos: 0,
            big: format == "big",
        }
    };
    let mut m = RawMesh::default();
    let mut normals = Vec::new();
    let mut poly = Vec::new();
    for e in &elems {
        for _ in 0..e.count {
            let mut p = [0.0; 3];
            let mut n = [0.0; 3];
            let mut has_n = false;
            poly.clear();
            for pr in &e.props {
                if let Some(ct) = pr.list {
                    let k = src.get(ct)? as usize;
                    for _ in 0..k {
                        let v = src.get(pr.ty)?;
                        if e.name == "face"
                            && (pr.name == "vertex_indices" || pr.name == "vertex_index")
                        {
                            poly.push(v as u32);
                        }
                    }
                } else {
                    let v = src.get(pr.ty)?;
                    if e.name == "vertex" {
                        match pr.name.as_str() {
                            "x" => p[0] = v,
                            "y" => p[1] = v,
                            "z" => p[2] = v,
                            "nx" => (n[0], has_n) = (v, true),
                            "ny" => n[1] = v,
                            "nz" => n[2] = v,
                            _ => {}
                        }
                    }
                }
            }
            if e.name == "vertex" {
                m.positions.push(p);
                if has_n {
                    normals.push(n);
                }
            } else if e.name == "face" {
                if poly.iter().any(|&i| i as usize >= m.positions.len()) {
                    return Err("PLY face index out of range".into());
                }
                let pc = poly.clone();
                fan(&mut m, &pc);
            }
        }
    }
    if !normals.is_empty() && normals.len() == m.positions.len() {
        m.normals = Some(normals);
    }
    Ok(m)
}

fn msms_paths(path: &Path) -> (PathBuf, PathBuf) {
    let s = path.to_string_lossy();
    let (base, gz) = match s.strip_suffix(".gz") {
        Some(b) => (b.to_string(), ".gz"),
        None => (s.to_string(), ""),
    };
    let stem = base.trim_end_matches(".vert").trim_end_matches(".face");
    (
        PathBuf::from(format!("{stem}.vert{gz}")),
        PathBuf::from(format!("{stem}.face{gz}")),
    )
}

/// MSMS data lines: skips `#` comments and the counts line (≤ 4 fields).
fn msms_rows(src: &str) -> Vec<Vec<&str>> {
    let mut rows: Vec<Vec<&str>> = src
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().collect())
        .collect();
    if rows.first().is_some_and(|r| r.len() <= 4) {
        rows.remove(0);
    }
    rows
}

/// MSMS writes component 0 to `<stem>.vert/.face` and, with
/// `-all_components`, component k to `<stem>_k.vert/.face`: read them all.
fn read_msms(path: &Path) -> Res<RawMesh> {
    let (vp, _) = msms_paths(path);
    let stem = vp.to_string_lossy().trim_end_matches(".vert").to_string();
    let mut parts = vec![PathBuf::from(&stem)];
    for k in 1.. {
        let p = PathBuf::from(format!("{stem}_{k}"));
        if !p.with_extension("vert").exists() {
            break;
        }
        parts.push(p);
    }
    let mut m = RawMesh::default();
    let mut normals = Vec::new();
    for p in parts {
        let base = m.positions.len() as u32;
        let part = read_msms_one(&p.with_extension("vert"))?;
        m.positions.extend(&part.positions);
        normals.extend(part.normals.unwrap_or_default());
        m.triangles
            .extend(part.triangles.iter().map(|t| t.map(|i| i + base)));
    }
    if normals.len() == m.positions.len() && !normals.is_empty() {
        m.normals = Some(normals);
    }
    Ok(m)
}

fn read_msms_one(path: &Path) -> Res<RawMesh> {
    let (vp, fp) = msms_paths(path);
    let (vs, fs_) = (text(&vp)?, text(&fp)?);
    let mut m = RawMesh::default();
    let mut normals = Vec::new();
    for r in msms_rows(&vs) {
        let v: Vec<f64> = r
            .iter()
            .take(6)
            .map(|s| num(s, "MSMS vertex"))
            .collect::<Res<_>>()?;
        m.positions.push([v[0], v[1], v[2]]);
        if v.len() == 6 {
            normals.push([v[3], v[4], v[5]]);
        }
    }
    let nv = m.positions.len();
    for r in msms_rows(&fs_) {
        let mut t = [0u32; 3];
        for k in 0..3 {
            let i: usize = num(r[k], "MSMS face index")?;
            if i == 0 || i > nv {
                return Err(format!("MSMS face index out of range: {i}"));
            }
            t[k] = (i - 1) as u32;
        }
        m.triangles.push(t);
    }
    if normals.len() == nv && nv > 0 {
        m.normals = Some(normals);
    }
    Ok(m)
}

fn read_stl(path: &Path) -> Res<RawMesh> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut corners: Vec<[f64; 3]> = Vec::new();
    let binary = bytes.len() >= 84 && {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        bytes.len() == 84 + 50 * n
    };
    if binary {
        let n = (bytes.len() - 84) / 50;
        for t in 0..n {
            let b = &bytes[84 + 50 * t..];
            for k in 0..3 {
                let o = 12 + 12 * k;
                let f = |i: usize| {
                    f64::from(f32::from_le_bytes([
                        b[o + 4 * i],
                        b[o + 4 * i + 1],
                        b[o + 4 * i + 2],
                        b[o + 4 * i + 3],
                    ]))
                };
                corners.push([f(0), f(1), f(2)]);
            }
        }
    } else {
        for line in String::from_utf8_lossy(&bytes).lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.first() == Some(&"vertex") {
                corners.push([num(t[1], "x")?, num(t[2], "y")?, num(t[3], "z")?]);
            }
        }
    }
    // STL has no shared vertices: weld exact duplicates.
    let mut m = RawMesh::default();
    let mut index = std::collections::HashMap::new();
    let mut tri = [0u32; 3];
    for (c, p) in corners.iter().enumerate() {
        let key = p.map(f64::to_bits);
        let id = *index.entry(key).or_insert_with(|| {
            m.positions.push(*p);
            (m.positions.len() - 1) as u32
        });
        tri[c % 3] = id;
        if c % 3 == 2 {
            m.triangles.push(tri);
        }
    }
    Ok(m)
}
