// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal Wavefront OBJ reader for round-trip tests (`v`, `vn`, `f` with
//! `a`, `a//n` or `a/t/n` indices; everything else ignored).

/// Parsed OBJ content.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjMesh {
    /// `v` records.
    pub positions: Vec<[f64; 3]>,
    /// `vn` records.
    pub normals: Vec<[f64; 3]>,
    /// `f` records as 0-based vertex indices (triangles only).
    pub triangles: Vec<[u32; 3]>,
    /// `f` records as 0-based normal indices, if given.
    pub triangle_normals: Vec<[u32; 3]>,
    /// Comment lines (without `#`).
    pub comments: Vec<String>,
}

/// Parses OBJ text.
///
/// # Errors
/// A message naming the first malformed line.
pub fn parse(text: &str) -> Result<ObjMesh, String> {
    let mut m = ObjMesh::default();
    for (n, line) in text.lines().enumerate() {
        let err = |what: &str| format!("line {}: {what}: {line:?}", n + 1);
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v" | "vn") => {
                let xyz: Vec<f64> = it
                    .map(|t| t.parse::<f64>().map_err(|_| err("bad number")))
                    .collect::<Result<_, _>>()?;
                if xyz.len() < 3 {
                    return Err(err("expected 3 coordinates"));
                }
                let p = [xyz[0], xyz[1], xyz[2]];
                if line.starts_with("vn") {
                    m.normals.push(p);
                } else {
                    m.positions.push(p);
                }
            }
            Some("f") => {
                let refs: Vec<&str> = it.collect();
                if refs.len() != 3 {
                    return Err(err("expected a triangle"));
                }
                let mut v = [0u32; 3];
                let mut vn = [0u32; 3];
                let mut has_n = false;
                for (k, r) in refs.iter().enumerate() {
                    let parts: Vec<&str> = r.split('/').collect();
                    let idx = |s: &str| {
                        s.parse::<u32>()
                            .ok()
                            .filter(|&i| i >= 1)
                            .map(|i| i - 1)
                            .ok_or_else(|| err("bad index"))
                    };
                    v[k] = idx(parts[0])?;
                    if let Some(nrm) = parts.get(2) {
                        vn[k] = idx(nrm)?;
                        has_n = true;
                    }
                }
                m.triangles.push(v);
                if has_n {
                    m.triangle_normals.push(vn);
                }
            }
            Some(c) if c.starts_with('#') => {
                m.comments
                    .push(line.trim_start_matches('#').trim().to_string());
            }
            _ => {}
        }
    }
    Ok(m)
}
