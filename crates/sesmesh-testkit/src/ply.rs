//! Minimal binary little-endian PLY reader for round-trip tests: `vertex`
//! scalar properties and a `face` list property (`uchar` count, `uint` or
//! `int` indices).

/// Parsed PLY content.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlyMesh {
    /// Vertex property names, in file order.
    pub vertex_properties: Vec<String>,
    /// Vertex property values as `f64`, one row per vertex.
    pub vertices: Vec<Vec<f64>>,
    /// Faces (vertex indices).
    pub faces: Vec<Vec<u32>>,
    /// Header comments.
    pub comments: Vec<String>,
}

impl PlyMesh {
    /// Column of property `name`.
    pub fn column(&self, name: &str) -> Option<Vec<f64>> {
        let k = self.vertex_properties.iter().position(|p| p == name)?;
        Some(self.vertices.iter().map(|v| v[k]).collect())
    }
}

fn size(ty: &str) -> Result<usize, String> {
    Ok(match ty {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
        "double" | "float64" => 8,
        _ => return Err(format!("unknown PLY type {ty}")),
    })
}

fn read(ty: &str, b: &[u8]) -> f64 {
    match ty {
        "char" | "int8" => f64::from(b[0] as i8),
        "uchar" | "uint8" => f64::from(b[0]),
        "short" | "int16" => f64::from(i16::from_le_bytes([b[0], b[1]])),
        "ushort" | "uint16" => f64::from(u16::from_le_bytes([b[0], b[1]])),
        "int" | "int32" => f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        "uint" | "uint32" => f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        "float" | "float32" => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        _ => f64::from_le_bytes(b[..8].try_into().unwrap()),
    }
}

/// Parses a binary little-endian PLY file.
///
/// # Errors
/// A message describing the first problem.
pub fn parse(data: &[u8]) -> Result<PlyMesh, String> {
    let end = b"end_header\n";
    let pos = data
        .windows(end.len())
        .position(|w| w == end)
        .ok_or("no end_header")?;
    let header = std::str::from_utf8(&data[..pos]).map_err(|e| e.to_string())?;
    let mut body = &data[pos + end.len()..];
    let mut m = PlyMesh::default();
    let (mut nv, mut nf) = (0usize, 0usize);
    let mut vtypes: Vec<String> = Vec::new();
    let mut face_types = (String::new(), String::new());
    let mut element = "";
    let mut lines = header.lines();
    if lines.next() != Some("ply") {
        return Err("not a PLY file".into());
    }
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["format", "binary_little_endian", "1.0"] => {}
            ["format", ..] => return Err(format!("unsupported {line}")),
            ["comment", rest @ ..] => m.comments.push(rest.join(" ")),
            ["element", "vertex", n] => {
                element = "vertex";
                nv = n.parse().map_err(|_| "bad count")?;
            }
            ["element", "face", n] => {
                element = "face";
                nf = n.parse().map_err(|_| "bad count")?;
            }
            ["property", "list", c, i, _] if element == "face" => {
                face_types = ((*c).to_string(), (*i).to_string());
            }
            ["property", ty, name] if element == "vertex" => {
                size(ty)?;
                vtypes.push((*ty).to_string());
                m.vertex_properties.push((*name).to_string());
            }
            _ => return Err(format!("unsupported header line {line:?}")),
        }
    }
    for _ in 0..nv {
        let mut row = Vec::with_capacity(vtypes.len());
        for ty in &vtypes {
            let s = size(ty)?;
            if body.len() < s {
                return Err("truncated vertex data".into());
            }
            row.push(read(ty, body));
            body = &body[s..];
        }
        m.vertices.push(row);
    }
    let (cs, is) = (size(&face_types.0)?, size(&face_types.1)?);
    for _ in 0..nf {
        if body.len() < cs {
            return Err("truncated face data".into());
        }
        let count = read(&face_types.0, body) as usize;
        body = &body[cs..];
        let mut face = Vec::with_capacity(count);
        for _ in 0..count {
            if body.len() < is {
                return Err("truncated face data".into());
            }
            face.push(read(&face_types.1, body) as u32);
            body = &body[is..];
        }
        m.faces.push(face);
    }
    if !body.is_empty() {
        return Err(format!("{} trailing bytes", body.len()));
    }
    Ok(m)
}
