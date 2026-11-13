//! Structure file readers: PDB, PDBx/mmCIF, XYZ / extended XYZ and PQR,
//! optionally gzip-compressed (Spec FR-IN, Design §5).
//!
//! ```no_run
//! use sesmesh::io::{read_path, ReadOptions};
//!
//! let structure = read_path("1crn.cif.gz", &ReadOptions::default())?;
//! println!("{} atoms in {} model(s)", structure.atom_count(), structure.models.len());
//! # Ok::<(), sesmesh::Error>(())
//! ```

mod assembly;
pub(crate) mod cif;

pub use assembly::AssemblySummary;
mod hybrid36;
mod lines;
mod model;
mod num;
mod pdb;
mod pqr;
mod xyz;

use std::borrow::Cow;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::diag::Diagnostics;
use crate::error::{Error, Result};
use crate::structure::{Model, Source, StringPool, Structure};

/// Supported structure file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// wwPDB fixed-column PDB format.
    Pdb,
    /// PDBx/mmCIF.
    Mmcif,
    /// XYZ or extended XYZ.
    Xyz,
    /// PQR (PDB with charges and radii, whitespace-delimited).
    Pqr,
}

impl Format {
    /// All formats.
    pub const ALL: [Format; 4] = [Format::Pdb, Format::Mmcif, Format::Xyz, Format::Pqr];

    /// Short lower-case name (`pdb`, `mmcif`, `xyz`, `pqr`).
    pub fn name(self) -> &'static str {
        match self {
            Format::Pdb => "pdb",
            Format::Mmcif => "mmcif",
            Format::Xyz => "xyz",
            Format::Pqr => "pqr",
        }
    }

    /// Parses a format name (case-insensitive; `cif` is accepted for mmCIF).
    pub fn from_name(name: &str) -> Option<Format> {
        match name.to_ascii_lowercase().as_str() {
            "pdb" | "ent" => Some(Format::Pdb),
            "mmcif" | "cif" | "mcif" => Some(Format::Mmcif),
            "xyz" | "extxyz" => Some(Format::Xyz),
            "pqr" => Some(Format::Pqr),
            _ => None,
        }
    }

    /// Format implied by a file name's extension, ignoring a trailing `.gz`
    /// (`1abc.cif.gz` → mmCIF, `pdb1abc.ent` → PDB).
    pub fn from_path(path: &Path) -> Option<Format> {
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        let name = name.strip_suffix(".gz").unwrap_or(&name);
        let ext = Path::new(name).extension()?.to_str()?;
        Self::from_name(ext)
    }

    /// Guesses the format from the first bytes of (decompressed) content.
    ///
    /// mmCIF and XYZ are recognized by their first meaningful line. Files
    /// with `ATOM`/`HETATM` records are PDB if the first such record has
    /// valid fixed-column coordinates, PQR if it splits into 10 or 11
    /// whitespace-separated fields ending in numbers. PQR files written with
    /// PDB columns cannot be told apart from PDB: name them `*.pqr`.
    pub fn sniff(data: &[u8]) -> Option<Format> {
        let head = &data[..data.len().min(1 << 16)];
        let mut lines = lines::Lines::new(head)
            .map(|(_, l)| l)
            .filter(|l| !l.trim_ascii().is_empty());
        let first = lines.next()?.trim_ascii();
        let lower = |s: &[u8]| s.to_ascii_lowercase();
        if first.starts_with(b"#") || first.starts_with(b"_") {
            return Some(Format::Mmcif);
        }
        if lower(first).starts_with(b"data_") || lower(first).starts_with(b"loop_") {
            return Some(Format::Mmcif);
        }
        if first.iter().all(u8::is_ascii_digit) {
            return Some(Format::Xyz);
        }
        let record = |l: &[u8]| l.starts_with(b"ATOM") || l.starts_with(b"HETATM");
        let atom_line = std::iter::once(first).chain(lines).find(|l| record(l))?;
        let pdb_coords = atom_line.len() >= 54
            && [30, 38, 46]
                .iter()
                .all(|&c| num::parse_f64(&atom_line[c..c + 8]).is_some());
        if pdb_coords {
            return Some(Format::Pdb);
        }
        let f: Vec<&[u8]> = lines::fields(atom_line).collect();
        let pqr = (f.len() == 10 || f.len() == 11)
            && f[f.len() - 5..].iter().all(|t| num::parse_f64(t).is_some());
        Some(if pqr { Format::Pqr } else { Format::Pdb })
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Format::Pdb => "PDB",
            Format::Mmcif => "mmCIF",
            Format::Xyz => "XYZ",
            Format::Pqr => "PQR",
        })
    }
}

/// Options for the readers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReadOptions {
    /// Input format; `None` detects it from the file name, then the content.
    pub format: Option<Format>,
    /// mmCIF data block to read; `None` uses the first block with
    /// `_atom_site`.
    pub data_block: Option<String>,
    /// Extended XYZ: name of a per-atom radius column in `Properties=`.
    pub radius_column: Option<String>,
    /// Build this biological assembly (mmCIF `pdbx_struct_assembly_gen`
    /// `assembly_id`, PDB `REMARK 350` biomolecule number) instead of the
    /// deposited coordinates (Spec FR-IN-8). Default `None`.
    pub assembly: Option<String>,
    /// Maximum size of decompressed gzip input, protecting against
    /// decompression bombs. Default [`ReadOptions::DEFAULT_MAX_DECOMPRESSED`]
    /// (the largest PDB entries decompress to a few hundred MB).
    pub max_decompressed_bytes: u64,
}

impl ReadOptions {
    /// Default for [`ReadOptions::max_decompressed_bytes`]: 8 GiB.
    pub const DEFAULT_MAX_DECOMPRESSED: u64 = 8 << 30;
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            format: None,
            data_block: None,
            radius_column: None,
            assembly: None,
            max_decompressed_bytes: Self::DEFAULT_MAX_DECOMPRESSED,
        }
    }
}

/// Output of a format reader, before the provenance is attached.
#[derive(Debug)]
pub(crate) struct Parsed {
    pub(crate) models: Vec<Model>,
    pub(crate) strings: StringPool,
    pub(crate) diagnostics: Diagnostics,
    pub(crate) data_block: Option<String>,
}

/// Reads a structure file. Gzip-compressed files are detected by their magic
/// bytes and decompressed (feature `gzip`).
///
/// # Errors
/// [`Error::Io`] if the file cannot be read, [`Error::Parse`] for malformed
/// content, [`Error::Unsupported`] if the format cannot be determined.
pub fn read_path(path: impl AsRef<Path>, options: &ReadOptions) -> Result<Structure> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|source| Error::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    let format = options.format.or_else(|| Format::from_path(path));
    read_impl(Cow::Owned(data), format, options, Some(path.to_path_buf()))
}

/// Reads a structure from any reader (e.g. stdin). The format is taken from
/// `options` or detected from the content.
///
/// # Errors
/// See [`read_path`].
pub fn read_reader<R: Read>(mut reader: R, options: &ReadOptions) -> Result<Structure> {
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .map_err(|source| Error::Io { path: None, source })?;
    read_impl(Cow::Owned(data), options.format, options, None)
}

/// Reads a structure from an in-memory buffer.
///
/// # Errors
/// See [`read_path`].
pub fn read_bytes(data: &[u8], options: &ReadOptions) -> Result<Structure> {
    read_impl(Cow::Borrowed(data), options.format, options, None)
}

fn read_impl(
    data: Cow<'_, [u8]>,
    format: Option<Format>,
    options: &ReadOptions,
    path: Option<PathBuf>,
) -> Result<Structure> {
    let data = decompress(data, path.as_deref(), options.max_decompressed_bytes)?;
    let format = match format {
        Some(f) => f,
        None => Format::sniff(&data).ok_or_else(|| {
            Error::Unsupported("could not determine the input format; specify it explicitly".into())
        })?,
    };
    let parsed = match format {
        Format::Pdb => pdb::parse(&data),
        Format::Mmcif => cif::parse(&data, options.data_block.as_deref()),
        Format::Xyz => xyz::parse(&data, options.radius_column.as_deref()),
        Format::Pqr => pqr::parse(&data),
    }
    .map_err(|e| e.into_error(format, path.clone()))?;
    let structure = Structure {
        models: parsed.models,
        strings: parsed.strings,
        source: Source {
            format,
            path: path.clone(),
            data_block: parsed.data_block,
        },
        diagnostics: parsed.diagnostics,
    };
    match &options.assembly {
        None => Ok(structure),
        Some(id) => {
            let defs = assembly_data(&data, format, options, path)?;
            assembly::expand(structure, &defs, id)
        }
    }
}

fn assembly_data(
    data: &[u8],
    format: Format,
    options: &ReadOptions,
    path: Option<PathBuf>,
) -> Result<assembly::AssemblyData> {
    match format {
        Format::Mmcif => assembly::parse_cif(data, options.data_block.as_deref())
            .map_err(|e| e.into_error(format, path)),
        Format::Pdb => Ok(assembly::parse_pdb(data)),
        _ => Err(Error::Unsupported(format!(
            "biological assemblies need mmCIF or PDB input, not {}",
            format.name()
        ))),
    }
}

/// Lists the biological assemblies a file defines (mmCIF
/// `pdbx_struct_assembly_gen`, PDB `REMARK 350`); empty if none.
///
/// # Errors
/// As [`read_path`]; [`Error::Unsupported`] for formats without assemblies.
pub fn read_assemblies(
    path: impl AsRef<Path>,
    options: &ReadOptions,
) -> Result<Vec<AssemblySummary>> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|source| Error::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    let data = decompress(Cow::Owned(data), Some(path), options.max_decompressed_bytes)?;
    let format = match options.format {
        Some(f) => f,
        None => Format::from_path(path)
            .or_else(|| Format::sniff(&data))
            .ok_or_else(|| Error::Unsupported("could not determine the input format".into()))?,
    };
    assembly_data(&data, format, options, Some(path.to_path_buf()))?.summaries()
}

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

#[cfg(feature = "gzip")]
fn decompress<'a>(data: Cow<'a, [u8]>, path: Option<&Path>, limit: u64) -> Result<Cow<'a, [u8]>> {
    if !data.starts_with(&GZIP_MAGIC) {
        return Ok(data);
    }
    let guess = data.len().saturating_mul(4);
    let mut out = Vec::with_capacity(guess.min(usize::try_from(limit).unwrap_or(usize::MAX)));
    flate2::read::MultiGzDecoder::new(&data[..])
        .take(limit.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|source| Error::Io {
            path: path.map(Path::to_path_buf),
            source,
        })?;
    if out.len() as u64 > limit {
        return Err(Error::ResourceLimit {
            what: "decompressed input".into(),
            limit,
        });
    }
    Ok(Cow::Owned(out))
}

#[cfg(not(feature = "gzip"))]
fn decompress<'a>(data: Cow<'a, [u8]>, _path: Option<&Path>, _limit: u64) -> Result<Cow<'a, [u8]>> {
    if data.starts_with(&GZIP_MAGIC) {
        return Err(Error::Unsupported(
            "gzip-compressed input requires the `gzip` feature".into(),
        ));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_path() {
        let f = |p: &str| Format::from_path(Path::new(p));
        assert_eq!(f("a/1abc.cif.gz"), Some(Format::Mmcif));
        assert_eq!(f("1ABC.PDB"), Some(Format::Pdb));
        assert_eq!(f("pdb1abc.ent.gz"), Some(Format::Pdb));
        assert_eq!(f("x.extxyz"), Some(Format::Xyz));
        assert_eq!(f("x.pqr"), Some(Format::Pqr));
        assert_eq!(f("x.gz"), None);
        assert_eq!(f("noext"), None);
    }

    #[test]
    fn sniffing() {
        assert_eq!(Format::sniff(b"\n# c\ndata_x\n"), Some(Format::Mmcif));
        assert_eq!(Format::sniff(b"DATA_X\n"), Some(Format::Mmcif));
        assert_eq!(Format::sniff(b" 3\nwater\n"), Some(Format::Xyz));
        let pdb = b"HEADER x\nATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00  0.00\n";
        assert_eq!(Format::sniff(pdb), Some(Format::Pdb));
        let pqr = b"REMARK x\nATOM 1 N ALA 1 -0.677 -1.230 -0.491 -0.3000 1.8240\n";
        assert_eq!(Format::sniff(pqr), Some(Format::Pqr));
        assert_eq!(Format::sniff(b""), None);
        assert_eq!(Format::sniff(b"hello world\n"), None);
    }

    #[test]
    fn names_round_trip() {
        for f in Format::ALL {
            assert_eq!(Format::from_name(f.name()), Some(f));
        }
    }

    #[test]
    fn parse_errors_carry_format_and_path() {
        let err = read_bytes(
            b"ATOM      1  CA  GLY A   1       1.000   2.000\n",
            &ReadOptions::default(),
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(
            text.starts_with("<input>:1:47: invalid PDB input"),
            "{text}"
        );
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn decompression_limit() {
        use std::io::Write;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&[b' '; 10_000]).unwrap();
        let gz = enc.finish().unwrap();
        let small = ReadOptions {
            max_decompressed_bytes: 9_999,
            ..ReadOptions::default()
        };
        let err = read_bytes(&gz, &small).unwrap_err();
        assert!(
            matches!(err, Error::ResourceLimit { limit: 9_999, .. }),
            "{err}"
        );
        let exact = ReadOptions {
            max_decompressed_bytes: 10_000,
            ..ReadOptions::default()
        };
        // Within the limit: decompresses, then fails only because it is blank.
        assert!(matches!(
            read_bytes(&gz, &exact),
            Err(Error::Unsupported(_))
        ));
    }

    #[cfg(not(feature = "gzip"))]
    #[test]
    fn gzip_without_feature_is_a_clear_error() {
        let err = read_bytes(&[0x1f, 0x8b, 8, 0], &ReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Unsupported(ref m) if m.contains("`gzip` feature")));
    }
}
