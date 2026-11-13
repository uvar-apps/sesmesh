//! Mesh output formats (Design §10): Wavefront OBJ ([`obj`]), binary PLY
//! ([`ply`]) and MSMS `.vert`/`.face` ([`msms`]).

mod fmt;
pub mod msms;
pub mod obj;
pub mod ply;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{Error, Result};

/// A buffered output file, gzip-compressed when the path ends in `.gz`
/// (feature `gzip`).
pub(crate) enum Sink {
    Plain(BufWriter<File>),
    #[cfg(feature = "gzip")]
    Gzip(Box<flate2::write::GzEncoder<BufWriter<File>>>),
}

impl Sink {
    pub(crate) fn create(path: &Path) -> Result<Self> {
        let io = |source| Error::Io {
            path: Some(path.to_path_buf()),
            source,
        };
        let gz = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gz"));
        #[cfg(not(feature = "gzip"))]
        if gz {
            return Err(Error::Unsupported(
                "gzip output requires the `gzip` feature".to_string(),
            ));
        }
        let file = BufWriter::with_capacity(1 << 20, File::create(path).map_err(io)?);
        #[cfg(feature = "gzip")]
        if gz {
            return Ok(Self::Gzip(Box::new(flate2::write::GzEncoder::new(
                file,
                flate2::Compression::fast(),
            ))));
        }
        Ok(Self::Plain(file))
    }

    /// Flushes (and finishes the gzip stream).
    pub(crate) fn finish(self) -> std::io::Result<()> {
        match self {
            Self::Plain(mut w) => w.flush(),
            #[cfg(feature = "gzip")]
            Self::Gzip(w) => w.finish()?.flush(),
        }
    }
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(w) => w.write(buf),
            #[cfg(feature = "gzip")]
            Self::Gzip(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(w) => w.flush(),
            #[cfg(feature = "gzip")]
            Self::Gzip(w) => w.flush(),
        }
    }
}
