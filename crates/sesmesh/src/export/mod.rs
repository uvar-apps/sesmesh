//! Mesh output formats (Design §10): Wavefront OBJ ([`obj`]), binary PLY
//! ([`ply`]) and MSMS `.vert`/`.face` ([`msms`]).

mod fmt;
pub mod msms;
pub mod obj;
pub mod ply;
