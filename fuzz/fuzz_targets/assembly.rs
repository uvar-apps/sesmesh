// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]
// Biological assembly parsing and expansion (mmCIF pdbx_struct_*, PDB
// REMARK 350, operator expressions): Ok or Err, never a panic or an
// unbounded allocation.
use sesmesh::io::{Format, ReadOptions, read_bytes};

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    for format in [Format::Mmcif, Format::Pdb] {
        let mut options = ReadOptions::default();
        options.format = Some(format);
        options.max_decompressed_bytes = 64 << 20;
        for id in ["1", "2", "A"] {
            options.assembly = Some(id.to_string());
            let _ = read_bytes(data, &options);
        }
    }
});
