// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]
mod common;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    common::exercise(data, Some(sesmesh::io::Format::Mmcif));
});
