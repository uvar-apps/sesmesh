// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]
libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let _ = sesmesh::fuzzing::cif_tokens(data);
});
