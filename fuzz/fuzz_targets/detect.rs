// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]
// Format sniffing and gzip handling on arbitrary bytes.
mod common;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    common::exercise(data, None);
});
