#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let _ = sesmesh::fuzzing::cif_tokens(data);
});
