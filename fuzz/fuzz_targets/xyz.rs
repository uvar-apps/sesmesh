#![no_main]
mod common;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    common::exercise(data, Some(sesmesh::io::Format::Xyz));
});
