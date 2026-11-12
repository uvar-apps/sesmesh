#![no_main]
// Format sniffing and gzip handling on arbitrary bytes.
mod common;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    common::exercise(data, None);
});
