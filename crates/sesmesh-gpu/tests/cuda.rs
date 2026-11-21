// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA back end on a real device (Plan T12.1). The tests skip without a
//! CUDA driver, device or kernels, unless `SESMESH_GPU_REQUIRE=1` (then they
//! fail, so a GPU machine cannot pass them by skipping).

#![cfg(feature = "cuda")]

use sesmesh_gpu::Error;
use sesmesh_gpu::cuda::{Device, Event, Runtime, Stream, architectures};

fn required() -> bool {
    std::env::var("SESMESH_GPU_REQUIRE").is_ok_and(|v| v == "1")
}

/// Device 0, or `None` (skip) when there is none.
fn device() -> Option<Device> {
    let result = Runtime::load().and_then(|rt| rt.open(0));
    match result {
        Ok(d) => Some(d),
        Err(e @ (Error::NoRuntime { .. } | Error::NoDevice(_) | Error::NoKernels))
            if !required() =>
        {
            eprintln!("skipped: {e}");
            None
        }
        Err(e) => panic!("opening the CUDA device failed: {e}"),
    }
}

#[test]
fn missing_library_is_no_runtime() {
    let err = Runtime::load_from(&["/nonexistent/libcuda.so.1"]).unwrap_err();
    assert!(matches!(err, Error::NoRuntime { .. }), "{err}");
    assert!(err.to_string().contains("/nonexistent/libcuda.so.1"));
    let none: [&str; 0] = [];
    let err = Runtime::load_from(&none).unwrap_err();
    assert!(err.to_string().contains("no CUDA driver"), "{err}");
}

#[test]
fn device_info_is_sane() {
    let Some(d) = device() else { return };
    let info = d.info();
    eprintln!("{info:?}");
    assert!(info.arch.starts_with("sm_"), "{info:?}");
    assert_eq!(info.warp_size, 32);
    assert!(info.compute_units > 0 && info.memory > 0);
    let archs = architectures();
    assert!(!archs.is_empty());
    assert!(
        archs.contains(&info.code) || info.code.ends_with(" PTX"),
        "{info:?} {archs:?}"
    );
    let (free, total) = d.memory().unwrap();
    assert!(free <= total && total > 0);
    assert!(!info.runtime.is_empty());
}

/// The arithmetic agreement test of the spike, as a crate test: one million
/// input pairs, every result bit-identical to the CPU's.
#[test]
fn arithmetic_agrees_with_the_cpu() {
    let Some(d) = device() else { return };
    for seed in [1, 2] {
        let report = d.self_test(1 << 20, seed).unwrap();
        assert!(report.passed(), "{report}");
    }
}

#[test]
fn device_buffer_round_trip() {
    let Some(d) = device() else { return };
    let values: Vec<f64> = (0..100_003).map(|i| f64::from(i) * 0.1 - 7.0).collect();
    let mut buf = d.upload(&values).unwrap();
    assert_eq!(buf.len(), values.len());
    assert_eq!(buf.to_vec().unwrap(), values);
    let zeros = d.alloc::<u32>(1000).unwrap();
    assert!(zeros.to_vec().unwrap().iter().all(|&z| z == 0));
    let err = buf.copy_from(&values[1..]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Length {
                buffer: 100_003,
                slice: 100_002
            }
        ),
        "{err}"
    );
    let empty = d.alloc::<f32>(0).unwrap();
    assert!(empty.is_empty() && empty.to_vec().unwrap().is_empty());
    buf.copy_from(&vec![1.5; values.len()]).unwrap();
    assert!(buf.to_vec().unwrap().iter().all(|&v| v == 1.5));
}

#[test]
fn host_buffer_is_zeroed_and_writable() {
    let Some(d) = device() else { return };
    let mut h = d.alloc_host::<u64>(4096).unwrap();
    assert!(h.iter().all(|&v| v == 0));
    for (i, v) in h.iter_mut().enumerate() {
        *v = i as u64 * 3;
    }
    assert_eq!(h[4095], 4095 * 3);
    assert!(d.alloc_host::<f32>(0).unwrap().is_empty());
}

#[test]
fn buffers_outlive_the_device() {
    let Some(d) = device() else { return };
    let buf = d.upload(&[1u32, 2, 3]).unwrap();
    let host = d.alloc_host::<f32>(16).unwrap();
    let stream = Stream::new(&d).unwrap();
    drop(d);
    assert_eq!(buf.to_vec().unwrap(), [1, 2, 3]);
    assert_eq!(host.len(), 16);
    stream.synchronize().unwrap();
}

#[test]
fn events_time_work() {
    let Some(d) = device() else { return };
    let stream = Stream::new(&d).unwrap();
    stream.synchronize().unwrap();
    let (start, stop) = (Event::new(&d).unwrap(), Event::new(&d).unwrap());
    start.record(None).unwrap();
    assert!(d.self_test(1 << 16, 3).unwrap().passed());
    stop.record(None).unwrap();
    stop.synchronize().unwrap();
    let ms = stop.elapsed_ms(&start).unwrap();
    assert!(ms >= 0.0 && ms.is_finite(), "{ms}");
}

#[test]
fn device_is_usable_from_threads() {
    let Some(d) = device() else { return };
    std::thread::scope(|s| {
        for seed in 10..14 {
            let d = &d;
            s.spawn(move || assert!(d.self_test(1 << 14, seed).unwrap().passed()));
        }
    });
}

#[test]
fn missing_device_index_is_no_device() {
    let Some(d) = device() else { return };
    let rt = d.runtime().clone();
    let n = rt.device_count().unwrap();
    assert!(n >= 1);
    assert!(matches!(rt.open(n), Err(Error::NoDevice(_))));
}
