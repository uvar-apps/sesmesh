// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIP feasibility spike (#24): hand-written FFI to libamdhip64, a code
//! object built by hipcc, and four measurements:
//!
//! 1. bit agreement of f32/f64 `+ − × ÷ √` (and a distance expression)
//!    between the device and this CPU;
//! 2. f32/f64 throughput (mul+add, fma, sqrt, div);
//! 3. launch latency;
//! 4. bandwidth of device, pinned host and managed memory (zero-copy on an
//!    APU).

#![allow(unsafe_code, non_camel_case_types)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::time::Instant;

type hipError_t = c_int;
type hipModule_t = *mut c_void;
type hipFunction_t = *mut c_void;
type hipStream_t = *mut c_void;
type hipEvent_t = *mut c_void;

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipInit(flags: c_uint) -> hipError_t;
    fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    fn hipSetDevice(dev: c_int) -> hipError_t;
    fn hipDeviceGetAttribute(value: *mut c_int, attr: c_int, dev: c_int) -> hipError_t;
    fn hipDeviceGetName(name: *mut c_char, len: c_int, dev: c_int) -> hipError_t;
    fn hipGetErrorString(e: hipError_t) -> *const c_char;
    fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    fn hipModuleGetFunction(
        f: *mut hipFunction_t,
        m: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    #[allow(clippy::too_many_arguments)]
    fn hipModuleLaunchKernel(
        f: hipFunction_t,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        shared: c_uint,
        stream: hipStream_t,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    fn hipFree(ptr: *mut c_void) -> hipError_t;
    fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    fn hipMallocManaged(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
    fn hipEventCreate(e: *mut hipEvent_t) -> hipError_t;
    fn hipEventRecord(e: hipEvent_t, s: hipStream_t) -> hipError_t;
    fn hipEventSynchronize(e: hipEvent_t) -> hipError_t;
    fn hipEventElapsedTime(ms: *mut f32, a: hipEvent_t, b: hipEvent_t) -> hipError_t;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
}

const H2D: c_int = 1;
const D2H: c_int = 2;
// hipDeviceAttribute_t values (hip_runtime_api.h, HIP 6/7).
const ATTR_CU: c_int = 63; // hipDeviceAttributeMultiprocessorCount
const ATTR_WARP: c_int = 87; // hipDeviceAttributeWarpSize
const ATTR_INTEGRATED: c_int = 16; // hipDeviceAttributeIntegrated

fn check(e: hipError_t, what: &str) {
    if e != 0 {
        let msg = unsafe { CStr::from_ptr(hipGetErrorString(e)) };
        panic!("{what}: {} ({e})", msg.to_string_lossy());
    }
}

struct Module(hipModule_t);

impl Module {
    fn load(image: &[u8]) -> Self {
        let mut m = std::ptr::null_mut();
        check(
            unsafe { hipModuleLoadData(&mut m, image.as_ptr().cast()) },
            "hipModuleLoadData",
        );
        Module(m)
    }
    fn func(&self, name: &str) -> hipFunction_t {
        let c = std::ffi::CString::new(name).unwrap();
        let mut f = std::ptr::null_mut();
        check(
            unsafe { hipModuleGetFunction(&mut f, self.0, c.as_ptr()) },
            name,
        );
        f
    }
}

/// Launches `f` with the given argument pointers; returns device time (ms).
fn launch(f: hipFunction_t, grid: u32, block: u32, args: &mut [*mut c_void]) -> f32 {
    unsafe {
        let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
        check(hipEventCreate(&mut a), "event");
        check(hipEventCreate(&mut b), "event");
        check(hipEventRecord(a, std::ptr::null_mut()), "record");
        check(
            hipModuleLaunchKernel(
                f,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
            "launch",
        );
        check(hipEventRecord(b, std::ptr::null_mut()), "record");
        check(hipEventSynchronize(b), "sync");
        let mut ms = 0.0;
        check(hipEventElapsedTime(&mut ms, a, b), "elapsed");
        ms
    }
}

fn dmalloc(bytes: usize) -> *mut c_void {
    let mut p = std::ptr::null_mut();
    check(unsafe { hipMalloc(&mut p, bytes) }, "hipMalloc");
    p
}

fn upload<T>(v: &[T]) -> *mut c_void {
    let p = dmalloc(size_of_val(v));
    check(
        unsafe { hipMemcpy(p, v.as_ptr().cast(), size_of_val(v), H2D) },
        "H2D",
    );
    p
}

fn download<T: Copy + Default>(p: *mut c_void, n: usize) -> Vec<T> {
    let mut v = vec![T::default(); n];
    check(
        unsafe { hipMemcpy(v.as_mut_ptr().cast(), p, n * size_of::<T>(), D2H) },
        "D2H",
    );
    v
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Values over many binades, both signs, plus subnormal-producing ones.
    fn f64(&mut self) -> f64 {
        let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        let e = (self.next() % 40) as i32 - 20;
        let s = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
        s * (0.5 + u) * 2f64.powi(e)
    }
}

const OPS: usize = 8;

fn cpu_f32(x: f32, y: f32) -> [f32; OPS] {
    let (dx, dy, dz) = (x - y, y * 0.5 - x, x * y - 1.0);
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    [
        x + y,
        x * y,
        x / y,
        x.abs().sqrt(),
        d,
        (d - y) / d,
        x.min(y),
        (d - y.abs()).abs(),
    ]
}

fn cpu_f64(x: f64, y: f64) -> [f64; OPS] {
    let (dx, dy, dz) = (x - y, y * 0.5 - x, x * y - 1.0);
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    [
        x + y,
        x * y,
        x / y,
        x.abs().sqrt(),
        d,
        (d - y) / d,
        f64::from((x * 1e3) as f32),
        (d - y.abs()).abs(),
    ]
}

fn agreement(m: &Module) {
    let n = 1 << 22;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let a64: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
    let b64: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
    let mut a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
    let mut b32: Vec<f32> = b64.iter().map(|&v| (v * 1e-3) as f32).collect();
    // A few f32 subnormals and their neighbours.
    for k in 0..1024 {
        a32[k] = f32::from_bits(k as u32 * 977 + 1);
        b32[k + 1024] = f32::from_bits(k as u32 * 1301 + 7);
    }
    for (name, f, bits32) in [("f32", "arith_f32", true), ("f64", "arith_f64", false)] {
        let (pa, pb, words) = if bits32 {
            (upload(&a32), upload(&b32), 4)
        } else {
            (upload(&a64), upload(&b64), 8)
        };
        let out = dmalloc(n * OPS * words);
        let mut nn = n as c_int;
        let (mut xa, mut xb, mut xo) = (pa, pb, out);
        let mut args = [
            (&raw mut xa).cast(),
            (&raw mut xb).cast(),
            (&raw mut xo).cast(),
            (&raw mut nn).cast(),
        ];
        launch(m.func(f), n.div_ceil(256) as u32, 256, &mut args);
        let mut bad = [0usize; OPS];
        if bits32 {
            let got: Vec<u32> = download(out, n * OPS);
            for i in 0..n {
                let want = cpu_f32(a32[i], b32[i]);
                for k in 0..OPS {
                    let w = want[k].to_bits();
                    let g = got[OPS * i + k];
                    // NaN payloads may differ; count only non-NaN mismatches.
                    if w != g && !(want[k].is_nan() && f32::from_bits(g).is_nan()) {
                        bad[k] += 1;
                    }
                }
            }
        } else {
            let got: Vec<u64> = download(out, n * OPS);
            for i in 0..n {
                let want = cpu_f64(a64[i], b64[i]);
                for k in 0..OPS {
                    let w = want[k].to_bits();
                    let g = got[OPS * i + k];
                    if w != g && !(want[k].is_nan() && f64::from_bits(g).is_nan()) {
                        bad[k] += 1;
                    }
                }
            }
        }
        println!(
            "agree {name}: {n} inputs, mismatches [+ × ÷ √ dist (d−y)/d {} |d−|y||] = {bad:?}",
            if bits32 { "min" } else { "f64→f32" }
        );
        unsafe {
            hipFree(pa);
            hipFree(pb);
            hipFree(out);
        }
    }
}

fn rates(m: &Module, cus: i32) {
    let blocks = (cus * 64) as u32;
    let block = 256u32;
    let threads = blocks as usize * block as usize;
    let out = dmalloc(threads * 8);
    for (name, flops_per_step, iters) in [
        ("rate_muladd_f32", 2.0, 4096),
        ("rate_fma_f32", 2.0, 4096),
        ("rate_muladd_f64", 2.0, 512),
        ("rate_fma_f64", 2.0, 512),
        ("rate_sqrt_f32", 1.0, 1024),
        ("rate_sqrt_f64", 1.0, 128),
        ("rate_div_f32", 1.0, 1024),
        ("rate_div_f64", 1.0, 128),
    ] {
        let f = m.func(name);
        let mut po = out;
        let mut it = iters as c_int;
        let mut s32 = 0.999_f32;
        let mut s64 = 0.999_f64;
        let is64 = name.ends_with("f64");
        let mut best = f32::MAX;
        for _ in 0..5 {
            let mut args: [*mut c_void; 3] = [
                (&raw mut po).cast(),
                (&raw mut it).cast(),
                if is64 {
                    (&raw mut s64).cast()
                } else {
                    (&raw mut s32).cast()
                },
            ];
            best = best.min(launch(f, blocks, block, &mut args));
        }
        let ops = threads as f64 * 8.0 * iters as f64;
        println!(
            "{name:16} {:9.1} G{}/s  ({best:.2} ms)",
            ops * flops_per_step / (best as f64 * 1e6),
            if flops_per_step == 2.0 { "FLOP" } else { "op" }
        );
    }
    unsafe { hipFree(out) };
}

fn latency(m: &Module) {
    let f = m.func("empty");
    let mut args: [*mut c_void; 0] = [];
    for _ in 0..10 {
        launch(f, 1, 64, &mut args);
    }
    let t = Instant::now();
    let reps = 2000;
    for _ in 0..reps {
        unsafe {
            check(
                hipModuleLaunchKernel(
                    f,
                    1,
                    1,
                    1,
                    64,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "launch",
            );
        }
    }
    check(unsafe { hipDeviceSynchronize() }, "sync");
    let queued = t.elapsed().as_secs_f64() / reps as f64;
    let t = Instant::now();
    for _ in 0..200 {
        launch(f, 1, 64, &mut args);
    }
    let synced = t.elapsed().as_secs_f64() / 200.0;
    println!(
        "launch: {:.1} µs queued, {:.1} µs launch+sync round trip",
        queued * 1e6,
        synced * 1e6
    );
}

fn bandwidth(m: &Module, cus: i32) {
    let f = m.func("stream");
    let bytes = 1usize << 30;
    let n = (bytes / 16) as i64;
    let host_alloc = |flags: c_uint| {
        let mut p = std::ptr::null_mut();
        check(
            unsafe { hipHostMalloc(&mut p, bytes, flags) },
            "hipHostMalloc",
        );
        p
    };
    let managed = || {
        let mut p = std::ptr::null_mut();
        check(
            unsafe { hipMallocManaged(&mut p, bytes, 1) },
            "hipMallocManaged",
        );
        p
    };
    let kinds: [(&str, Box<dyn Fn() -> *mut c_void>); 4] = [
        ("device (hipMalloc)", Box::new(|| dmalloc(bytes))),
        (
            "pinned host (hipHostMalloc)",
            Box::new(move || host_alloc(0)),
        ),
        (
            "pinned host, non-coherent",
            Box::new(move || host_alloc(0x8000_0000)),
        ),
        ("managed (hipMallocManaged)", Box::new(managed)),
    ];
    for (name, alloc) in &kinds {
        let (src, dst) = (alloc(), alloc());
        let (mut s, mut d, mut nn) = (src, dst, n);
        let mut best = f32::MAX;
        for _ in 0..5 {
            let mut args: [*mut c_void; 3] = [
                (&raw mut s).cast(),
                (&raw mut d).cast(),
                (&raw mut nn).cast(),
            ];
            best = best.min(launch(f, (cus * 32) as u32, 256, &mut args));
        }
        println!(
            "stream {name:28} {:6.1} GB/s (read + write)",
            2.0 * bytes as f64 / (best as f64 * 1e6)
        );
        unsafe {
            if name.starts_with("device") || name.starts_with("managed") {
                hipFree(src);
                hipFree(dst);
            } else {
                hipHostFree(src);
                hipHostFree(dst);
            }
        }
    }
    // CPU write into pinned memory, device read: the zero-copy hand-off.
    let p = host_alloc(0);
    let t = Instant::now();
    unsafe { std::ptr::write_bytes(p.cast::<u8>(), 1, bytes) };
    println!(
        "CPU memset of pinned host memory: {:.1} GB/s",
        bytes as f64 / t.elapsed().as_secs_f64() / 1e9
    );
    unsafe { hipHostFree(p) };
}

fn main() {
    let image = include_bytes!(concat!(env!("OUT_DIR"), "/spike.hsaco"));
    unsafe {
        check(hipInit(0), "hipInit");
        let mut n = 0;
        check(hipGetDeviceCount(&mut n), "count");
        assert!(n > 0, "no HIP device");
        check(hipSetDevice(0), "set");
        let mut name = [0 as c_char; 256];
        check(hipDeviceGetName(name.as_mut_ptr(), 256, 0), "name");
        let (mut cus, mut warp, mut integ) = (0, 0, 0);
        check(hipDeviceGetAttribute(&mut cus, ATTR_CU, 0), "cu");
        check(hipDeviceGetAttribute(&mut warp, ATTR_WARP, 0), "warp");
        check(
            hipDeviceGetAttribute(&mut integ, ATTR_INTEGRATED, 0),
            "integrated",
        );
        let (mut free, mut total) = (0, 0);
        check(hipMemGetInfo(&mut free, &mut total), "meminfo");
        println!(
            "device: {} | {cus} CUs | wave {warp} | integrated {integ} | {:.1} / {:.1} GiB free",
            CStr::from_ptr(name.as_ptr()).to_string_lossy(),
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64
        );
        let args: Vec<String> = std::env::args().collect();
        if args.len() > 2 && args[1] == "ses" {
            for path in &args[2..] {
                ses_bench(path);
            }
            return;
        }
        let m = Module::load(image);
        agreement(&m);
        rates(&m, cus);
        latency(&m);
        bandwidth(&m, cus);
    }
}

/// Concatenated sub-blocks of a `SESMESH_DUMP` file, blockers made absolute.
#[derive(Default)]
struct Dump {
    desc: Vec<u32>,
    f: [Vec<f32>; 25],
    s_start: Vec<u32>,
    s_blk: Vec<u32>,
    c_start: Vec<u32>,
    c_blk: Vec<u32>,
    cpu: Vec<f32>,
    probe: f32,
    cutoff2: f32,
}

// Indices into `Dump::f`.
const XS: usize = 0; // xs ys zs
const NX: usize = 3; // nx ny nz nr2 nr2tol
const SX: usize = 8; // sx sy sz sr
const CX: usize = 12; // cx cy cz ux uy uz ex ey ez rho
const VX: usize = 22; // vx vy vz

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn word(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.at..self.at + 4].try_into().unwrap());
        self.at += 4;
        v
    }
    fn take(&mut self, dst: &mut Vec<f32>, n: usize) {
        for _ in 0..n {
            dst.push(f32::from_bits(self.word()));
        }
    }
}

fn read_dump(path: &str) -> Dump {
    let bytes = std::fs::read(path).expect("dump");
    let mut d = Dump::default();
    d.s_start.push(0);
    d.c_start.push(0);
    let mut cur = Cursor { b: &bytes, at: 0 };
    let mut words = Vec::new();
    while cur.at < bytes.len() {
        let h: Vec<usize> = (0..8).map(|_| cur.word() as usize).collect();
        let (m, nin, nn, ns, nsb, nc, ncb, nv) = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        d.probe = f32::from_bits(cur.word());
        d.cutoff2 = f32::from_bits(cur.word());
        let (samp0, near0) = (d.f[XS].len(), d.f[NX].len());
        let (s0, c0, v0) = (d.f[SX].len(), d.f[CX].len(), d.f[VX].len());
        for k in 0..3 {
            cur.take(&mut d.f[XS + k], m);
        }
        for k in 0..5 {
            cur.take(&mut d.f[NX + k], nn);
        }
        for k in 0..4 {
            cur.take(&mut d.f[SX + k], ns);
        }
        words.clear();
        for _ in 0..ns + 1 {
            words.push(cur.word());
        }
        let base = *d.s_start.last().unwrap();
        d.s_start.extend(words[1..].iter().map(|&o| base + o));
        for _ in 0..nsb {
            let k = cur.word();
            d.s_blk.push(near0 as u32 + k);
        }
        for k in 0..10 {
            cur.take(&mut d.f[CX + k], nc);
        }
        words.clear();
        for _ in 0..nc + 1 {
            words.push(cur.word());
        }
        let base = *d.c_start.last().unwrap();
        d.c_start.extend(words[1..].iter().map(|&o| base + o));
        for _ in 0..ncb {
            let k = cur.word();
            d.c_blk.push(near0 as u32 + k);
        }
        for k in 0..3 {
            cur.take(&mut d.f[VX + k], nv);
        }
        cur.take(&mut d.cpu, m);
        d.desc
            .extend([samp0, m, near0, nin, s0, ns, c0, nc, v0, nv].map(|v| v as u32));
    }
    d
}

#[repr(C)]
struct Arrays {
    desc: *mut c_void,
    p: [*mut c_void; 32],
    probe: f32,
    cutoff2: f32,
}

fn ses_bench(path: &str) {
    let t = Instant::now();
    let d = read_dump(path);
    let nsb = d.desc.len() / 10;
    let n = d.cpu.len();
    println!(
        "{path}: {nsb} sub-blocks, {n} samples, {} near, {} S, {} C, {} V (read {:.1} s)",
        d.f[NX].len(),
        d.f[SX].len(),
        d.f[CX].len(),
        d.f[VX].len(),
        t.elapsed().as_secs_f64()
    );
    // Device arrays in the order of `struct Arrays` in ses.hip.
    let mut p: Vec<*mut c_void> = Vec::new();
    for k in XS..XS + 3 {
        p.push(upload(&d.f[k]));
    }
    for k in NX..NX + 5 {
        p.push(upload(&d.f[k]));
    }
    for k in SX..SX + 4 {
        p.push(upload(&d.f[k]));
    }
    p.push(upload(&d.s_start));
    p.push(upload(&d.s_blk));
    for k in CX..CX + 10 {
        p.push(upload(&d.f[k]));
    }
    p.push(upload(&d.c_start));
    p.push(upload(&d.c_blk));
    for k in VX..VX + 3 {
        p.push(upload(&d.f[k]));
    }
    let out = dmalloc(n * 4);
    p.push(out);
    let mut a = Arrays {
        desc: upload(&d.desc),
        p: [std::ptr::null_mut(); 32],
        probe: d.probe,
        cutoff2: d.cutoff2,
    };
    a.p[..p.len()].copy_from_slice(&p);
    // `struct Arrays` in HIP has 30 pointers after `desc`.
    assert_eq!(p.len(), 30);
    #[repr(C)]
    struct Packed {
        desc: *mut c_void,
        p: [*mut c_void; 30],
        probe: f32,
        cutoff2: f32,
    }
    let mut packed = Packed {
        desc: a.desc,
        p: std::array::from_fn(|i| a.p[i]),
        probe: a.probe,
        cutoff2: a.cutoff2,
    };
    for (obj, wave) in [
        (
            &include_bytes!(concat!(env!("OUT_DIR"), "/ses.hsaco"))[..],
            32,
        ),
        (
            &include_bytes!(concat!(env!("OUT_DIR"), "/ses64.hsaco"))[..],
            64,
        ),
    ] {
        let m = Module::load(obj);
        for (kname, block) in [
            ("ses_kernel", 64u32),
            ("ses_kernel", 256),
            ("ses_kernel_f64", 64),
            ("ses_kernel_f64", 256),
        ] {
            let f = m.func(kname);
            let mut best = f32::MAX;
            for _ in 0..5 {
                let mut args: [*mut c_void; 1] = [(&raw mut packed).cast()];
                best = best.min(launch(f, nsb as u32, block, &mut args));
            }
            let got: Vec<f32> = download(out, n);
            let bad = got
                .iter()
                .zip(&d.cpu)
                .filter(|(g, c)| g.to_bits() != c.to_bits())
                .count();
            let worst = got
                .iter()
                .zip(&d.cpu)
                .map(|(g, c)| (g - c).abs())
                .fold(0.0f32, f32::max);
            println!(
                "  {kname:14} wave{wave} block {block:3}: {best:8.2} ms, {:6.1} ns/sample, mismatches {bad} (max |Δ| {worst:e})",
                best as f64 * 1e6 / n as f64
            );
        }
    }
    for q in p {
        unsafe { hipFree(q) };
    }
    unsafe { hipFree(a.desc) };
}
