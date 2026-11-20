// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AMD GPUs through HIP (ROCm).
//!
//! [`Runtime::load`] opens the HIP runtime library (`libamdhip64`), and
//! [`Runtime::open`] opens a device:
//!
//! 1. It reads the device's properties.
//! 2. It loads the first embedded code object that runs on the device. The
//!    build compiles one per ROCm generic target (RDNA 2, 3 and 4 by
//!    default, `SESMESH_HIP_ARCHS` to change).
//! 3. It runs a self-test. The device is used only if its IEEE operations
//!    agree with the CPU's bit for bit.
//!
//! ```no_run
//! use sesmesh_gpu::hip::Runtime;
//!
//! let device = Runtime::load()?.open(0)?;
//! println!("{} ({})", device.info().name, device.info().arch);
//! # Ok::<(), sesmesh_gpu::Error>(())
//! ```

#[cfg(not(target_pointer_width = "64"))]
compile_error!("the HIP back end needs a 64-bit target");

mod ffi;
mod memory;
mod resident;
mod selftest;
mod stream;

use std::ffi::{CString, c_int, c_void};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

pub use memory::{DeviceBuffer, DeviceCopy, HostBuffer};
pub use resident::{
    ArrangementRef, Batch, BatchParams, BatchTimes, EDGE_CONVERGED, FEATURE_BEYOND, FEATURE_CIRCLE,
    FEATURE_SHIFT, FEATURE_SOLVENT, FEATURE_SPHERE, FEATURE_VERTEX, GridRef, REC_WORDS, RefineRef,
    RefineTimes, Resident, STATE_IN, STATE_OUT, STATE_SAMPLE,
};
pub use selftest::SelfTestReport;
pub use stream::{Event, Stream};

use crate::{Error, Result};
use ffi::{Api, Handle};

include!(concat!(env!("OUT_DIR"), "/hip_code_objects.rs"));

/// Values checked by the self-test that [`Runtime::open`] runs.
const OPEN_SELF_TEST_VALUES: usize = 4096;

/// Wave size the kernels are compiled for (`-mwavefrontsize64` on RDNA, the
/// native size on CDNA). Block sizes are multiples of it.
pub const KERNEL_WAVE_SIZE: u32 = 64;

/// The offload targets this build has code objects for, in load order
/// (empty when built without kernels).
#[must_use]
pub fn code_object_targets() -> Vec<&'static str> {
    CODE_OBJECTS.iter().map(|&(t, _)| t).collect()
}

/// A loaded HIP runtime library. Cheap to clone.
#[derive(Clone)]
pub struct Runtime {
    api: Arc<Api>,
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runtime")
            .field("library", &self.api.path)
            .finish()
    }
}

impl Runtime {
    /// Loads the HIP runtime: `libamdhip64.so.7`, `.so.6` or `.so` through the
    /// dynamic loader, then from `$ROCM_PATH/lib`, `$HIP_PATH/lib` and
    /// `/opt/rocm/lib` (`amdhip64*.dll` on Windows).
    ///
    /// # Errors
    ///
    /// [`Error::NoRuntime`] when none loads, [`Error::BadRuntime`] when the
    /// library lacks a function sesmesh needs.
    pub fn load() -> Result<Self> {
        Self::load_from(&ffi::candidates())
    }

    /// Loads the HIP runtime from the first of `paths` that opens.
    ///
    /// # Errors
    ///
    /// As [`Runtime::load`].
    pub fn load_from<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        Ok(Self {
            api: Arc::new(ffi::load(paths)?),
        })
    }

    /// Path or name the library was loaded from.
    #[must_use]
    pub fn library(&self) -> &str {
        &self.api.path
    }

    /// The runtime version, e.g. `7.16.26385`.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the runtime cannot report it.
    pub fn version(&self) -> Result<String> {
        let mut v: c_int = 0;
        // SAFETY: valid out-pointer.
        let code = unsafe { (self.api.hipRuntimeGetVersion)(&raw mut v) };
        self.api.check("hipRuntimeGetVersion", code)?;
        Ok(format!(
            "{}.{}.{}",
            v / 10_000_000,
            (v / 100_000) % 100,
            v % 100_000
        ))
    }

    fn init(&self) -> Result<()> {
        // SAFETY: `hipInit(0)` has no preconditions and may be repeated.
        let code = unsafe { (self.api.hipInit)(0) };
        match code {
            ffi::ERROR_NO_DEVICE | ffi::ERROR_INSUFFICIENT_DRIVER => {
                let e = self.api.check("hipInit", code).unwrap_err();
                Err(Error::NoDevice(e.to_string()))
            }
            _ => self.api.check("hipInit", code),
        }
    }

    /// Number of HIP devices.
    ///
    /// # Errors
    ///
    /// [`Error::NoDevice`] when the driver is missing or has no device.
    pub fn device_count(&self) -> Result<usize> {
        self.init()?;
        let mut n: c_int = 0;
        // SAFETY: valid out-pointer.
        let code = unsafe { (self.api.hipGetDeviceCount)(&raw mut n) };
        if code == ffi::ERROR_NO_DEVICE {
            return Ok(0);
        }
        self.api.check("hipGetDeviceCount", code)?;
        Ok(usize::try_from(n).unwrap_or(0))
    }

    /// Opens device `index`: reads its properties, loads the embedded code
    /// object that runs on it and runs the self-test.
    ///
    /// # Errors
    ///
    /// * [`Error::NoDevice`]: no such device;
    /// * [`Error::NoKernels`]: the build has no code objects;
    /// * [`Error::Unsupported`]: none of them runs on the device;
    /// * [`Error::SelfTest`]: the device's arithmetic differs from the CPU's;
    /// * [`Error::Call`]: a runtime call failed.
    pub fn open(&self, index: usize) -> Result<Device> {
        let count = self.device_count()?;
        if index >= count {
            return Err(Error::NoDevice(format!(
                "device {index} requested, {count} present"
            )));
        }
        let ordinal = c_int::try_from(index).expect("index < device count");
        let api = &self.api;
        // SAFETY: `ordinal` is a valid device index.
        api.check("hipSetDevice", unsafe { (api.hipSetDevice)(ordinal) })?;
        let props = self.properties(ordinal)?;
        if CODE_OBJECTS.is_empty() {
            return Err(Error::NoKernels);
        }
        let mut loaded = None;
        for &(target, object) in CODE_OBJECTS {
            let mut module: Handle = std::ptr::null_mut();
            // SAFETY: `object` is a complete code object emitted by hipcc
            // and embedded by build.rs; `module` is a valid out-pointer.
            let code = unsafe { (api.hipModuleLoadData)(&raw mut module, object.as_ptr().cast()) };
            if code != 0 {
                continue; // not for this device
            }
            match Kernels::get(api, module) {
                Ok(k) => {
                    loaded = Some((target, module, k));
                    break;
                }
                Err(_) => {
                    // SAFETY: `module` was loaded above and is not used again.
                    unsafe { (api.hipModuleUnload)(module) };
                }
            }
        }
        let Some((target, module, kernels)) = loaded else {
            return Err(Error::Unsupported {
                arch: props.arch,
                built: code_object_targets().join(", "),
            });
        };
        let info = DeviceInfo {
            index,
            name: props.name,
            arch: props.arch,
            compute_units: props.compute_units,
            warp_size: props.warp_size,
            integrated: props.integrated,
            memory: props.memory,
            code_object: target,
            runtime: self.version().unwrap_or_default(),
        };
        let device = Device {
            runtime: self.clone(),
            ordinal,
            module,
            kernels,
            info,
        };
        let report = device.self_test(OPEN_SELF_TEST_VALUES, 1)?;
        if !report.passed() {
            return Err(Error::SelfTest(report.to_string()));
        }
        Ok(device)
    }

    fn properties(&self, ordinal: c_int) -> Result<Properties> {
        // `hipDeviceProp_tR0600` is 1 472 bytes; read into a larger, aligned
        // buffer.
        let mut raw = [0u64; 512];
        debug_assert!(size_of_val(&raw) >= ffi::props::SIZE);
        // SAFETY: the buffer is larger than `hipDeviceProp_tR0600` and
        // suitably aligned; `ordinal` is a valid device.
        self.api.check("hipGetDevicePropertiesR0600", unsafe {
            (self.api.hipGetDevicePropertiesR0600)(raw.as_mut_ptr().cast(), ordinal)
        })?;
        let bytes: Vec<u8> = raw.iter().flat_map(|w| w.to_ne_bytes()).collect();
        let text = |at: usize| {
            let field = &bytes[at..at + ffi::props::STRING_LEN];
            let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
            String::from_utf8_lossy(&field[..end]).into_owned()
        };
        let int = |at: usize| i32::from_ne_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
        let props = Properties {
            name: text(ffi::props::NAME),
            arch: text(ffi::props::GCN_ARCH_NAME),
            memory: u64::from_ne_bytes(
                bytes[ffi::props::TOTAL_GLOBAL_MEM..ffi::props::TOTAL_GLOBAL_MEM + 8]
                    .try_into()
                    .expect("8 bytes"),
            ),
            warp_size: u32::try_from(int(ffi::props::WARP_SIZE)).unwrap_or(0),
            compute_units: u32::try_from(int(ffi::props::MULTIPROCESSOR_COUNT)).unwrap_or(0),
            integrated: int(ffi::props::INTEGRATED) != 0,
        };
        if !props.arch.starts_with("gfx") || !matches!(props.warp_size, 32 | 64) {
            return Err(Error::BadRuntime {
                library: self.api.path.clone(),
                reason: format!(
                    "unexpected hipDeviceProp_tR0600 layout (arch {:?}, warp size {})",
                    props.arch, props.warp_size
                ),
            });
        }
        Ok(props)
    }
}

struct Properties {
    name: String,
    arch: String,
    memory: u64,
    warp_size: u32,
    compute_units: u32,
    integrated: bool,
}

/// What a [`Device`] is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// Device index (HIP ordinal).
    pub index: usize,
    /// Marketing name, e.g. `AMD Radeon 8060S Graphics`.
    pub name: String,
    /// Architecture with target features, e.g. `gfx1151` or
    /// `gfx90a:sramecc+:xnack-`.
    pub arch: String,
    /// Multiprocessors as HIP counts them: compute units, or work-group
    /// processors (two CUs) on RDNA in WGP mode.
    pub compute_units: u32,
    /// Native wave size (32 on RDNA, 64 on GCN and CDNA). The kernels use
    /// [`KERNEL_WAVE_SIZE`].
    pub warp_size: u32,
    /// Integrated GPU sharing system memory (APU): host buffers need no copy.
    pub integrated: bool,
    /// Device memory in bytes (on an APU, the memory the driver reports).
    pub memory: u64,
    /// Offload target of the code object in use, e.g. `gfx11-generic`.
    pub code_object: &'static str,
    /// HIP runtime version.
    pub runtime: String,
}

/// The kernels resolved from the loaded code object.
#[derive(Clone, Copy)]
struct Kernels {
    selftest_f32: Handle,
    resident_count: Handle,
    resident_fill: Handle,
    resident_sample: Handle,
    resident_edge_count: Handle,
    resident_refine: Handle,
}

impl Kernels {
    fn get(api: &Api, module: Handle) -> Result<Self> {
        let func = |name: &str| {
            let c = CString::new(name).expect("no NUL in kernel names");
            let mut f: Handle = std::ptr::null_mut();
            // SAFETY: `module` is loaded; valid out-pointer and C string.
            api.check("hipModuleGetFunction", unsafe {
                (api.hipModuleGetFunction)(&raw mut f, module, c.as_ptr())
            })
            .map(|()| f)
        };
        Ok(Self {
            selftest_f32: func("sesmesh_selftest_f32")?,
            resident_count: func("sesmesh_resident_count")?,
            resident_fill: func("sesmesh_resident_fill")?,
            resident_sample: func("sesmesh_resident_sample")?,
            resident_edge_count: func("sesmesh_resident_edge_count")?,
            resident_refine: func("sesmesh_resident_refine")?,
        })
    }
}

/// An open HIP device with sesmesh's kernels loaded.
///
/// Every operation selects the device first, so a `Device` can be used from
/// any thread.
pub struct Device {
    runtime: Runtime,
    ordinal: c_int,
    module: Handle,
    kernels: Kernels,
    info: DeviceInfo,
}

// SAFETY: the HIP runtime is thread-safe; module and function handles are
// process-wide, and every call selects the device first (`bind`).
unsafe impl Send for Device {}
// SAFETY: as above; `Device` has no interior mutability on the Rust side.
unsafe impl Sync for Device {}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").field("info", &self.info).finish()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.bind().is_ok() {
            // SAFETY: the module was loaded by `Runtime::open` and its
            // functions are not used after this.
            unsafe { (self.runtime.api.hipModuleUnload)(self.module) };
        }
    }
}

impl Device {
    /// Properties of the device and the code object in use.
    #[must_use]
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The runtime this device was opened with.
    #[must_use]
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Free and total device memory in bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the runtime cannot report it.
    pub fn memory(&self) -> Result<(u64, u64)> {
        self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: valid out-pointers.
        self.api().check("hipMemGetInfo", unsafe {
            (self.api().hipMemGetInfo)(&raw mut free, &raw mut total)
        })?;
        Ok((free as u64, total as u64))
    }

    /// Runs the arithmetic self-test on `values` pseudo-random inputs (from
    /// `seed`, including subnormals): the device computes f32 `+ − × ÷ √`
    /// and a few compound expressions, on device and on pinned host memory,
    /// and the host compares the bits with its own results.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if a runtime call fails; mismatches are reported in
    /// the [`SelfTestReport`], not as an error.
    pub fn self_test(&self, values: usize, seed: u64) -> Result<SelfTestReport> {
        selftest::run(self, values, seed)
    }

    /// Waits for all work on the device.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] with the error of any failed work.
    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        // SAFETY: no preconditions.
        self.api().check("hipDeviceSynchronize", unsafe {
            (self.api().hipDeviceSynchronize)()
        })
    }

    pub(crate) fn api(&self) -> &Api {
        &self.runtime.api
    }

    /// Makes this device current on the calling thread.
    pub(crate) fn bind(&self) -> Result<()> {
        bind(&self.runtime.api, self.ordinal)
    }

    pub(crate) fn ordinal(&self) -> c_int {
        self.ordinal
    }

    /// Launches `function` on a 1-D grid and waits for it.
    ///
    /// # Safety
    ///
    /// `args` must match the kernel's parameters in number, order and type,
    /// and every pointer passed must stay valid, and not be accessed by the
    /// host, until this returns.
    pub(crate) unsafe fn launch_sync(
        &self,
        function: Handle,
        grid: u32,
        block: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        self.bind()?;
        let api = self.api();
        // SAFETY: the caller guarantees `args`; null stream and extra are
        // allowed.
        api.check("hipModuleLaunchKernel", unsafe {
            (api.hipModuleLaunchKernel)(
                function,
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
            )
        })?;
        self.synchronize()
    }
}

pub(crate) fn bind(api: &Api, ordinal: c_int) -> Result<()> {
    // SAFETY: `ordinal` came from a successful `Runtime::open`.
    api.check("hipSetDevice", unsafe { (api.hipSetDevice)(ordinal) })
}
