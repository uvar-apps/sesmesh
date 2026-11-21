// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NVIDIA GPUs through the CUDA driver API (docs/06-gpu-cuda.md).
//!
//! [`Runtime::load`] opens the driver library (`libcuda`), and
//! [`Runtime::open`] opens a device:
//!
//! 1. It reads the device's properties and makes its primary context
//!    current.
//! 2. It loads the embedded fat binary: machine code for the architectures
//!    the build compiled (`SESMESH_CUDA_ARCHS`, by default Turing to
//!    Blackwell), or the PTX of the lowest one, which the driver compiles
//!    for newer GPUs.
//! 3. It runs a self-test. The device is used only if its IEEE operations
//!    agree with the CPU's bit for bit.
//!
//! The kernels are those of the HIP back end (`kernels/sesmesh.cuh`).
//! Batches live in device memory and are copied to and from pinned host
//! buffers (a discrete GPU reads host memory only over the bus).
//!
//! ```no_run
//! use sesmesh_gpu::cuda::Runtime;
//!
//! let device = Runtime::load()?.open(0)?;
//! println!("{} ({})", device.info().name, device.info().arch);
//! # Ok::<(), sesmesh_gpu::Error>(())
//! ```

#[cfg(not(target_pointer_width = "64"))]
compile_error!("the CUDA back end needs a 64-bit target");

mod ffi;
mod memory;
mod resident;
mod selftest;
mod stream;

use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

pub use crate::DeviceCopy;
pub use crate::batch::{
    ArrangementRef, BatchParams, BatchTimes, EDGE_CONVERGED, FEATURE_BEYOND, FEATURE_CIRCLE,
    FEATURE_SHIFT, FEATURE_SOLVENT, FEATURE_SPHERE, FEATURE_VERTEX, GridRef, REC_WORDS, RefineRef,
    RefineTimes, STATE_IN, STATE_OUT, STATE_SAMPLE,
};
pub use crate::selftest::SelfTestReport;
pub use memory::{DeviceBuffer, HostBuffer};
pub use resident::{Batch, Resident};
pub use stream::{Event, Stream};

use crate::{Error, Result};
use ffi::{Api, CuDevice, Handle};

include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));

/// Values checked by the self-test that [`Runtime::open`] runs.
const OPEN_SELF_TEST_VALUES: usize = 4096;

/// The architectures this build has machine code for, as `sm_XY` (empty
/// when built without kernels); the first also has PTX.
#[must_use]
pub fn architectures() -> Vec<String> {
    ARCHS.iter().map(|a| format!("sm_{a}")).collect()
}

/// A loaded CUDA driver library. Cheap to clone.
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
    /// Loads the CUDA driver: `libcuda.so.1` or `libcuda.so` through the
    /// dynamic loader (`nvcuda.dll` on Windows).
    ///
    /// # Errors
    ///
    /// [`Error::NoRuntime`] when none loads, [`Error::BadRuntime`] when the
    /// library lacks a function sesmesh needs.
    pub fn load() -> Result<Self> {
        Self::load_from(&ffi::candidates())
    }

    /// Loads the CUDA driver from the first of `paths` that opens.
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

    /// The CUDA version the driver supports, e.g. `13.4`.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the driver cannot report it.
    pub fn version(&self) -> Result<String> {
        let mut v: c_int = 0;
        // SAFETY: valid out-pointer.
        let code = unsafe { (self.api.cuDriverGetVersion)(&raw mut v) };
        self.api.check("cuDriverGetVersion", code)?;
        Ok(format!("{}.{}", v / 1000, (v % 1000) / 10))
    }

    fn init(&self) -> Result<()> {
        // SAFETY: `cuInit(0)` has no preconditions and may be repeated.
        let code = unsafe { (self.api.cuInit)(0) };
        match code {
            ffi::ERROR_NO_DEVICE | ffi::ERROR_INSUFFICIENT_DRIVER => {
                let e = self.api.check("cuInit", code).unwrap_err();
                Err(Error::NoDevice(e.to_string()))
            }
            _ => self.api.check("cuInit", code),
        }
    }

    /// Number of CUDA devices.
    ///
    /// # Errors
    ///
    /// [`Error::NoDevice`] when the driver has no device or is too old.
    pub fn device_count(&self) -> Result<usize> {
        self.init()?;
        let mut n: c_int = 0;
        // SAFETY: valid out-pointer.
        let code = unsafe { (self.api.cuDeviceGetCount)(&raw mut n) };
        if code == ffi::ERROR_NO_DEVICE {
            return Ok(0);
        }
        self.api.check("cuDeviceGetCount", code)?;
        Ok(usize::try_from(n).unwrap_or(0))
    }

    /// Opens device `index`: reads its properties, makes its primary context
    /// current, loads the embedded kernels and runs the self-test.
    ///
    /// # Errors
    ///
    /// * [`Error::NoDevice`]: no such device;
    /// * [`Error::NoKernels`]: the build has no GPU code;
    /// * [`Error::Unsupported`]: none of it runs on the device;
    /// * [`Error::SelfTest`]: the device's arithmetic differs from the CPU's;
    /// * [`Error::Call`]: a driver call failed.
    pub fn open(&self, index: usize) -> Result<Device> {
        let count = self.device_count()?;
        if index >= count {
            return Err(Error::NoDevice(format!(
                "device {index} requested, {count} present"
            )));
        }
        let api = &self.api;
        let ordinal = c_int::try_from(index).expect("index < device count");
        let mut device: CuDevice = 0;
        // SAFETY: valid out-pointer and ordinal.
        api.check("cuDeviceGet", unsafe {
            (api.cuDeviceGet)(&raw mut device, ordinal)
        })?;
        let props = self.properties(device)?;
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer and device.
        api.check("cuDevicePrimaryCtxRetain", unsafe {
            (api.cuDevicePrimaryCtxRetain)(&raw mut handle, device)
        })?;
        // From here `Context::drop` releases it.
        let context = Arc::new(Context {
            runtime: self.clone(),
            device,
            handle,
        });
        let mut opened = Device {
            context,
            module: std::ptr::null_mut(),
            kernels: None,
            info: DeviceInfo {
                index,
                name: props.name,
                arch: format!("sm_{}", props.capability),
                compute_units: props.compute_units,
                warp_size: props.warp_size,
                integrated: props.integrated,
                memory: props.memory,
                code: String::new(),
                runtime: self.version().unwrap_or_default(),
            },
        };
        opened.bind()?;
        if FATBIN.is_empty() {
            return Err(Error::NoKernels);
        }
        if !props.unified_addressing {
            return Err(Error::Unsupported {
                arch: opened.info.arch.clone(),
                built: "devices with unified addressing".to_owned(),
            });
        }
        // The driver wants the image aligned; `include_bytes!` is not.
        let mut image = vec![0u64; FATBIN.len().div_ceil(8)];
        for (w, chunk) in image.iter_mut().zip(FATBIN.chunks(8)) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            *w = u64::from_ne_bytes(b);
        }
        let mut module: Handle = std::ptr::null_mut();
        // SAFETY: `image` holds the complete fat binary nvcc wrote (build.rs),
        // 8-byte aligned; `module` is a valid out-pointer; the context is
        // current.
        let code = unsafe { (api.cuModuleLoadData)(&raw mut module, image.as_ptr().cast()) };
        if matches!(
            code,
            ffi::ERROR_NO_BINARY_FOR_GPU
                | ffi::ERROR_INVALID_PTX
                | ffi::ERROR_UNSUPPORTED_PTX_VERSION
        ) {
            return Err(Error::Unsupported {
                arch: opened.info.arch.clone(),
                built: format!("{} (PTX {})", architectures().join(", "), ptx_arch()),
            });
        }
        api.check("cuModuleLoadData", code)?;
        opened.module = module;
        opened.kernels = Some(Kernels::get(api, module)?);
        opened.info.code = code_for(props.capability);
        let report = opened.self_test(OPEN_SELF_TEST_VALUES, 1)?;
        if !report.passed() {
            return Err(Error::SelfTest(report.to_string()));
        }
        Ok(opened)
    }

    fn properties(&self, device: CuDevice) -> Result<Properties> {
        let api = &self.api;
        let mut name = [0 as c_char; 256];
        // SAFETY: the buffer holds `len` bytes; valid device.
        api.check("cuDeviceGetName", unsafe {
            (api.cuDeviceGetName)(name.as_mut_ptr(), 256, device)
        })?;
        let name = name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c.cast_unsigned())
            .collect::<Vec<u8>>();
        let attribute = |a: i32| -> Result<i32> {
            let mut v: c_int = 0;
            // SAFETY: valid out-pointer, attribute and device.
            api.check("cuDeviceGetAttribute", unsafe {
                (api.cuDeviceGetAttribute)(&raw mut v, a, device)
            })?;
            Ok(v)
        };
        let mut memory = 0usize;
        // SAFETY: valid out-pointer and device.
        api.check("cuDeviceTotalMem_v2", unsafe {
            (api.cuDeviceTotalMem_v2)(&raw mut memory, device)
        })?;
        let major = attribute(ffi::attr::COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(ffi::attr::COMPUTE_CAPABILITY_MINOR)?;
        Ok(Properties {
            name: String::from_utf8_lossy(&name).into_owned(),
            capability: u32::try_from(10 * major + minor).unwrap_or(0),
            memory: memory as u64,
            warp_size: u32::try_from(attribute(ffi::attr::WARP_SIZE)?).unwrap_or(0),
            compute_units: u32::try_from(attribute(ffi::attr::MULTIPROCESSOR_COUNT)?).unwrap_or(0),
            integrated: attribute(ffi::attr::INTEGRATED)? != 0,
            unified_addressing: attribute(ffi::attr::UNIFIED_ADDRESSING)? != 0,
        })
    }
}

/// The PTX architecture of this build (`compute_XY`), or `none`.
fn ptx_arch() -> String {
    ARCHS
        .first()
        .map_or_else(|| "none".to_owned(), |a| format!("compute_{a}"))
}

/// Which code of the fat binary runs on compute capability `cc` (× 10):
/// machine code for its architecture, or else the driver's compilation of
/// the PTX.
fn code_for(cc: u32) -> String {
    if ARCHS.contains(&cc) {
        format!("sm_{cc}")
    } else {
        format!("{} PTX", ptx_arch())
    }
}

struct Properties {
    name: String,
    capability: u32,
    memory: u64,
    warp_size: u32,
    compute_units: u32,
    integrated: bool,
    unified_addressing: bool,
}

/// What a [`Device`] is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// Device index (CUDA ordinal, as `nvidia-smi` and
    /// `CUDA_VISIBLE_DEVICES` count, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
    pub index: usize,
    /// Marketing name, e.g. `NVIDIA GeForce RTX 2070 with Max-Q Design`.
    pub name: String,
    /// Architecture, `sm_` and the compute capability, e.g. `sm_75`.
    pub arch: String,
    /// Streaming multiprocessors.
    pub compute_units: u32,
    /// Warp size (32).
    pub warp_size: u32,
    /// Integrated GPU sharing system memory.
    pub integrated: bool,
    /// Device memory in bytes.
    pub memory: u64,
    /// The code of the fat binary in use: machine code (`sm_75`) or the
    /// driver's compilation of the PTX (`compute_75 PTX`).
    pub code: String,
    /// The CUDA version the driver supports, e.g. `13.4`.
    pub runtime: String,
}

/// The kernels resolved from the loaded module.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
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
            api.check("cuModuleGetFunction", unsafe {
                (api.cuModuleGetFunction)(&raw mut f, module, c.as_ptr())
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

/// A retained primary context, released when the last [`Device`], buffer,
/// stream or event holding it drops.
pub(crate) struct Context {
    runtime: Runtime,
    device: CuDevice,
    handle: Handle,
}

// SAFETY: a CUDA context may be current on several threads at once; the
// handle is only passed to the thread-safe driver API.
unsafe impl Send for Context {}
// SAFETY: as above; no interior mutability on the Rust side.
unsafe impl Sync for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: retained once by `Runtime::open`; nothing holds the
        // context any more (this is the last reference).
        unsafe { (self.runtime.api.cuDevicePrimaryCtxRelease_v2)(self.device) };
    }
}

impl Context {
    pub(crate) fn api(&self) -> &Api {
        &self.runtime.api
    }

    /// Makes the context current on the calling thread.
    pub(crate) fn bind(&self) -> Result<()> {
        // SAFETY: the context is retained while `self` lives.
        self.api().check("cuCtxSetCurrent", unsafe {
            (self.api().cuCtxSetCurrent)(self.handle)
        })
    }
}

/// An open CUDA device with sesmesh's kernels loaded.
///
/// Every operation makes the device's primary context current first, so a
/// `Device` can be used from any thread.
pub struct Device {
    context: Arc<Context>,
    module: Handle,
    kernels: Option<Kernels>,
    info: DeviceInfo,
}

// SAFETY: the CUDA driver API is thread-safe; the primary context may be
// current on several threads, module and function handles belong to it, and
// every call makes it current first (`bind`).
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
        if !self.module.is_null() && self.bind().is_ok() {
            // SAFETY: the module was loaded by `Runtime::open` and its
            // functions are not used after this.
            unsafe { (self.api().cuModuleUnload)(self.module) };
        }
    }
}

impl Device {
    /// Properties of the device and the code in use.
    #[must_use]
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The runtime this device was opened with.
    #[must_use]
    pub fn runtime(&self) -> &Runtime {
        &self.context.runtime
    }

    /// Free and total device memory in bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the driver cannot report it.
    pub fn memory(&self) -> Result<(u64, u64)> {
        self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: valid out-pointers; the context is current.
        self.api().check("cuMemGetInfo_v2", unsafe {
            (self.api().cuMemGetInfo_v2)(&raw mut free, &raw mut total)
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
    /// [`Error::Call`] if a driver call fails; mismatches are reported in the
    /// [`SelfTestReport`], not as an error.
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
        // SAFETY: no preconditions; the context is current.
        self.api().check("cuCtxSynchronize", unsafe {
            (self.api().cuCtxSynchronize)()
        })
    }

    pub(crate) fn api(&self) -> &Api {
        self.context.api()
    }

    pub(crate) fn kernels(&self) -> &Kernels {
        self.kernels
            .as_ref()
            .expect("kernels are loaded before the device is returned")
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Makes this device's context current on the calling thread.
    pub(crate) fn bind(&self) -> Result<()> {
        self.context.bind()
    }

    /// Launches `function` on a 1-D grid on the null stream and waits for it.
    ///
    /// # Safety
    ///
    /// `args` must match the kernel's parameters in number, order and type,
    /// and every address passed must stay valid, and not be accessed by the
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
        // allowed; the context is current.
        api.check("cuLaunchKernel", unsafe {
            (api.cuLaunchKernel)(
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
