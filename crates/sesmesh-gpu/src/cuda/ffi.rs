// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CUDA driver API functions sesmesh uses, resolved from `libcuda` at
//! run time (hand-written from `cuda.h`; the `_v2` names are the current
//! ABI of every driver since CUDA 4).

use std::ffi::{CStr, c_char, c_int, c_uchar, c_uint, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;

use crate::{Error, Result};

/// `CUresult`.
pub(crate) type CuResult = c_int;
/// Context, module, function, stream and event handles.
pub(crate) type Handle = *mut c_void;
/// `CUdeviceptr`: a device address.
pub(crate) type DevicePtr = u64;
/// `CUdevice`: a device ordinal.
pub(crate) type CuDevice = c_int;

pub(crate) const ERROR_NO_DEVICE: CuResult = 100;
pub(crate) const ERROR_INSUFFICIENT_DRIVER: CuResult = 35;
pub(crate) const ERROR_NO_BINARY_FOR_GPU: CuResult = 209;
pub(crate) const ERROR_INVALID_PTX: CuResult = 218;
pub(crate) const ERROR_UNSUPPORTED_PTX_VERSION: CuResult = 222;

/// `CUdevice_attribute` values read.
pub(crate) mod attr {
    pub(crate) const WARP_SIZE: i32 = 10;
    pub(crate) const MULTIPROCESSOR_COUNT: i32 = 16;
    pub(crate) const INTEGRATED: i32 = 18;
    pub(crate) const UNIFIED_ADDRESSING: i32 = 41;
    pub(crate) const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub(crate) const COMPUTE_CAPABILITY_MINOR: i32 = 76;
}

macro_rules! cuda_api {
    ($($name:ident($($arg:ident: $ty:ty),* $(,)?);)*) => {
        /// Resolved driver entry points; valid while `_library` is loaded.
        #[allow(non_snake_case)]
        pub(crate) struct Api {
            $(pub(crate) $name: unsafe extern "C" fn($($ty),*) -> CuResult,)*
            /// Where the library was loaded from.
            pub(crate) path: String,
            _library: Library,
        }

        impl Api {
            /// Resolves every entry point of `library`.
            ///
            /// # Safety
            ///
            /// `library` must be the CUDA driver library: its symbols of these
            /// names must have the signatures declared here.
            #[allow(non_snake_case)]
            unsafe fn resolve(library: Library, path: String) -> std::result::Result<Self, String> {
                $(
                    // SAFETY: the caller guarantees the symbol's signature;
                    // the function pointer is used only while `_library`
                    // (stored next to it) stays loaded.
                    let $name = *unsafe {
                        library.get::<unsafe extern "C" fn($($ty),*) -> CuResult>(
                            concat!(stringify!($name), "\0").as_bytes(),
                        )
                    }
                    .map_err(|e| format!("{}: {e}", stringify!($name)))?;
                )*
                Ok(Self {
                    $($name,)*
                    path,
                    _library: library,
                })
            }
        }
    };
}

cuda_api! {
    cuInit(flags: c_uint);
    cuDriverGetVersion(version: *mut c_int);
    cuGetErrorString(error: CuResult, text: *mut *const c_char);
    cuDeviceGetCount(count: *mut c_int);
    cuDeviceGet(device: *mut CuDevice, ordinal: c_int);
    cuDeviceGetName(name: *mut c_char, len: c_int, device: CuDevice);
    cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: CuDevice);
    cuDeviceTotalMem_v2(bytes: *mut usize, device: CuDevice);
    cuDevicePrimaryCtxRetain(context: *mut Handle, device: CuDevice);
    cuDevicePrimaryCtxRelease_v2(device: CuDevice);
    cuCtxSetCurrent(context: Handle);
    cuCtxSynchronize();
    cuMemGetInfo_v2(free: *mut usize, total: *mut usize);
    cuModuleLoadData(module: *mut Handle, image: *const c_void);
    cuModuleUnload(module: Handle);
    cuModuleGetFunction(function: *mut Handle, module: Handle, name: *const c_char);
    cuLaunchKernel(
        function: Handle,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_bytes: c_uint,
        stream: Handle,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    );
    cuMemAlloc_v2(ptr: *mut DevicePtr, size: usize);
    cuMemFree_v2(ptr: DevicePtr);
    cuMemHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint);
    cuMemFreeHost(ptr: *mut c_void);
    cuMemcpyHtoD_v2(dst: DevicePtr, src: *const c_void, size: usize);
    cuMemcpyDtoH_v2(dst: *mut c_void, src: DevicePtr, size: usize);
    cuMemsetD8_v2(dst: DevicePtr, value: c_uchar, count: usize);
    cuStreamCreate(stream: *mut Handle, flags: c_uint);
    cuStreamDestroy_v2(stream: Handle);
    cuStreamSynchronize(stream: Handle);
    cuEventCreate(event: *mut Handle, flags: c_uint);
    cuEventDestroy_v2(event: Handle);
    cuEventRecord(event: Handle, stream: Handle);
    cuEventSynchronize(event: Handle);
    cuEventElapsedTime(ms: *mut f32, start: Handle, stop: Handle);
}

impl Api {
    /// `Ok` for `CUDA_SUCCESS`, else the driver's message.
    pub(crate) fn check(&self, call: &'static str, code: CuResult) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut text: *const c_char = std::ptr::null();
        // SAFETY: valid out-pointer; the driver stores a static string or
        // returns an error (text stays null).
        let known = unsafe { (self.cuGetErrorString)(code, &raw mut text) } == 0;
        let message = if known && !text.is_null() {
            // SAFETY: non-null, NUL-terminated, static (see above).
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        } else {
            "unknown error".to_owned()
        };
        Err(Error::Call {
            call,
            code,
            message,
        })
    }
}

/// Library names tried by [`load`], most specific first.
pub(crate) fn candidates() -> Vec<PathBuf> {
    if cfg!(windows) {
        vec![PathBuf::from("nvcuda.dll")]
    } else if cfg!(target_os = "linux") {
        ["libcuda.so.1", "libcuda.so"]
            .map(PathBuf::from)
            .into_iter()
            .collect()
    } else {
        Vec::new()
    }
}

/// Loads the first of `paths` that opens and resolves.
pub(crate) fn load<P: AsRef<Path>>(paths: &[P]) -> Result<Api> {
    let mut tried = Vec::new();
    for path in paths {
        let path = path.as_ref();
        let shown = path.display().to_string();
        // SAFETY: opening a library runs its initialisers; the CUDA driver's
        // are those every CUDA program runs. Other files are rejected by the
        // symbol lookup in `resolve`.
        match unsafe { Library::new(path) } {
            Ok(library) => {
                // SAFETY: a library exporting these CUDA driver names is the
                // CUDA driver.
                return unsafe { Api::resolve(library, shown.clone()) }.map_err(|reason| {
                    Error::BadRuntime {
                        library: shown,
                        reason,
                    }
                });
            }
            Err(_) => tried.push(shown),
        }
    }
    Err(Error::NoRuntime {
        tried: if tried.is_empty() {
            "no CUDA driver on this platform".to_owned()
        } else {
            format!("tried {}", tried.join(", "))
        },
    })
}
