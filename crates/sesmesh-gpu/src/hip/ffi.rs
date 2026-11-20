// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HIP runtime functions sesmesh uses, resolved from `libamdhip64` at run
//! time (hand-written from `hip_runtime_api.h`, HIP 6 and 7).

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;

use crate::{Error, Result};

/// `hipError_t`.
pub(crate) type HipError = c_int;
/// Module, function, stream and event handles.
pub(crate) type Handle = *mut c_void;

pub(crate) const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub(crate) const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub(crate) const ERROR_NO_DEVICE: HipError = 100;
pub(crate) const ERROR_INSUFFICIENT_DRIVER: HipError = 35;

/// Layout of `hipDeviceProp_tR0600` (frozen by its versioned symbol
/// `hipGetDevicePropertiesR0600@hip_6.0`): byte offsets of the fields read.
pub(crate) mod props {
    pub(crate) const SIZE: usize = 1472;
    pub(crate) const NAME: usize = 0;
    pub(crate) const TOTAL_GLOBAL_MEM: usize = 288;
    pub(crate) const WARP_SIZE: usize = 308;
    pub(crate) const MULTIPROCESSOR_COUNT: usize = 388;
    pub(crate) const INTEGRATED: usize = 396;
    pub(crate) const GCN_ARCH_NAME: usize = 1160;
    pub(crate) const STRING_LEN: usize = 256;
}

macro_rules! hip_api {
    ($($name:ident($($arg:ident: $ty:ty),* $(,)?);)*) => {
        /// Resolved HIP entry points; valid while `_library` is loaded.
        #[allow(non_snake_case)]
        pub(crate) struct Api {
            $(pub(crate) $name: unsafe extern "C" fn($($ty),*) -> HipError,)*
            hipGetErrorString: unsafe extern "C" fn(HipError) -> *const c_char,
            /// Where the library was loaded from.
            pub(crate) path: String,
            _library: Library,
        }

        impl Api {
            /// Resolves every entry point of `library`.
            ///
            /// # Safety
            ///
            /// `library` must be a HIP runtime: its symbols of these names
            /// must have the signatures declared here.
            #[allow(non_snake_case)]
            unsafe fn resolve(library: Library, path: String) -> std::result::Result<Self, String> {
                $(
                    // SAFETY: the caller guarantees the symbol's signature;
                    // the function pointer is used only while `_library`
                    // (stored next to it) stays loaded.
                    let $name = *unsafe {
                        library.get::<unsafe extern "C" fn($($ty),*) -> HipError>(
                            concat!(stringify!($name), "\0").as_bytes(),
                        )
                    }
                    .map_err(|e| format!("{}: {e}", stringify!($name)))?;
                )*
                // SAFETY: as above.
                let error_string = *unsafe {
                    library.get::<unsafe extern "C" fn(HipError) -> *const c_char>(
                        b"hipGetErrorString\0",
                    )
                }
                .map_err(|e| format!("hipGetErrorString: {e}"))?;
                Ok(Self {
                    $($name,)*
                    hipGetErrorString: error_string,
                    path,
                    _library: library,
                })
            }
        }
    };
}

hip_api! {
    hipInit(flags: c_uint);
    hipGetDeviceCount(count: *mut c_int);
    hipSetDevice(device: c_int);
    hipGetDevicePropertiesR0600(props: *mut c_void, device: c_int);
    hipRuntimeGetVersion(version: *mut c_int);
    hipMemGetInfo(free: *mut usize, total: *mut usize);
    hipModuleLoadData(module: *mut Handle, image: *const c_void);
    hipModuleUnload(module: Handle);
    hipModuleGetFunction(function: *mut Handle, module: Handle, name: *const c_char);
    hipModuleLaunchKernel(
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
    hipMalloc(ptr: *mut *mut c_void, size: usize);
    hipFree(ptr: *mut c_void);
    hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint);
    hipHostFree(ptr: *mut c_void);
    hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int);
    hipMemset(dst: *mut c_void, value: c_int, size: usize);
    hipDeviceSynchronize();
    hipStreamCreate(stream: *mut Handle);
    hipStreamDestroy(stream: Handle);
    hipStreamSynchronize(stream: Handle);
    hipEventCreate(event: *mut Handle);
    hipEventDestroy(event: Handle);
    hipEventRecord(event: Handle, stream: Handle);
    hipEventSynchronize(event: Handle);
    hipEventElapsedTime(ms: *mut f32, start: Handle, stop: Handle);
}

impl Api {
    /// `Ok` for `hipSuccess`, else the runtime's message.
    pub(crate) fn check(&self, call: &'static str, code: HipError) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: `hipGetErrorString` returns a static NUL-terminated string
        // for any code (or null, handled below).
        let text = unsafe { (self.hipGetErrorString)(code) };
        let message = if text.is_null() {
            "unknown error".to_owned()
        } else {
            // SAFETY: non-null, NUL-terminated, static (see above).
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::Call {
            call,
            code,
            message,
        })
    }
}

/// Library names and paths tried by [`load`], most specific first.
pub(crate) fn candidates() -> Vec<PathBuf> {
    let mut list: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        list.extend(["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"].map(PathBuf::from));
    } else if cfg!(target_os = "linux") {
        list.extend(["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"].map(PathBuf::from));
        let roots = ["ROCM_PATH", "HIP_PATH"]
            .iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .chain([PathBuf::from("/opt/rocm")]);
        list.extend(roots.map(|r| r.join("lib").join("libamdhip64.so")));
    }
    list
}

/// Loads the first of `paths` that opens and resolves.
pub(crate) fn load<P: AsRef<Path>>(paths: &[P]) -> Result<Api> {
    let mut tried = Vec::new();
    for path in paths {
        let path = path.as_ref();
        let shown = path.display().to_string();
        // SAFETY: opening a library runs its initialisers; the HIP runtime's
        // are those every HIP program runs. Other files are rejected by the
        // symbol lookup in `resolve`.
        match unsafe { Library::new(path) } {
            Ok(library) => {
                // SAFETY: a library exporting these HIP names is the HIP
                // runtime.
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
            "nothing: no HIP runtime on this platform".to_owned()
        } else {
            tried.join(", ")
        },
    })
}
