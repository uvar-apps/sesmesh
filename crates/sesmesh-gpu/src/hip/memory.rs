// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device buffers and pinned host buffers.

use std::ffi::{c_int, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use super::{Device, Runtime, bind, ffi};
use crate::{Error, Result};

mod sealed {
    #[allow(unreachable_pub)]
    pub trait Sealed {}
}

/// Plain values that can be copied to and from a device bit for bit: the
/// integer and floating-point types. Sealed.
pub trait DeviceCopy: Copy + Send + Sync + fmt::Debug + sealed::Sealed + 'static {
    /// The value with all bits zero (`0`, `+0.0`), which buffers start with.
    const ZERO: Self;
}

macro_rules! device_copy {
    ($($t:ty = $zero:literal),*) => {$(
        impl sealed::Sealed for $t {}
        impl DeviceCopy for $t {
            const ZERO: Self = $zero;
        }
    )*};
}
device_copy!(
    u8 = 0,
    u16 = 0,
    u32 = 0,
    u64 = 0,
    i32 = 0,
    i64 = 0,
    f32 = 0.0,
    f64 = 0.0
);

fn bytes<T>(len: usize) -> usize {
    len.checked_mul(size_of::<T>())
        .expect("buffer size overflows usize")
}

/// A buffer of `len` values in device memory (`hipMalloc`), zero-filled on
/// creation.
pub struct DeviceBuffer<T: DeviceCopy> {
    runtime: Runtime,
    ordinal: c_int,
    ptr: *mut c_void,
    len: usize,
    _values: PhantomData<T>,
}

// SAFETY: the buffer owns its device allocation; HIP memory operations are
// thread-safe and `&self` methods only read it.
unsafe impl<T: DeviceCopy> Send for DeviceBuffer<T> {}
// SAFETY: as above.
unsafe impl<T: DeviceCopy> Sync for DeviceBuffer<T> {}

impl<T: DeviceCopy> fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("type", &std::any::type_name::<T>())
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() && bind(&self.runtime.api, self.ordinal).is_ok() {
            // SAFETY: `ptr` came from `hipMalloc` and is freed once.
            unsafe { (self.runtime.api.hipFree)(self.ptr) };
        }
    }
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `src` (of the same length) into the buffer.
    ///
    /// # Errors
    ///
    /// [`Error::Length`] if the lengths differ, [`Error::Call`] if the copy
    /// fails.
    pub fn copy_from(&mut self, src: &[T]) -> Result<()> {
        self.check_len(src.len())?;
        if self.len == 0 {
            return Ok(());
        }
        bind(&self.runtime.api, self.ordinal)?;
        let api = &self.runtime.api;
        // SAFETY: both ranges hold `len` values; the device range is owned.
        api.check("hipMemcpy", unsafe {
            (api.hipMemcpy)(
                self.ptr,
                src.as_ptr().cast(),
                bytes::<T>(self.len),
                ffi::MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    /// Copies the buffer into `dst` (of the same length).
    ///
    /// # Errors
    ///
    /// As [`DeviceBuffer::copy_from`].
    pub fn copy_to(&self, dst: &mut [T]) -> Result<()> {
        self.check_len(dst.len())?;
        if self.len == 0 {
            return Ok(());
        }
        bind(&self.runtime.api, self.ordinal)?;
        let api = &self.runtime.api;
        // SAFETY: both ranges hold `len` values; every bit pattern is a valid
        // `T` (`DeviceCopy` types are plain numbers).
        api.check("hipMemcpy", unsafe {
            (api.hipMemcpy)(
                dst.as_mut_ptr().cast(),
                self.ptr,
                bytes::<T>(self.len),
                ffi::MEMCPY_DEVICE_TO_HOST,
            )
        })
    }

    /// The buffer's values.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the copy fails.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut v = vec![T::ZERO; self.len];
        self.copy_to(&mut v)?;
        Ok(v)
    }

    /// Device address, for kernel arguments.
    pub(crate) fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn check_len(&self, slice: usize) -> Result<()> {
        if slice == self.len {
            Ok(())
        } else {
            Err(Error::Length {
                buffer: self.len,
                slice,
            })
        }
    }
}

/// A buffer of `len` values in pinned host memory (`hipHostMalloc`),
/// zero-filled on creation. The host reads and writes it as a slice; kernels
/// read and write it in place over the bus, or directly on an APU (zero
/// copy).
pub struct HostBuffer<T: DeviceCopy> {
    runtime: Runtime,
    ordinal: c_int,
    ptr: NonNull<T>,
    len: usize,
}

// SAFETY: the buffer owns its pinned allocation like a `Box<[T]>`.
unsafe impl<T: DeviceCopy> Send for HostBuffer<T> {}
// SAFETY: as above; shared access only reads.
unsafe impl<T: DeviceCopy> Sync for HostBuffer<T> {}

impl<T: DeviceCopy> fmt::Debug for HostBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostBuffer")
            .field("type", &std::any::type_name::<T>())
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<T: DeviceCopy> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        if self.len != 0 && bind(&self.runtime.api, self.ordinal).is_ok() {
            // SAFETY: `ptr` came from `hipHostMalloc` and is freed once.
            unsafe { (self.runtime.api.hipHostFree)(self.ptr.as_ptr().cast()) };
        }
    }
}

impl<T: DeviceCopy> Deref for HostBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` is aligned and holds `len` initialised values (or is
        // dangling with `len == 0`); no kernel runs on it while borrowed,
        // since launches take the buffer by `&mut` and wait for completion.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: DeviceCopy> DerefMut for HostBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: DeviceCopy> HostBuffer<T> {
    /// Address kernels use (unified virtual addressing: the host address).
    pub(crate) fn as_device_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }
}

impl Device {
    /// Allocates a zero-filled device buffer of `len` values.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the allocation fails (out of device memory).
    ///
    /// # Panics
    ///
    /// If the size in bytes overflows `usize`.
    pub fn alloc<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        let size = bytes::<T>(len);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if size != 0 {
            self.bind()?;
            let api = self.api();
            // SAFETY: valid out-pointer.
            api.check("hipMalloc", unsafe { (api.hipMalloc)(&raw mut ptr, size) })?;
            let buffer = self.wrap(ptr, len);
            // SAFETY: `ptr` holds `size` bytes.
            api.check("hipMemset", unsafe { (api.hipMemset)(ptr, 0, size) })?;
            return Ok(buffer);
        }
        Ok(self.wrap(ptr, len))
    }

    /// Allocates a device buffer holding a copy of `values`.
    ///
    /// # Errors
    ///
    /// As [`Device::alloc`].
    ///
    /// # Panics
    ///
    /// As [`Device::alloc`].
    pub fn upload<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut buffer = self.alloc(values.len())?;
        buffer.copy_from(values)?;
        Ok(buffer)
    }

    /// Allocates a zero-filled pinned host buffer of `len` values.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the allocation fails.
    ///
    /// # Panics
    ///
    /// As [`Device::alloc`].
    pub fn alloc_host<T: DeviceCopy>(&self, len: usize) -> Result<HostBuffer<T>> {
        let size = bytes::<T>(len);
        let mut ptr = NonNull::<T>::dangling();
        if size != 0 {
            self.bind()?;
            let api = self.api();
            let mut raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid out-pointer; flags 0 = `hipHostMallocDefault`.
            api.check("hipHostMalloc", unsafe {
                (api.hipHostMalloc)(&raw mut raw, size, 0)
            })?;
            ptr = NonNull::new(raw.cast::<T>()).ok_or(Error::Call {
                call: "hipHostMalloc",
                code: 0,
                message: "returned a null pointer".to_owned(),
            })?;
            // SAFETY: the allocation holds `size` bytes, page-aligned.
            unsafe { ptr.as_ptr().cast::<u8>().write_bytes(0, size) };
        }
        Ok(HostBuffer {
            runtime: self.runtime.clone(),
            ordinal: self.ordinal(),
            ptr,
            len,
        })
    }

    fn wrap<T: DeviceCopy>(&self, ptr: *mut c_void, len: usize) -> DeviceBuffer<T> {
        DeviceBuffer {
            runtime: self.runtime.clone(),
            ordinal: self.ordinal(),
            ptr,
            len,
            _values: PhantomData,
        }
    }
}
