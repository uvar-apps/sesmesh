// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device buffers and pinned host buffers.

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

use super::ffi::DevicePtr;
use super::{Context, Device};
use crate::{DeviceCopy, Error, Result};

fn bytes<T>(len: usize) -> usize {
    len.checked_mul(size_of::<T>())
        .expect("buffer size overflows usize")
}

/// A buffer of `len` values in device memory (`cuMemAlloc`), zero-filled on
/// creation.
pub struct DeviceBuffer<T: DeviceCopy> {
    context: Arc<Context>,
    ptr: DevicePtr,
    len: usize,
    _values: PhantomData<T>,
}

// SAFETY: the buffer owns its device allocation; driver memory operations
// are thread-safe and `&self` methods only read it.
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
        if self.ptr != 0 && self.context.bind().is_ok() {
            // SAFETY: `ptr` came from `cuMemAlloc` in this context, which
            // `self.context` keeps alive, and is freed once.
            unsafe { (self.context.api().cuMemFree_v2)(self.ptr) };
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
        self.write_prefix(src)
    }

    /// Copies the buffer into `dst` (of the same length).
    ///
    /// # Errors
    ///
    /// As [`DeviceBuffer::copy_from`].
    pub fn copy_to(&self, dst: &mut [T]) -> Result<()> {
        self.check_len(dst.len())?;
        self.read_prefix(dst)
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

    /// Copies `src` into the first `src.len()` values.
    pub(crate) fn write_prefix(&mut self, src: &[T]) -> Result<()> {
        self.check_prefix(src.len())?;
        if src.is_empty() {
            return Ok(());
        }
        self.context.bind()?;
        let api = self.context.api();
        // SAFETY: the device range holds at least `src.len()` values and is
        // owned; the host range is a live slice.
        api.check("cuMemcpyHtoD_v2", unsafe {
            (api.cuMemcpyHtoD_v2)(self.ptr, src.as_ptr().cast(), bytes::<T>(src.len()))
        })
    }

    /// Copies the first `dst.len()` values into `dst`.
    pub(crate) fn read_prefix(&self, dst: &mut [T]) -> Result<()> {
        self.check_prefix(dst.len())?;
        if dst.is_empty() {
            return Ok(());
        }
        self.context.bind()?;
        let api = self.context.api();
        // SAFETY: both ranges hold `dst.len()` values; every bit pattern is
        // a valid `T` (`DeviceCopy` types are plain numbers).
        api.check("cuMemcpyDtoH_v2", unsafe {
            (api.cuMemcpyDtoH_v2)(dst.as_mut_ptr().cast(), self.ptr, bytes::<T>(dst.len()))
        })
    }

    /// Device address, for kernel arguments.
    pub(crate) fn address(&self) -> DevicePtr {
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

    fn check_prefix(&self, slice: usize) -> Result<()> {
        if slice <= self.len {
            Ok(())
        } else {
            Err(Error::Length {
                buffer: self.len,
                slice,
            })
        }
    }
}

/// A buffer of `len` values in pinned host memory (`cuMemHostAlloc`),
/// zero-filled on creation. The host reads and writes it as a slice; copies
/// between it and device buffers run at the bus's full speed, and kernels
/// can address it in place (unified addressing), over the bus.
pub struct HostBuffer<T: DeviceCopy> {
    context: Arc<Context>,
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
        if self.len != 0 && self.context.bind().is_ok() {
            // SAFETY: `ptr` came from `cuMemHostAlloc` in this context and is
            // freed once.
            unsafe { (self.context.api().cuMemFreeHost)(self.ptr.as_ptr().cast()) };
        }
    }
}

impl<T: DeviceCopy> Deref for HostBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` is aligned and holds `len` initialised values (or is
        // dangling with `len == 0`); no kernel or copy writes it while
        // borrowed: launches and copies that use it take `&mut` access to
        // their batch and wait for completion.
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
    /// Address kernels use (unified addressing: the host address).
    pub(crate) fn address(&mut self) -> u64 {
        self.ptr.as_ptr() as u64
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
        let mut buffer = DeviceBuffer {
            context: Arc::clone(self.context()),
            ptr: 0,
            len,
            _values: PhantomData,
        };
        if size != 0 {
            self.bind()?;
            let api = self.api();
            let mut ptr: DevicePtr = 0;
            // SAFETY: valid out-pointer; the context is current.
            api.check("cuMemAlloc_v2", unsafe {
                (api.cuMemAlloc_v2)(&raw mut ptr, size)
            })?;
            buffer.ptr = ptr;
            // SAFETY: `ptr` holds `size` bytes.
            api.check("cuMemsetD8_v2", unsafe {
                (api.cuMemsetD8_v2)(ptr, 0, size)
            })?;
        }
        Ok(buffer)
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
            // SAFETY: valid out-pointer; flags 0: page-locked, and with
            // unified addressing also mapped into the device's space.
            api.check("cuMemHostAlloc", unsafe {
                (api.cuMemHostAlloc)(&raw mut raw, size, 0)
            })?;
            ptr = NonNull::new(raw.cast::<T>()).ok_or(Error::Call {
                call: "cuMemHostAlloc",
                code: 0,
                message: "returned a null pointer".to_owned(),
            })?;
            // SAFETY: the allocation holds `size` bytes, page-aligned.
            unsafe { ptr.as_ptr().cast::<u8>().write_bytes(0, size) };
        }
        Ok(HostBuffer {
            context: Arc::clone(self.context()),
            ptr,
            len,
        })
    }
}
