// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streams (ordered work queues) and events (timing points).

use std::fmt;
use std::sync::Arc;

use super::{Context, Device, ffi::Handle};
use crate::Result;

/// A CUDA stream: work queued on it runs in order.
pub struct Stream {
    context: Arc<Context>,
    handle: Handle,
}

// SAFETY: CUDA streams may be used from any thread.
unsafe impl Send for Stream {}
// SAFETY: as above.
unsafe impl Sync for Stream {}

impl fmt::Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream").finish_non_exhaustive()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.context.bind().is_ok() {
            // SAFETY: created by `cuStreamCreate` in this context, destroyed
            // once.
            unsafe { (self.context.api().cuStreamDestroy_v2)(self.handle) };
        }
    }
}

impl Stream {
    /// Creates a stream on `device`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if the driver cannot create one.
    pub fn new(device: &Device) -> Result<Self> {
        device.bind()?;
        let api = device.api();
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer; flags 0 = `CU_STREAM_DEFAULT`.
        api.check("cuStreamCreate", unsafe {
            (api.cuStreamCreate)(&raw mut handle, 0)
        })?;
        Ok(Self {
            context: Arc::clone(device.context()),
            handle,
        })
    }

    /// Waits until all work queued on the stream has finished.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] with the error of any failed work.
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind()?;
        let api = self.context.api();
        // SAFETY: valid stream.
        api.check("cuStreamSynchronize", unsafe {
            (api.cuStreamSynchronize)(self.handle)
        })
    }
}

/// A CUDA event: a point in a stream's work, for waiting and timing.
pub struct Event {
    context: Arc<Context>,
    handle: Handle,
}

// SAFETY: CUDA events may be used from any thread.
unsafe impl Send for Event {}
// SAFETY: as above.
unsafe impl Sync for Event {}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Event").finish_non_exhaustive()
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.context.bind().is_ok() {
            // SAFETY: created by `cuEventCreate` in this context, destroyed
            // once.
            unsafe { (self.context.api().cuEventDestroy_v2)(self.handle) };
        }
    }
}

impl Event {
    /// Creates an event on `device`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if the driver cannot create one.
    pub fn new(device: &Device) -> Result<Self> {
        device.bind()?;
        let api = device.api();
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer; flags 0 = `CU_EVENT_DEFAULT` (timed).
        api.check("cuEventCreate", unsafe {
            (api.cuEventCreate)(&raw mut handle, 0)
        })?;
        Ok(Self {
            context: Arc::clone(device.context()),
            handle,
        })
    }

    /// Records the event after the work queued so far on `stream`, or on the
    /// null stream, where [`Device`] launches its kernels.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if the driver rejects it.
    pub fn record(&self, stream: Option<&Stream>) -> Result<()> {
        self.context.bind()?;
        let api = self.context.api();
        let s = stream.map_or(std::ptr::null_mut(), |s| s.handle);
        // SAFETY: valid event and stream (or the null stream).
        api.check("cuEventRecord", unsafe {
            (api.cuEventRecord)(self.handle, s)
        })
    }

    /// Waits until the recorded work has finished.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] with the error of any failed work.
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind()?;
        let api = self.context.api();
        // SAFETY: valid event.
        api.check("cuEventSynchronize", unsafe {
            (api.cuEventSynchronize)(self.handle)
        })
    }

    /// Milliseconds from `start` to this event; both must have been
    /// recorded and have completed.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if either is not recorded or not complete.
    pub fn elapsed_ms(&self, start: &Self) -> Result<f32> {
        self.context.bind()?;
        let api = self.context.api();
        let mut ms = 0.0f32;
        // SAFETY: valid out-pointer and events.
        api.check("cuEventElapsedTime", unsafe {
            (api.cuEventElapsedTime)(&raw mut ms, start.handle, self.handle)
        })?;
        Ok(ms)
    }
}
