// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streams (ordered work queues) and events (timing points).

use std::ffi::c_int;
use std::fmt;

use super::{Device, Runtime, bind, ffi::Handle};
use crate::Result;

/// A HIP stream: work queued on it runs in order.
pub struct Stream {
    runtime: Runtime,
    ordinal: c_int,
    handle: Handle,
}

// SAFETY: HIP streams may be used from any thread.
unsafe impl Send for Stream {}
// SAFETY: as above.
unsafe impl Sync for Stream {}

impl fmt::Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream")
            .field("device", &self.ordinal)
            .finish_non_exhaustive()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if bind(&self.runtime.api, self.ordinal).is_ok() {
            // SAFETY: created by `hipStreamCreate`, destroyed once.
            unsafe { (self.runtime.api.hipStreamDestroy)(self.handle) };
        }
    }
}

impl Stream {
    /// Creates a stream on `device`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if the runtime cannot create one.
    pub fn new(device: &Device) -> Result<Self> {
        device.bind()?;
        let api = device.api();
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        api.check("hipStreamCreate", unsafe {
            (api.hipStreamCreate)(&raw mut handle)
        })?;
        Ok(Self {
            runtime: device.runtime().clone(),
            ordinal: device.ordinal(),
            handle,
        })
    }

    /// Waits until all work queued on the stream has finished.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] with the error of any failed work.
    pub fn synchronize(&self) -> Result<()> {
        bind(&self.runtime.api, self.ordinal)?;
        let api = &self.runtime.api;
        // SAFETY: valid stream.
        api.check("hipStreamSynchronize", unsafe {
            (api.hipStreamSynchronize)(self.handle)
        })
    }
}

/// A HIP event: a point in a stream's work, for waiting and timing.
pub struct Event {
    runtime: Runtime,
    ordinal: c_int,
    handle: Handle,
}

// SAFETY: HIP events may be used from any thread.
unsafe impl Send for Event {}
// SAFETY: as above.
unsafe impl Sync for Event {}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Event")
            .field("device", &self.ordinal)
            .finish_non_exhaustive()
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if bind(&self.runtime.api, self.ordinal).is_ok() {
            // SAFETY: created by `hipEventCreate`, destroyed once.
            unsafe { (self.runtime.api.hipEventDestroy)(self.handle) };
        }
    }
}

impl Event {
    /// Creates an event on `device`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if the runtime cannot create one.
    pub fn new(device: &Device) -> Result<Self> {
        device.bind()?;
        let api = device.api();
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        api.check("hipEventCreate", unsafe {
            (api.hipEventCreate)(&raw mut handle)
        })?;
        Ok(Self {
            runtime: device.runtime().clone(),
            ordinal: device.ordinal(),
            handle,
        })
    }

    /// Records the event after the work queued so far on `stream`, or on the
    /// device's default (null) stream, where [`Device`] launches its kernels.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if the runtime rejects it.
    pub fn record(&self, stream: Option<&Stream>) -> Result<()> {
        bind(&self.runtime.api, self.ordinal)?;
        let api = &self.runtime.api;
        let s = stream.map_or(std::ptr::null_mut(), |s| s.handle);
        // SAFETY: valid event and stream (or the null stream).
        api.check("hipEventRecord", unsafe {
            (api.hipEventRecord)(self.handle, s)
        })
    }

    /// Waits until the recorded work has finished.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] with the error of any failed work.
    pub fn synchronize(&self) -> Result<()> {
        bind(&self.runtime.api, self.ordinal)?;
        let api = &self.runtime.api;
        // SAFETY: valid event.
        api.check("hipEventSynchronize", unsafe {
            (api.hipEventSynchronize)(self.handle)
        })
    }

    /// Milliseconds from `start` to this event; both must have been
    /// recorded and have completed.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if either is not recorded or not complete.
    pub fn elapsed_ms(&self, start: &Self) -> Result<f32> {
        bind(&self.runtime.api, self.ordinal)?;
        let api = &self.runtime.api;
        let mut ms = 0.0f32;
        // SAFETY: valid out-pointer and events.
        api.check("hipEventElapsedTime", unsafe {
            (api.hipEventElapsedTime)(&raw mut ms, start.handle, self.handle)
        })?;
        Ok(ms)
    }
}
