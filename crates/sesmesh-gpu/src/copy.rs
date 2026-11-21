// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Value types the back ends' buffers hold.

use std::fmt;

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
