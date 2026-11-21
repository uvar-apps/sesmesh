// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// HIP build of the shared kernels (build.rs, feature `hip`).

#include <hip/hip_runtime.h>

#include "../sesmesh.cuh"
