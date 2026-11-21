// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// CUDA build of the shared kernels (build.rs, feature `cuda`). nvcc includes
// the CUDA runtime headers itself.

#include <math.h>

#include "../sesmesh.cuh"
