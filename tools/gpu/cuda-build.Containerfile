# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Build environment for the CUDA back end (feature `cuda`) on a machine
# without an NVIDIA GPU: Rust stable plus CUDA's nvcc (docs/06-gpu-cuda.md).
# Only nvcc and the runtime headers are installed; the binaries it builds
# load the driver's libcuda when they run.
#
#   podman build -t localhost/sesmesh-cuda-build -f tools/gpu/cuda-build.Containerfile .
FROM docker.io/library/ubuntu:24.04

ARG CUDA=13-3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gcc g++ make \
      pkg-config python3 git xz-utils \
 && curl -fsSLo /tmp/keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/keyring.deb && rm /tmp/keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvcc-${CUDA} cuda-cudart-dev-${CUDA} \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH CUDA_PATH=/usr/local/cuda
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal \
      --default-toolchain stable --component clippy,rustfmt
