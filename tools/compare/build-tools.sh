#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Builds the programs compiled from source, inside the sesmesh-compare
# container, into /work/opt/bin (the mounted bench directory):
#   sesmesh (this repository, release), meshcheck (tools/compare/meshcheck),
#   NanoShaper 0.7 (CGAL + Boost threads), ESES (OpenMP).
# Usage (on the host):
#   podman run --rm -v ~/sesmesh-bench:/work sesmesh-compare \
#       bash /work/src/sesmesh/tools/compare/build-tools.sh [sesmesh|meshcheck|nanoshaper|eses]...
set -euo pipefail

W=/work
BIN=$W/opt/bin
mkdir -p "$BIN" "$W/build"
JOBS=$(nproc)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(sesmesh meshcheck nanoshaper eses)

for t in "${targets[@]}"; do
  case "$t" in
    sesmesh)
      CARGO_TARGET_DIR=$W/build/sesmesh cargo build --release --locked \
        --manifest-path "$W/src/sesmesh/Cargo.toml" -p sesmesh-cli
      cp "$W/build/sesmesh/release/sesmesh" "$BIN/"
      ;;
    meshcheck)
      # rsync keeps source mtimes older than a previous build's artefacts.
      find "$W/src/sesmesh/tools/compare/meshcheck/src" -name '*.rs' -exec touch {} +
      CARGO_TARGET_DIR=$W/build/meshcheck cargo build --release \
        --manifest-path "$W/src/sesmesh/tools/compare/meshcheck/Cargo.toml"
      cp "$W/build/meshcheck/release/meshcheck" "$BIN/"
      ;;
    nanoshaper)
      # GCC 13 no longer pulls <memory> in transitively (std::shared_ptr).
      grep -q '#include <memory>' "$W/src/NanoShaper/src/Configuration.h" ||
        sed -i 's|#include <string>|#include <string>\n#include <memory>|' "$W/src/NanoShaper/src/Configuration.h"
      cmake -S "$W/src/NanoShaper" -B "$W/build/nanoshaper" \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_CGAL=ON -DENABLE_BOOST=ON -DENABLE_EXE=ON
      cmake --build "$W/build/nanoshaper" -j "$JOBS"
      cp "$(find "$W/build/nanoshaper" -maxdepth 2 -type f -name NanoShaper -perm -u+x | head -1)" "$BIN/NanoShaper"
      ;;
    eses)
      cmake -S "$W/src/ESES" -B "$W/build/eses" -DCMAKE_BUILD_TYPE=Release
      cmake --build "$W/build/eses" -j "$JOBS"
      cp "$(find "$W/build/eses" -maxdepth 2 -type f -name MS_Intersection -perm -u+x | head -1)" "$BIN/MS_Intersection"
      ;;
    *) echo "unknown target $t" >&2; exit 2 ;;
  esac
  echo "built $t"
done
ls -la "$BIN"
