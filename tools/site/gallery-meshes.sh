#!/bin/sh
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

# Builds the meshes behind site/img/gallery/*.webp into target/site-work/
# (needs `cargo build --release -p sesmesh-cli` and the structures from
# `cargo xtask fetch --set small --set bench`). Render them with
# site/render.html (see tools/site/README.md).
set -e
B=${SESMESH:-target/release/sesmesh}
C=${CACHE:-tests/data/cache}
W=target/site-work
mkdir -p "$W"
mesh() { name=$1; shift; "$B" "$@" -o "$W/$name.ply" --quiet; python3 tools/site/ply2web.py "$W/$name.ply" "$W/$name.sesm.gz"; }

mesh 1crn-element  data/1crn.cif -q high --vertex-colors element
mesh 1crn-patch    data/1crn.cif -q high --vertex-colors patch
mesh 1crn-vdw      data/1crn.cif -q high --surface vdw --vertex-colors element
mesh 1crn-sas      data/1crn.cif -q high --surface sas --vertex-colors element
mesh 1crn-coarse   data/1crn.cif -s 0.7 --vertex-colors element
mesh 1crn-decimated data/1crn.cif -q high --max-triangles 13996 --vertex-colors element
mesh 4hhb-chain    "$C/4HHB.cif.gz" -q high --vertex-colors chain --decimate 0.5
mesh 1aon-chain    "$C/1AON.cif.gz" -s 0.5 --vertex-colors chain --max-triangles 1500000
mesh 1aon-cavities "$C/1AON.cif.gz" -q high --cavities only --min-cavity-volume 5 --vertex-colors component
mesh 1k4r-capsid   "$C/1K4R.cif.gz" --assembly 1 -q draft --vertex-colors chain --max-triangles 3000000
