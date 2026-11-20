// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compiles `spike.hip` to a code object with hipcc and links libamdhip64.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let arch = env::var("SPIKE_ARCH").unwrap_or_else(|_| "gfx1151".into());
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    for (src, obj, extra) in [
        ("spike.hip", "spike.hsaco", None),
        ("ses.hip", "ses.hsaco", None),
        ("ses.hip", "ses64.hsaco", Some("-mwavefrontsize64")),
    ] {
        let mut cmd = Command::new(format!("{rocm}/bin/hipcc"));
        cmd.args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "-ffp-contract=off",
            "-fno-fast-math",
        ]);
        cmd.args(extra);
        let status = cmd
            .arg("-o")
            .arg(dir.join(obj))
            .arg(src)
            .status()
            .expect("hipcc");
        assert!(status.success(), "hipcc failed on {src}");
    }
    println!("cargo:rerun-if-changed=ses.hip");
    println!("cargo:rerun-if-changed=spike.hip");
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
