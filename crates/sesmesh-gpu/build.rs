// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds the HIP code objects (feature `hip`, docs/05-gpu-hip.md §9).
//!
//! `hipcc --genco` compiles `kernels/hip/sesmesh.hip` once per target of
//! `SESMESH_HIP_ARCHS` (default: ROCm's generic targets for RDNA 2, 3 and 4),
//! and `$OUT_DIR/hip_code_objects.rs` embeds the objects. Without `hipcc` the
//! build stops with an error, except under docs.rs or with
//! `SESMESH_GPU_NO_KERNELS=1` (code checks such as clippy or rustdoc on
//! machines without ROCm): such builds embed no code object, and opening a
//! device reports it.

use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One code object per GPU family (ROCm generic targets, code object v6).
const DEFAULT_TARGETS: &[&str] = &["gfx10-3-generic", "gfx11-generic", "gfx12-generic"];

/// Flags of every exact kernel (docs/05-gpu-hip.md §3): no contraction into
/// FMA, no fast math; HIP's defaults keep `/` and `sqrt` correctly rounded and
/// denormals.
const FLAGS: &[&str] = &["--genco", "-O3", "-ffp-contract=off", "-fno-fast-math"];

const HEADER: &str = "// sesmesh — Solvent Excluded Surface meshes for molecules\n\
                      // Copyright (c) 2026 Ugo Varetto\n\
                      // SPDX-License-Identifier: MIT OR Apache-2.0\n\n";

fn main() {
    println!("cargo::rerun-if-changed=kernels");
    for var in [
        "HIPCC",
        "ROCM_PATH",
        "HIP_PATH",
        "SESMESH_HIP_ARCHS",
        "SESMESH_GPU_NO_KERNELS",
        "DOCS_RS",
    ] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    if env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let skip = env::var_os("DOCS_RS").is_some()
        || env::var("SESMESH_GPU_NO_KERNELS").is_ok_and(|v| !v.is_empty() && v != "0");
    let mut objects = Vec::new();
    if !skip {
        let Some(hipcc) = find_hipcc() else {
            println!(
                "cargo::error=the `hip` feature needs ROCm's hipcc to compile the GPU \
                 kernels, and none was found ($HIPCC, $ROCM_PATH/bin, $HIP_PATH/bin, /opt/rocm/bin, \
                 PATH). Install ROCm or point ROCM_PATH or HIPCC at it. For code checks only \
                 (clippy, rustdoc) set SESMESH_GPU_NO_KERNELS=1: the build then has no kernels."
            );
            return;
        };
        let source = Path::new("kernels/hip/sesmesh.hip");
        for target in targets() {
            let object = out.join(format!("sesmesh-{target}.hsaco"));
            let mut cmd = Command::new(&hipcc);
            cmd.args(FLAGS).arg(format!("--offload-arch={target}"));
            // RDNA runs wave32 by default; the kernels are written for any
            // wave size and wave64 was 1.3× faster (docs/05-gpu-hip.md §2.4).
            if target.starts_with("gfx1") {
                cmd.arg("-mwavefrontsize64");
            }
            cmd.arg("-o").arg(&object).arg(source);
            match cmd.output() {
                Ok(o) if o.status.success() => objects.push((target, object)),
                Ok(o) => {
                    let log = String::from_utf8_lossy(&o.stderr).replace('\n', " | ");
                    println!(
                        "cargo::error=hipcc failed for {target} ({}): {log}",
                        o.status
                    );
                    return;
                }
                Err(e) => {
                    println!("cargo::error=cannot run {}: {e}", hipcc.display());
                    return;
                }
            }
        }
    }
    let mut code = String::from(HEADER);
    code.push_str(
        "/// Embedded HIP code objects, `(offload target, object)`, in load order.\n\
         pub(crate) const CODE_OBJECTS: &[(&str, &[u8])] = &[\n",
    );
    for (target, object) in &objects {
        writeln!(
            code,
            "    ({target:?}, include_bytes!({:?})),",
            object.display().to_string()
        )
        .expect("write to String");
    }
    code.push_str("];\n");
    std::fs::write(out.join("hip_code_objects.rs"), code).expect("write hip_code_objects.rs");
}

/// Targets from `SESMESH_HIP_ARCHS` (comma or space separated) or the
/// default; specific targets load before generic ones.
fn targets() -> Vec<String> {
    let mut list: Vec<String> = match env::var("SESMESH_HIP_ARCHS") {
        Ok(v) if !v.trim().is_empty() => v
            .split([',', ' '])
            .filter(|t| !t.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => DEFAULT_TARGETS.iter().map(|&t| t.to_owned()).collect(),
    };
    list.sort_by_key(|t| t.ends_with("-generic"));
    list.dedup();
    list
}

fn find_hipcc() -> Option<PathBuf> {
    if let Some(p) = env::var_os("HIPCC") {
        return Some(PathBuf::from(p));
    }
    let mut dirs: Vec<PathBuf> = ["ROCM_PATH", "HIP_PATH"]
        .iter()
        .filter_map(env::var_os)
        .map(|root| PathBuf::from(root).join("bin"))
        .collect();
    dirs.push(PathBuf::from("/opt/rocm/bin"));
    if let Some(path) = env::var_os("PATH") {
        dirs.extend(env::split_paths(&path));
    }
    dirs.into_iter()
        .map(|d| d.join("hipcc"))
        .find(|p| p.is_file())
}
