// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds the GPU code of the enabled back ends (docs/05-gpu-hip.md §9,
//! docs/06-gpu-cuda.md).
//!
//! * `hip`: `hipcc --genco` compiles `kernels/hip/sesmesh.hip` once per
//!   target of `SESMESH_HIP_ARCHS` (default: ROCm's generic targets for
//!   RDNA 2, 3 and 4), and `$OUT_DIR/hip_code_objects.rs` embeds the objects.
//! * `cuda`: `nvcc -fatbin` compiles `kernels/cuda/sesmesh.cu` into one fat
//!   binary with machine code for each architecture of `SESMESH_CUDA_ARCHS`
//!   (default: Turing to Blackwell, those this nvcc supports) and PTX for the
//!   lowest, which the driver compiles for newer GPUs;
//!   `$OUT_DIR/cuda_kernels.rs` embeds it.
//!
//! Both kernel files include `kernels/sesmesh.cuh`. Without the compiler the
//! build stops with an error, except under docs.rs or with
//! `SESMESH_GPU_NO_KERNELS=1` (code checks such as clippy or rustdoc on
//! machines without ROCm or CUDA): such builds embed no GPU code, and
//! opening a device reports it.

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
        "NVCC",
        "CUDA_PATH",
        "CUDA_HOME",
        "SESMESH_CUDA_ARCHS",
        "SESMESH_GPU_NO_KERNELS",
        "DOCS_RS",
    ] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let skip = env::var_os("DOCS_RS").is_some()
        || env::var("SESMESH_GPU_NO_KERNELS").is_ok_and(|v| !v.is_empty() && v != "0");
    if env::var_os("CARGO_FEATURE_HIP").is_some() {
        hip(&out, skip);
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        cuda(&out, skip);
    }
}

fn hip(out: &Path, skip: bool) {
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

/// CUDA architectures compiled to machine code by default (compute
/// capability × 10): Turing, Ampere, Ada, Hopper, Blackwell. Those the nvcc
/// in use does not list (`--list-gpu-arch`) are left out.
const CUDA_DEFAULT_ARCHS: &[u32] = &[75, 80, 86, 89, 90, 100, 120];

/// Flags of the exact CUDA kernels (docs/06-gpu-cuda.md): no contraction
/// into FMA, IEEE division and square root, subnormals kept.
const CUDA_FLAGS: &[&str] = &[
    "-fatbin",
    "-O3",
    "-fmad=false",
    "-prec-div=true",
    "-prec-sqrt=true",
    "-ftz=false",
];

fn cuda(out: &Path, skip: bool) {
    let mut fatbin = None;
    let mut archs = Vec::new();
    if !skip {
        let Some(nvcc) = find_nvcc() else {
            println!(
                "cargo::error=the `cuda` feature needs CUDA's nvcc to compile the GPU \
                 kernels, and none was found ($NVCC, $CUDA_PATH/bin, $CUDA_HOME/bin, \
                 /usr/local/cuda/bin, /opt/cuda/bin, PATH). Install the CUDA toolkit or point \
                 CUDA_PATH or NVCC at it. For code checks only (clippy, rustdoc) set \
                 SESMESH_GPU_NO_KERNELS=1: the build then has no kernels."
            );
            return;
        };
        archs = match cuda_archs(&nvcc) {
            Ok(a) => a,
            Err(e) => {
                println!("cargo::error={e}");
                return;
            }
        };
        let object = out.join("sesmesh.fatbin");
        let mut cmd = Command::new(&nvcc);
        cmd.args(CUDA_FLAGS);
        for a in &archs {
            cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        }
        let lowest = archs[0];
        cmd.arg(format!(
            "-gencode=arch=compute_{lowest},code=compute_{lowest}"
        ));
        cmd.arg("-o")
            .arg(&object)
            .arg(Path::new("kernels/cuda/sesmesh.cu"));
        match cmd.output() {
            Ok(o) if o.status.success() => fatbin = Some(object),
            Ok(o) => {
                let log = String::from_utf8_lossy(&o.stderr).replace('\n', " | ");
                println!("cargo::error=nvcc failed ({}): {log}", o.status);
                return;
            }
            Err(e) => {
                println!("cargo::error=cannot run {}: {e}", nvcc.display());
                return;
            }
        }
    }
    let mut code = String::from(HEADER);
    code.push_str(
        "/// Embedded CUDA fat binary (empty when built without kernels).\n\
         pub(crate) const FATBIN: &[u8] = ",
    );
    match &fatbin {
        Some(f) => writeln!(code, "include_bytes!({:?});", f.display().to_string()),
        None => writeln!(code, "&[];"),
    }
    .expect("write to String");
    code.push_str(
        "/// Architectures with machine code in [`FATBIN`] (compute capability × 10);\n\
         /// the first also has PTX.\n\
         pub(crate) const ARCHS: &[u32] = &[",
    );
    let list: Vec<String> = archs.iter().map(u32::to_string).collect();
    code.push_str(&list.join(", "));
    code.push_str("];\n");
    std::fs::write(out.join("cuda_kernels.rs"), code).expect("write cuda_kernels.rs");
}

/// Architectures from `SESMESH_CUDA_ARCHS` (comma or space separated, as
/// `75` or `sm_75`) or the default, intersected with what `nvcc` supports,
/// sorted ascending.
fn cuda_archs(nvcc: &Path) -> Result<Vec<u32>, String> {
    let listed = Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map_err(|e| format!("cannot run {}: {e}", nvcc.display()))?;
    let supported: Vec<u32> = String::from_utf8_lossy(&listed.stdout)
        .split_whitespace()
        .filter_map(|t| t.strip_prefix("compute_")?.parse().ok())
        .collect();
    let requested: Vec<u32> = match env::var("SESMESH_CUDA_ARCHS") {
        Ok(v) if !v.trim().is_empty() => {
            let mut list = Vec::new();
            for t in v.split([',', ' ']).filter(|t| !t.is_empty()) {
                let n = t
                    .trim_start_matches("sm_")
                    .trim_start_matches("compute_")
                    .parse()
                    .map_err(|_| format!("SESMESH_CUDA_ARCHS: bad architecture {t:?}"))?;
                if !supported.contains(&n) {
                    return Err(format!(
                        "SESMESH_CUDA_ARCHS: {} does not support sm_{n} (it lists {supported:?})",
                        nvcc.display()
                    ));
                }
                list.push(n);
            }
            list
        }
        _ => CUDA_DEFAULT_ARCHS
            .iter()
            .copied()
            .filter(|a| supported.contains(a))
            .collect(),
    };
    let mut archs = requested;
    archs.sort_unstable();
    archs.dedup();
    if archs.is_empty() {
        return Err(format!(
            "{} supports none of the default CUDA architectures {CUDA_DEFAULT_ARCHS:?} \
             (it lists {supported:?}); set SESMESH_CUDA_ARCHS",
            nvcc.display()
        ));
    }
    Ok(archs)
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

fn find_nvcc() -> Option<PathBuf> {
    if let Some(p) = env::var_os("NVCC") {
        return Some(PathBuf::from(p));
    }
    let mut dirs: Vec<PathBuf> = ["CUDA_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(env::var_os)
        .map(|root| PathBuf::from(root).join("bin"))
        .collect();
    dirs.push(PathBuf::from("/usr/local/cuda/bin"));
    dirs.push(PathBuf::from("/opt/cuda/bin"));
    if let Some(path) = env::var_os("PATH") {
        dirs.extend(env::split_paths(&path));
    }
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    dirs.into_iter().map(|d| d.join(exe)).find(|p| p.is_file())
}
