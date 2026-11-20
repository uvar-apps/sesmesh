// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end tests of the `sesmesh` binary (Plan T4.8).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use sesmesh_testkit::obj;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_sesmesh"))
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/fixtures")
        .join(name)
}

fn tmp(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cli");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn run(args: &[&str]) -> Output {
    bin().args(args).output().unwrap()
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

/// Closed, consistently oriented mesh: every directed edge appears once and
/// its reverse once.
fn assert_closed(m: &obj::ObjMesh) {
    let mut edges: Vec<(u32, u32)> = m
        .triangles
        .iter()
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .collect();
    edges.sort_unstable();
    for w in edges.windows(2) {
        assert_ne!(w[0], w[1], "repeated directed edge");
    }
    for &(a, b) in &edges {
        assert!(edges.binary_search(&(b, a)).is_ok(), "open edge {a}-{b}");
    }
}

#[test]
fn surface_writes_a_closed_obj() {
    let out = tmp("1crn.obj");
    let o = run(&[
        "surface",
        fixture("1CRN.cif.gz").to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "-s",
        "0.5",
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("327 atoms"), "{stderr}");
    let text = std::fs::read_to_string(&out).unwrap();
    let m = obj::parse(&text).unwrap();
    assert!(m.comments[0].starts_with("sesmesh "));
    assert!(
        m.comments
            .iter()
            .any(|c| c.contains("grid_spacing: 0.5000"))
    );
    assert!(m.triangles.len() > 20_000);
    assert_eq!(m.normals.len(), m.positions.len());
    for n in &m.normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-3);
    }
    assert_closed(&m);
}

#[test]
fn default_subcommand_stdout_and_options() {
    let o = run(&[
        fixture("1L2Y.pdb.gz").to_str().unwrap(),
        "-q",
        "draft",
        "-o",
        "-",
        "--no-normals",
        "--precision",
        "2",
        "--quiet",
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert!(
        o.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let text = String::from_utf8(o.stdout).unwrap();
    let m = obj::parse(&text).unwrap();
    assert!(m.normals.is_empty() && !m.triangles.is_empty());
    let v = text.lines().find(|l| l.starts_with("v ")).unwrap();
    assert!(
        v.split_whitespace()
            .skip(1)
            .all(|c| c.split('.').nth(1).unwrap().len() == 2)
    );
    assert_closed(&m);
}

#[test]
fn output_is_identical_across_thread_counts() {
    let mut outputs = Vec::new();
    for j in ["1", "3", "8"] {
        let out = tmp(&format!("ubq-j{j}.obj"));
        let o = run(&[
            fixture("1UBQ.cif.gz").to_str().unwrap(),
            "-s",
            "0.6",
            "-j",
            j,
            "-o",
            out.to_str().unwrap(),
        ]);
        assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
        outputs.push(std::fs::read(&out).unwrap());
    }
    assert!(outputs.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn progress_lines_without_terminal() {
    let o = run(&[
        fixture("1L2Y.cif.gz").to_str().unwrap(),
        "-o",
        "-",
        "--progress",
    ]);
    assert_eq!(code(&o), 0);
    let stderr = String::from_utf8(o.stderr).unwrap();
    for stage in ["geometry", "classify", "mesh", "assemble", "measure"] {
        assert!(
            stderr.contains(&format!("sesmesh: {stage} done")),
            "{stderr}"
        );
    }
}

#[test]
fn surface_kinds_and_radii_sets() {
    for args in [
        &["--surface", "sas"][..],
        &["--surface", "vdw"],
        &["--radii", "protor"],
        &["--radii", "uniform:1.6", "--probe", "1.2"],
        &["--refine", "linear", "--altloc", "first"],
    ] {
        let mut all = vec![fixture("1L2Y.cif.gz").to_str().unwrap().to_string()];
        all.extend(["-q", "low", "-o", "-", "--quiet"].map(String::from));
        all.extend(args.iter().map(ToString::to_string));
        let o = bin().args(&all).output().unwrap();
        assert_eq!(
            code(&o),
            0,
            "{args:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let m = obj::parse(&String::from_utf8(o.stdout).unwrap()).unwrap();
        assert_closed(&m);
    }
}

#[test]
fn info_and_radii_commands() {
    let o = run(&["info", fixture("1UBQ.pdb.gz").to_str().unwrap()]);
    assert_eq!(code(&o), 0);
    let s = String::from_utf8(o.stdout).unwrap();
    for want in [
        "format: pdb",
        "model 1: 660 atoms",
        "selected: 602 atoms",
        "bounding box",
        "radii (bondi)",
    ] {
        assert!(s.contains(want), "{want:?} missing in\n{s}");
    }
    let o = run(&["radii"]);
    assert_eq!(code(&o), 0);
    let s = String::from_utf8(o.stdout).unwrap();
    assert!(s.lines().any(|l| l == "6\tC\t1.70"), "{s}");
    let o = run(&["radii", "--set", "protor"]);
    assert!(
        String::from_utf8(o.stdout)
            .unwrap()
            .contains("ALA\tCB\tC4H3\t1.88")
    );
}

#[test]
fn exit_codes() {
    let crn = fixture("1CRN.pdb.gz");
    let crn = crn.to_str().unwrap();
    // Usage errors: 2.
    assert_eq!(code(&run(&[crn, "--bogus"])), 2);
    assert_eq!(code(&run(&[crn, "-s", "0", "-o", "-"])), 2);
    assert_eq!(code(&run(&[crn, "--radii", "nonsense", "-o", "-"])), 2);
    assert_eq!(code(&run(&[crn, "--altloc", "xy", "-o", "-"])), 2);
    assert_eq!(code(&run(&[crn, "--precision", "12", "-o", "-"])), 2);
    // Runtime errors: 1.
    assert_eq!(code(&run(&["does-not-exist.pdb"])), 1);
    assert_eq!(code(&run(&[crn, "--chains", "Z", "-o", "-"])), 1);
    // Parse errors: 3.
    let bad = tmp("bad.cif");
    std::fs::write(&bad, "data_x\nloop_\n_atom_site.Cartn_x\n").unwrap();
    assert_eq!(code(&run(&[bad.to_str().unwrap(), "-o", "-"])), 3);
    // Help and version: 0.
    assert_eq!(code(&run(&["--help"])), 0);
    assert_eq!(code(&run(&["--version"])), 0);
    assert_eq!(code(&run(&["surface", "--help"])), 0);
}

#[test]
fn cavity_modes_and_groups() {
    let crn = fixture("1CRN.cif.gz");
    let crn = crn.to_str().unwrap();
    let count = |args: &[&str]| -> (usize, usize) {
        let mut all = vec![crn, "-s", "0.5", "-o", "-", "--quiet", "--group-components"];
        all.extend_from_slice(args);
        let o = run(&all);
        assert_eq!(
            code(&o),
            0,
            "{args:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let text = String::from_utf8(o.stdout).unwrap();
        let groups = text.lines().filter(|l| l.starts_with("g ")).count();
        let cavities = text.lines().filter(|l| l.starts_with("g cavity_")).count();
        assert_closed(&obj::parse(&text).unwrap());
        (groups, cavities)
    };
    // Crambin has one internal cavity.
    assert_eq!(count(&[]), (2, 1));
    assert_eq!(count(&["--cavities", "remove"]), (1, 0));
    assert_eq!(count(&["--cavities", "only"]), (1, 1));
    assert_eq!(count(&["--min-cavity-volume", "1e6"]), (1, 0));
}

/// Three atoms spanned by an SES sheet thinner than the grid (Design §9.1):
/// one sub-grid fragment at 0.25 Å, kept unless `--drop-fragments` (the
/// default header stays as in 1.0).
#[test]
fn sub_grid_fragments() {
    let xyz = tmp("thin-sheet.xyz");
    std::fs::write(
        &xyz,
        "3\nthin sheet\nO -2.171 7.406 15.697\nO -2.245 11.599 13.186\nC -4.576 11.056 16.662\n",
    )
    .unwrap();
    let stats = tmp("thin-sheet.json");
    let run_with = |extra: &[&str]| {
        let mut args = vec![
            xyz.to_str().unwrap(),
            "-s",
            "0.25",
            "-o",
            "-",
            "--group-components",
            "--stats",
            stats.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let o = run(&args);
        assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
        let text = String::from_utf8(o.stdout).unwrap();
        assert_closed(&obj::parse(&text).unwrap());
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&stats).unwrap()).unwrap();
        let groups = text.lines().filter(|l| l.starts_with("g ")).count();
        (groups, json, String::from_utf8(o.stderr).unwrap(), text)
    };
    let (groups, json, err, text) = run_with(&[]);
    assert_eq!(groups, 2);
    assert_eq!(json["mesh"]["removed_fragments"], 0);
    assert_eq!(json["parameters"]["drop_fragments"], false);
    assert_eq!(json["components"][1]["triangles"], 8);
    assert!(!err.contains("fragment"), "{err}");
    assert!(!text.contains("fragment"));
    assert!(text.contains("min_component_area: 0  normals: analytic\n"));
    assert!(text.contains("removed: 0  euler: "));
    let (groups, json, err, text) = run_with(&["--drop-fragments"]);
    assert_eq!(groups, 1);
    assert_eq!(json["mesh"]["removed_fragments"], 1);
    assert_eq!(json["mesh"]["removed_components"], 0);
    assert_eq!(json["parameters"]["drop_fragments"], true);
    assert!(err.contains("dropped 1 sub-grid fragment(s)"), "{err}");
    assert!(text.contains("drop_fragments: true  normals: analytic\n"));
    assert!(text.contains("fragments_removed: 1  euler: "));
    let (_, _, err, _) = run_with(&["--drop-fragments", "--quiet"]);
    assert!(err.is_empty(), "{err}");
}

#[test]
fn colours_csv_and_stats() {
    let crn = fixture("1CRN.cif.gz");
    let (stats, atoms, verts, palette) = (
        tmp("crn-stats.json"),
        tmp("crn-atoms.csv"),
        tmp("crn-verts.csv"),
        tmp("palette.json"),
    );
    std::fs::write(
        &palette,
        r##"{"format":"eigenvista-atom-colors","version":1,"colors":[{"symbol":"C","color":"#ff0000"}]}"##,
    )
    .unwrap();
    let o = run(&[
        crn.to_str().unwrap(),
        "-s",
        "0.5",
        "-o",
        "-",
        "--quiet",
        "--vertex-colors",
        "element",
        "--palette",
        palette.to_str().unwrap(),
        "--stats",
        stats.to_str().unwrap(),
        "--atom-areas",
        atoms.to_str().unwrap(),
        "--vertex-data",
        verts.to_str().unwrap(),
        "--face-normals",
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let text = String::from_utf8(o.stdout).unwrap();
    let m = obj::parse(&text).unwrap();
    let v: Vec<&str> = text.lines().filter(|l| l.starts_with("v ")).collect();
    assert!(v.iter().all(|l| l.split_whitespace().count() == 7));
    assert!(
        v.iter().any(|l| l.ends_with(" 1.0000 0.0000 0.0000")),
        "red carbon"
    );
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&stats).unwrap()).unwrap();
    assert_eq!(json["parameters"]["cavities"], "keep");
    assert_eq!(json["parameters"]["normals"], "face-average");
    assert_eq!(json["mesh"]["components"], 2);
    assert_eq!(json["components"].as_array().unwrap().len(), 2);
    assert!(json["timings_ms"]["total"].as_f64().unwrap() > 0.0);
    let area = json["mesh"]["area"].as_f64().unwrap();
    let atom_csv = std::fs::read_to_string(&atoms).unwrap();
    assert_eq!(atom_csv.lines().count(), 1 + 327);
    let sum: f64 = atom_csv
        .lines()
        .skip(1)
        .map(|l| l.rsplit(',').next().unwrap().parse::<f64>().unwrap())
        .sum();
    assert!((sum - area).abs() < 0.05, "{sum} vs {area}");
    let vert_csv = std::fs::read_to_string(&verts).unwrap();
    assert_eq!(vert_csv.lines().count(), 1 + m.positions.len());
    assert!(vert_csv.lines().nth(1).unwrap().split(',').count() == 14);
}

#[test]
fn phase7_usage_and_input_errors() {
    let crn = fixture("1CRN.pdb.gz");
    let crn = crn.to_str().unwrap();
    assert_eq!(code(&run(&[crn, "-o", "-", "--stats", "-"])), 2);
    assert_eq!(
        code(&run(&[crn, "-o", "-", "--no-normals", "--face-normals"])),
        2
    );
    assert_eq!(
        code(&run(&[crn, "-o", "-", "--palette", "x.json"])),
        2,
        "needs --vertex-colors"
    );
    assert_eq!(
        code(&run(&[crn, "-o", "-", "--min-component-area", "-1"])),
        2
    );
    let bad = tmp("bad-palette.json");
    std::fs::write(&bad, r##"{"colors":[{"symbol":"Xx","color":"#000000"}]}"##).unwrap();
    let o = run(&[
        crn,
        "-o",
        "-",
        "--vertex-colors",
        "element",
        "--palette",
        bad.to_str().unwrap(),
    ]);
    assert_eq!(code(&o), 3, "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn batch_mode_keep_going_and_formats() {
    let dir = tmp("batch");
    let _ = std::fs::remove_dir_all(&dir);
    let good = [fixture("1CRN.cif.gz"), fixture("1L2Y.pdb.gz")];
    let bad = tmp("broken.pdb");
    std::fs::write(&bad, "ATOM  garbage\n").unwrap();
    let args = |keep: bool| {
        let mut v: Vec<String> = vec![
            good[0].to_str().unwrap().into(),
            bad.to_str().unwrap().into(),
            good[1].to_str().unwrap().into(),
            "--output-dir".into(),
            dir.to_str().unwrap().into(),
            "--output-format".into(),
            "ply".into(),
            "-q".into(),
            "draft".into(),
            "--stats".into(),
            dir.to_str().unwrap().into(),
        ];
        if keep {
            v.push("--keep-going".into());
        }
        v
    };
    // Without --keep-going the batch stops at the broken input.
    let o = bin().args(args(false)).output().unwrap();
    assert_ne!(code(&o), 0);
    assert!(dir.join("1CRN.ply").exists() && !dir.join("1L2Y.ply").exists());
    let o = bin().args(args(true)).output().unwrap();
    assert_ne!(code(&o), 0);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("1 of 3 inputs failed"), "{stderr}");
    for stem in ["1CRN", "1L2Y"] {
        let ply = std::fs::read(dir.join(format!("{stem}.ply"))).unwrap();
        let m = sesmesh_testkit::ply::parse(&ply).unwrap();
        assert!(m.faces.len() > 1000 && m.faces.iter().all(|f| f.len() == 3));
        let stats: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("{stem}.stats.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(
            stats["mesh"]["triangles"].as_u64().unwrap() as usize,
            m.faces.len()
        );
    }
    // -o with several inputs is a usage error.
    let o = run(&[
        good[0].to_str().unwrap(),
        good[1].to_str().unwrap(),
        "-o",
        "x.obj",
    ]);
    assert_eq!(code(&o), 2);
}

#[test]
fn missing_output_directories_are_created() {
    let root = tmp("new-dirs");
    let _ = std::fs::remove_dir_all(&root);
    let inputs = [fixture("1CRN.cif.gz"), fixture("1L2Y.pdb.gz")];
    let sub = |name: &str| root.join("batch").join(name);
    let (meshes, stats, areas, verts) = (
        sub("out/meshes"),
        sub("stats"),
        sub("side/areas"),
        sub("verts"),
    );
    let o = run(&[
        inputs[0].to_str().unwrap(),
        inputs[1].to_str().unwrap(),
        "--output-dir",
        meshes.to_str().unwrap(),
        "--output-format",
        "ply",
        "-q",
        "draft",
        "--stats",
        stats.to_str().unwrap(),
        "--atom-areas",
        areas.to_str().unwrap(),
        "--vertex-data",
        verts.to_str().unwrap(),
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    for stem in ["1CRN", "1L2Y"] {
        for path in [
            meshes.join(format!("{stem}.ply")),
            stats.join(format!("{stem}.stats.json")),
            areas.join(format!("{stem}.atoms.csv")),
            verts.join(format!("{stem}.vertices.csv")),
        ] {
            assert!(path.is_file(), "{} missing", path.display());
        }
    }

    // One input: the parents of -o and of the side files are created.
    let single = root.join("single");
    let output = single.join("new/dir/x.obj");
    let stats = single.join("stats/x.json");
    let areas = single.join("areas/deeper/x.csv");
    let o = run(&[
        inputs[0].to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-q",
        "draft",
        "--stats",
        stats.to_str().unwrap(),
        "--atom-areas",
        areas.to_str().unwrap(),
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let m = obj::parse(&std::fs::read_to_string(&output).unwrap()).unwrap();
    assert_closed(&m);
    assert!(stats.is_file() && areas.is_file());

    // A directory that cannot be created is a runtime error (exit 1).
    let blocker = root.join("file");
    std::fs::write(&blocker, "").unwrap();
    let o = run(&[
        inputs[0].to_str().unwrap(),
        "-o",
        blocker.join("x.obj").to_str().unwrap(),
        "-q",
        "draft",
    ]);
    assert_eq!(code(&o), 1, "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn msms_density_smoothing_memory() {
    let crn = fixture("1CRN.cif.gz");
    let crn = crn.to_str().unwrap();
    let stem = tmp("crn-msms");
    let vert = stem.with_extension("vert");
    let o = run(&[
        crn,
        "-o",
        vert.to_str().unwrap(),
        "--density",
        "4",
        "--smooth",
        "2",
        "--quiet",
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let v = std::fs::read_to_string(&vert).unwrap();
    let f = std::fs::read_to_string(stem.with_extension("face")).unwrap();
    let nv: usize = v
        .lines()
        .nth(2)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let nf: usize = f
        .lines()
        .nth(2)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(v.lines().count(), 3 + nv);
    assert_eq!(f.lines().count(), 3 + nf);
    let density: f64 = v
        .lines()
        .nth(2)
        .unwrap()
        .split_whitespace()
        .nth(2)
        .unwrap()
        .parse()
        .unwrap();
    assert!((density / 4.0 - 1.0).abs() < 0.25, "{density}");
    // Memory guard: exit code 4.
    assert_eq!(
        code(&run(&[crn, "-o", "-", "--max-memory", "10K", "--quiet"])),
        4
    );
    assert_eq!(code(&run(&[crn, "-o", "-", "--max-memory", "lots"])), 2);
    assert_eq!(
        code(&run(&[crn, "-o", "-", "-s", "0.5", "--density", "3"])),
        2
    );
    assert_eq!(code(&run(&[crn, "--output-format", "msms", "-o", "-"])), 2);
}

#[test]
fn assemblies() {
    let crn = fixture("1CRN.cif.gz");
    let o = run(&["info", crn.to_str().unwrap()]);
    assert!(
        String::from_utf8(o.stdout)
            .unwrap()
            .contains("assemblies: 1 (1 copy)")
    );
    let o = run(&[
        crn.to_str().unwrap(),
        "--assembly",
        "1",
        "-o",
        "-",
        "-q",
        "draft",
        "--quiet",
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert_closed(&obj::parse(&String::from_utf8(o.stdout).unwrap()).unwrap());
    assert_eq!(
        code(&run(&[
            crn.to_str().unwrap(),
            "--assembly",
            "42",
            "-o",
            "-"
        ])),
        1
    );
}

/// A reader that closes standard output early (`sesmesh … -o - | head`)
/// stops the CLI quietly with success (fixed in 0.9.1).
#[test]
fn closed_stdout_is_not_an_error() {
    use std::process::Stdio;
    let crn = fixture("1CRN.cif.gz");
    for format in ["obj", "ply"] {
        let mut child = bin()
            .args([
                crn.to_str().unwrap(),
                "-q",
                "draft",
                "-o",
                "-",
                "--output-format",
                format,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        // Close the read end before the CLI writes its (> 64 KiB) output.
        drop(child.stdout.take());
        let out = child.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{format}: {:?} {stderr}", out.status);
        assert!(!stderr.contains("error"), "{format}: {stderr}");
    }
}

/// Ctrl-C (SIGINT) cancels a running computation: exit code 130 (#12).
#[cfg(unix)]
#[test]
fn sigint_cancels() {
    use std::process::Stdio;
    let crn = fixture("1CRN.cif.gz");
    // A fine grid keeps the computation busy for a while.
    let child = bin()
        .args([crn.to_str().unwrap(), "-s", "0.04", "-o", "-"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(400));
    let killed = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(130), "{stderr}");
    assert!(stderr.contains("interrupted"), "{stderr}");
}

/// `--refine fast` (Plan T11.5′): on the CPU the exact mesh (only the header
/// names the mode); `engine.refine_cpu_edges` only in fast mode. On a GPU,
/// the exact mode's faces.
#[test]
fn refine_fast() {
    let ubq = fixture("1UBQ.cif.gz");
    let mesh = |name: &str, extra: &[&str]| {
        let out = tmp(name);
        let stats = tmp(&format!("{name}.json"));
        let mut args = vec![ubq.to_str().unwrap(), "-s", "0.6", "--quiet", "-o"];
        args.push(out.to_str().unwrap());
        args.extend_from_slice(&["--stats", stats.to_str().unwrap()]);
        args.extend_from_slice(extra);
        let o = run(&args);
        assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
        let obj = std::fs::read_to_string(out).unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(stats).unwrap()).unwrap();
        (obj, stats)
    };
    let body = |obj: &str| {
        obj.lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let faces = |obj: &str| {
        obj.lines()
            .filter(|l| l.starts_with("f "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let (exact, se) = mesh("ubq-exact.obj", &[]);
    let (fast, sf) = mesh("ubq-fast.obj", &["--refine", "fast"]);
    assert_eq!(body(&fast), body(&exact));
    assert!(fast.contains("refine: fast"));
    assert_eq!(sf["parameters"]["refine"], "fast");
    assert_eq!(sf["engine"]["refine_cpu_edges"], 0);
    assert!(se["engine"].get("refine_cpu_edges").is_none());
    if cfg!(feature = "hip") && std::env::var("SESMESH_GPU_REQUIRE").as_deref() == Ok("1") {
        let (gpu, sg) = mesh(
            "ubq-fast-hip.obj",
            &["--refine", "fast", "--backend", "hip"],
        );
        assert_eq!(faces(&gpu), faces(&exact));
        assert_eq!(sg["engine"]["backend"], "hip");
        assert!(sg["engine"]["refine_cpu_edges"].as_u64().is_some());
        let (again, _) = mesh(
            "ubq-fast-hip2.obj",
            &["--refine", "fast", "--backend", "hip"],
        );
        assert_eq!(again, gpu);
    }
}

/// `--backend` (Plan T11.2): every back end writes the CPU's bytes; a back
/// end missing from the build is a usage error, an unusable device a
/// runtime error.
#[test]
fn backends() {
    let ubq = fixture("1UBQ.cif.gz");
    let mesh = |name: &str, extra: &[&str]| {
        let out = tmp(name);
        let mut args = vec![ubq.to_str().unwrap(), "-s", "0.6", "--quiet", "-o"];
        args.push(out.to_str().unwrap());
        args.extend_from_slice(extra);
        let o = run(&args);
        (o, out)
    };
    let (o, cpu) = mesh("ubq-cpu.obj", &[]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let cpu = std::fs::read(cpu).unwrap();
    let (o, explicit) = mesh("ubq-cpu2.obj", &["--backend", "cpu"]);
    assert_eq!(code(&o), 0);
    assert_eq!(std::fs::read(explicit).unwrap(), cpu);
    let (o, auto) = mesh("ubq-auto.obj", &["--backend", "auto"]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        std::fs::read(auto).unwrap(),
        cpu,
        "--backend auto changed the mesh"
    );

    let (o, hip) = mesh("ubq-hip.obj", &["--backend", "hip"]);
    let err = String::from_utf8_lossy(&o.stderr);
    if cfg!(feature = "hip") {
        match code(&o) {
            0 => assert_eq!(
                std::fs::read(hip).unwrap(),
                cpu,
                "--backend hip changed the mesh"
            ),
            1 => assert!(
                err.contains("GPU back end unavailable: hip device 0"),
                "{err}"
            ),
            c => panic!("exit code {c}: {err}"),
        }
    } else {
        assert_eq!(code(&o), 2, "{err}");
        assert!(err.contains("--backend hip is not in this build"), "{err}");
    }
    let (o, _) = mesh("ubq-dev.obj", &["--gpu-device", "1"]);
    assert_eq!(code(&o), 2);
    assert!(String::from_utf8_lossy(&o.stderr).contains("--gpu-device needs --backend hip"));
    if cfg!(feature = "hip") {
        let (o, _) = mesh("ubq-dev.obj", &["--backend", "hip", "--gpu-device", "999"]);
        assert_eq!(code(&o), 1, "{}", String::from_utf8_lossy(&o.stderr));
    }

    let stats = tmp("ubq-auto.json");
    let (o, _) = mesh(
        "ubq-auto.ply",
        &["--backend", "auto", "--stats", stats.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let stats: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&stats).unwrap()).unwrap();
    assert_eq!(stats["parameters"]["backend"], "auto");
    let backend = stats["engine"]["backend"].as_str().unwrap();
    assert!(matches!(backend, "cpu" | "hip"), "{backend}");
    assert_eq!(stats["engine"]["device"].is_null(), backend == "cpu");
}
