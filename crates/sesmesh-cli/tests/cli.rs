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
