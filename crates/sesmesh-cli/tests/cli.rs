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
