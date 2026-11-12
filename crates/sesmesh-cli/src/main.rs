//! `sesmesh` command-line tool (Spec FR-CLI, Design §14).
//!
//! ```text
//! sesmesh [surface] <INPUT> [-o <OUTPUT>] [OPTIONS]
//! sesmesh info <INPUT>
//! sesmesh radii [--set bondi|alvarez|protor]
//! ```
//!
//! Exit codes (FR-CLI-6): 0 success, 1 runtime error, 2 usage error, 3 input
//! parse error, 4 resource limit exceeded.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io::{self, BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use sesmesh::export::obj::{ObjOptions, write_obj, write_obj_path};
use sesmesh::io::{Format, ReadOptions};
use sesmesh::radii::{BuiltinRadii, RadiusOptions, RadiusRules, RadiusSet};
use sesmesh::structure::{AltLocPolicy, ChainIdKind, ModelChoice, WATER_RESIDUES};
use sesmesh::{
    AtomSet, Control, Diagnostics, Error, Progress, Quality, RefineMode, Selection, SimdLevel,
    Sphere, Stage, Structure, SurfaceKind, SurfaceOutput, SurfaceParams, compute_surface_with,
};

/// Compute Solvent Excluded Surfaces (Connolly surfaces) of molecules and
/// write them as triangle meshes with normals.
///
/// `sesmesh <INPUT>` is short for `sesmesh surface <INPUT>`.
#[derive(Debug, Parser)]
#[command(name = "sesmesh", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compute a surface and write it as an OBJ mesh (default command).
    Surface(Box<SurfaceArgs>),
    /// Summarize an input: models, chains, atoms, bounding box, radii coverage.
    Info(Box<InfoArgs>),
    /// Print a built-in radius table.
    Radii(RadiiArgs),
}

#[derive(Debug, Args)]
struct InputArgs {
    /// Input structure: PDB, mmCIF, XYZ or PQR, optionally gzip-compressed.
    input: PathBuf,
    /// Input format (default: from the file name, then the content).
    #[arg(long, value_enum, default_value_t = FormatArg::Auto, help_heading = "Input")]
    format: FormatArg,
    /// Model number (PDB `MODEL`, mmCIF `pdbx_PDB_model_num`); default: the
    /// first model.
    #[arg(long, help_heading = "Input", conflicts_with = "frame")]
    model: Option<i32>,
    /// 0-based model position, e.g. an XYZ trajectory frame.
    #[arg(long, help_heading = "Input")]
    frame: Option<usize>,
    /// mmCIF data block (default: the first block with atoms).
    #[arg(long, help_heading = "Input")]
    block: Option<String>,
}

#[derive(Debug, Args)]
struct SelectArgs {
    /// Alternate locations: `highest` (occupancy per residue), `first`,
    /// `all`, or a conformer letter.
    #[arg(long, default_value = "highest", help_heading = "Selection")]
    altloc: String,
    /// Keep hydrogen atoms.
    #[arg(long, help_heading = "Selection")]
    hydrogens: bool,
    /// Keep waters.
    #[arg(long, help_heading = "Selection")]
    waters: bool,
    /// Drop hetero (HETATM) atoms.
    #[arg(long, help_heading = "Selection")]
    no_hetatm: bool,
    /// Keep only these chains (comma-separated).
    #[arg(long, value_delimiter = ',', help_heading = "Selection")]
    chains: Option<Vec<String>>,
    /// Which chain identifiers `--chains` refers to.
    #[arg(long, value_enum, default_value_t = ChainIds::Auth, help_heading = "Selection")]
    chain_ids: ChainIds,
    /// Drop residues with these names (comma-separated).
    #[arg(long, value_delimiter = ',', help_heading = "Selection")]
    exclude_resn: Vec<String>,
}

#[derive(Debug, Args)]
struct RadiusArgs {
    /// Radius set: `bondi`, `alvarez`, `protor`, `uniform:<r>` or
    /// `file:<path>` (TOML rules).
    #[arg(long, default_value = "bondi", help_heading = "Radii")]
    radii: String,
    /// Ignore radii given by the input (PQR, extended XYZ).
    #[arg(long, help_heading = "Radii")]
    no_file_radii: bool,
    /// Radius (Å) for atoms the set does not cover.
    #[arg(long, default_value_t = 1.8, help_heading = "Radii")]
    unknown_radius: f64,
    /// Fail if some atom has no radius in the set.
    #[arg(long, help_heading = "Radii")]
    strict_radii: bool,
}

#[derive(Debug, Args)]
struct SurfaceArgs {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    select: SelectArgs,
    #[command(flatten)]
    radius: RadiusArgs,
    /// Surface kind.
    #[arg(long, value_enum, default_value_t = KindArg::Ses, help_heading = "Surface")]
    surface: KindArg,
    /// Probe radius (Å).
    #[arg(long, default_value_t = 1.4, help_heading = "Surface")]
    probe: f64,
    /// Grid spacing (Å); overrides `--quality`.
    #[arg(short = 's', long, help_heading = "Resolution")]
    grid_spacing: Option<f64>,
    /// Quality preset: draft (1.0 Å), low (0.7), medium (0.5), high (0.25),
    /// ultra (0.1).
    #[arg(short = 'q', long, value_enum, default_value_t = QualityArg::Medium, help_heading = "Resolution")]
    quality: QualityArg,
    /// Vertex placement on grid edges.
    #[arg(long, value_enum, default_value_t = RefineArg::Exact, help_heading = "Resolution")]
    refine: RefineArg,
    /// Brick edge in grid cells (expert).
    #[arg(long, default_value_t = 10, help_heading = "Resolution")]
    brick_size: usize,
    /// Output file (`-`: standard output; a `.gz` suffix compresses).
    /// Default: the input name with an `.obj` extension, in the current
    /// directory.
    #[arg(short = 'o', long, help_heading = "Output")]
    output: Option<PathBuf>,
    /// Decimals of coordinates and normals (0–9).
    #[arg(long, default_value_t = 4, help_heading = "Output")]
    precision: usize,
    /// Omit vertex normals.
    #[arg(long, help_heading = "Output")]
    no_normals: bool,
    /// Worker threads (default: all logical cores).
    #[arg(short = 'j', long, help_heading = "Performance")]
    threads: Option<usize>,
    /// Field kernel instruction set (`auto`: best supported; explicit x86
    /// levels are capped at what the CPU supports).
    #[arg(long, value_enum, default_value_t = SimdArg::Auto, help_heading = "Performance")]
    simd: SimdArg,
    /// Print errors only.
    #[arg(long, help_heading = "Diagnostics")]
    quiet: bool,
    /// Report progress on standard error (a live line on a terminal, one
    /// line per stage otherwise).
    #[arg(long, help_heading = "Diagnostics")]
    progress: bool,
    /// Print stage timings.
    #[arg(long, help_heading = "Diagnostics")]
    timings: bool,
}

#[derive(Debug, Args)]
struct InfoArgs {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    select: SelectArgs,
    #[command(flatten)]
    radius: RadiusArgs,
}

#[derive(Debug, Args)]
struct RadiiArgs {
    /// Radius set to print.
    #[arg(long, value_enum, default_value_t = SetArg::Bondi)]
    set: SetArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FormatArg {
    Auto,
    Pdb,
    Mmcif,
    Xyz,
    Pqr,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ChainIds {
    Auth,
    Label,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KindArg {
    Ses,
    Sas,
    Vdw,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QualityArg {
    Draft,
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RefineArg {
    Exact,
    Linear,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SimdArg {
    Auto,
    Scalar,
    #[value(name = "sse4.2")]
    Sse42,
    Avx2,
    Avx512,
    Neon,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SetArg {
    Bondi,
    Alvarez,
    Protor,
}

/// A failure with its exit code (FR-CLI-6).
#[derive(Debug)]
enum Failure {
    Runtime(String),
    Usage(String),
    Parse(String),
    Resource(String),
}

impl Failure {
    fn code(&self) -> u8 {
        match self {
            Self::Runtime(_) => 1,
            Self::Usage(_) => 2,
            Self::Parse(_) => 3,
            Self::Resource(_) => 4,
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Runtime(m) | Self::Usage(m) | Self::Parse(m) | Self::Resource(m) => m,
        }
    }
}

impl From<Error> for Failure {
    fn from(e: Error) -> Self {
        let m = e.to_string();
        match e {
            Error::Parse { .. } | Error::Unsupported(_) => Self::Parse(m),
            Error::ResourceLimit { .. } => Self::Resource(m),
            _ => Self::Runtime(m),
        }
    }
}

type CliResult<T> = Result<T, Failure>;

const SUBCOMMANDS: [&str; 4] = ["surface", "info", "radii", "help"];

/// Inserts the default `surface` subcommand when the first argument is not
/// a subcommand or a top-level flag.
fn with_default_subcommand(mut args: Vec<OsString>) -> Vec<OsString> {
    if let Some(first) = args.get(1).and_then(|a| a.to_str()) {
        let top_level_flag = matches!(first, "-h" | "--help" | "-V" | "--version");
        if !top_level_flag && !SUBCOMMANDS.contains(&first) {
            args.insert(1, "surface".into());
        }
    }
    args
}

fn main() -> ExitCode {
    let cli = Cli::parse_from(with_default_subcommand(std::env::args_os().collect()));
    let result = match cli.command {
        Command::Surface(a) => surface(&a),
        Command::Info(a) => info(&a),
        Command::Radii(a) => radii(&a),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(f) => {
            eprintln!("sesmesh: error: {}", f.message());
            ExitCode::from(f.code())
        }
    }
}

fn read(input: &InputArgs) -> CliResult<Structure> {
    let mut options = ReadOptions::default();
    options.format = match input.format {
        FormatArg::Auto => None,
        FormatArg::Pdb => Some(Format::Pdb),
        FormatArg::Mmcif => Some(Format::Mmcif),
        FormatArg::Xyz => Some(Format::Xyz),
        FormatArg::Pqr => Some(Format::Pqr),
    };
    options.data_block.clone_from(&input.block);
    Ok(sesmesh::io::read_path(&input.input, &options)?)
}

fn selection(input: &InputArgs, a: &SelectArgs) -> CliResult<Selection> {
    let mut s = Selection::default();
    s.model = match (input.model, input.frame) {
        (Some(n), _) => ModelChoice::Number(n),
        (None, Some(i)) => ModelChoice::Index(i),
        (None, None) => ModelChoice::First,
    };
    s.alt_loc = match a.altloc.to_ascii_lowercase().as_str() {
        "highest" => AltLocPolicy::HighestOccupancy,
        "first" => AltLocPolicy::First,
        "all" => AltLocPolicy::All,
        _ => {
            let mut chars = a.altloc.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => AltLocPolicy::Only(c),
                _ => {
                    return Err(Failure::Usage(format!(
                        "invalid --altloc {:?} (highest, first, all or a letter)",
                        a.altloc
                    )));
                }
            }
        }
    };
    s.hydrogens = a.hydrogens;
    s.waters = a.waters;
    s.hetero = !a.no_hetatm;
    s.chains.clone_from(&a.chains);
    s.chain_ids = match a.chain_ids {
        ChainIds::Auth => ChainIdKind::Auth,
        ChainIds::Label => ChainIdKind::Label,
    };
    s.exclude_residues.clone_from(&a.exclude_resn);
    Ok(s)
}

fn radius_options(a: &RadiusArgs) -> CliResult<RadiusOptions> {
    let set = if let Some(path) = a.radii.strip_prefix("file:") {
        RadiusSet::Custom(Box::new(RadiusRules::from_path(path)?))
    } else {
        RadiusSet::from_name(&a.radii).ok_or_else(|| {
            Failure::Usage(format!(
                "invalid --radii {:?} (bondi, alvarez, protor, uniform:<r>, file:<path>)",
                a.radii
            ))
        })?
    };
    let mut o = RadiusOptions::new(set);
    o.use_file_radii = !a.no_file_radii;
    o.unknown_radius = a.unknown_radius;
    o.strict = a.strict_radii;
    Ok(o)
}

/// Selected atoms with their radii.
fn atoms_and_radii<'a>(
    structure: &'a Structure,
    input: &InputArgs,
    select: &SelectArgs,
    radius: &RadiusArgs,
    diags: &mut Diagnostics,
) -> CliResult<(AtomSet<'a>, Vec<f64>)> {
    let selection = selection(input, select)?;
    let options = radius_options(radius)?;
    let atoms = structure.select(&selection, diags)?;
    let radii = sesmesh::radii::assign(&atoms, &options, diags)?;
    Ok((atoms, radii))
}

fn params(a: &SurfaceArgs) -> CliResult<SurfaceParams> {
    let base = match a.surface {
        KindArg::Ses => SurfaceParams::ses(a.probe),
        KindArg::Sas => SurfaceParams::sas(a.probe),
        KindArg::Vdw => SurfaceParams::vdw(),
    };
    let base = match a.grid_spacing {
        Some(h) => base.with_spacing(h),
        None => base.with_quality(match a.quality {
            QualityArg::Draft => Quality::Draft,
            QualityArg::Low => Quality::Low,
            QualityArg::Medium => Quality::Medium,
            QualityArg::High => Quality::High,
            QualityArg::Ultra => Quality::Ultra,
        }),
    };
    let p = base
        .with_refine(match a.refine {
            RefineArg::Exact => RefineMode::Exact,
            RefineArg::Linear => RefineMode::Linear,
        })
        .with_brick_size(a.brick_size)
        .with_simd(match a.simd {
            SimdArg::Auto => SimdLevel::Auto,
            SimdArg::Scalar => SimdLevel::Scalar,
            SimdArg::Sse42 => SimdLevel::Sse4_2,
            SimdArg::Avx2 => SimdLevel::Avx2,
            SimdArg::Avx512 => SimdLevel::Avx512,
            SimdArg::Neon => SimdLevel::Neon,
        });
    p.validate().map_err(|e| Failure::Usage(e.to_string()))?;
    if p.simd.effective().is_none() {
        return Err(Failure::Usage(format!(
            "--simd {} is not available on this CPU",
            p.simd.name()
        )));
    }
    if a.precision > 9 {
        return Err(Failure::Usage(format!(
            "--precision must be 0..=9, got {}",
            a.precision
        )));
    }
    Ok(p)
}

/// `1crn.cif.gz` → `1crn.obj`.
fn default_output(input: &Path) -> PathBuf {
    let name = input.file_name().map_or_else(
        || "surface".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let name = name
        .strip_suffix(".gz")
        .or_else(|| name.strip_suffix(".GZ"))
        .unwrap_or(&name);
    let stem = Path::new(name)
        .file_stem()
        .map_or_else(|| "surface".into(), |s| s.to_string_lossy().into_owned());
    PathBuf::from(format!("{stem}.obj"))
}

fn kind_name(k: SurfaceKind) -> &'static str {
    match k {
        SurfaceKind::Ses => "ses",
        SurfaceKind::Sas => "sas",
        _ => "vdw",
    }
}

fn ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1e3)
}

fn print_warnings(diags: &Diagnostics) {
    for w in diags.warnings() {
        eprintln!("sesmesh: warning: {w}");
    }
}

fn surface(a: &SurfaceArgs) -> CliResult<()> {
    let start = Instant::now();
    let params = params(a)?;
    if let Some(n) = a.threads {
        if n == 0 {
            return Err(Failure::Usage("--threads must be at least 1".into()));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| Failure::Runtime(format!("thread pool: {e}")))?;
    }
    let structure = read(&a.input)?;
    let mut diags = structure.diagnostics.clone();
    let (atoms, radii) = atoms_and_radii(&structure, &a.input, &a.select, &a.radius, &mut diags)?;
    let spheres = Sphere::from_atoms(&atoms, &radii)?;
    let read_time = start.elapsed();
    let reporter = ProgressLine::new(io::stderr().is_terminal());
    let report = |p: Progress| reporter.update(p);
    let mut control = Control::new();
    if a.progress && !a.quiet {
        control = control.with_progress(&report);
    }
    let out = compute_surface_with(&spheres, &params, &control)?;
    reporter.finish();
    diags.merge(out.diagnostics.clone());

    let write_start = Instant::now();
    let mut options = ObjOptions::default();
    options.precision = a.precision;
    options.normals = !a.no_normals;
    options.object_name = kind_name(out.stats.kind).to_string();
    options.comments = header(a, &structure, &atoms, &params, &out);
    let output = a
        .output
        .clone()
        .unwrap_or_else(|| default_output(&a.input.input));
    let to_stdout = output.as_os_str() == "-";
    if to_stdout {
        let stdout = io::stdout().lock();
        let mut w = BufWriter::with_capacity(1 << 20, stdout);
        write_obj(&mut w, &out.mesh, &options)?;
        w.flush()
            .map_err(|e| Failure::Runtime(format!("standard output: {e}")))?;
    } else {
        write_obj_path(&output, &out.mesh, &options)?;
    }
    let write_time = write_start.elapsed();

    if !a.quiet {
        print_warnings(&diags);
        let s = &out.stats;
        eprintln!(
            "sesmesh: {}: {} atoms (model {}), {} surface, probe {:.2} Å, grid {:.3} Å",
            a.input.input.display(),
            atoms.len(),
            atoms.model().number,
            kind_name(s.kind),
            params.effective_probe(),
            s.spacing,
        );
        eprintln!(
            "sesmesh: {} vertices, {} triangles, {} component(s) ({} cavit{}), area {:.2} Å², volume {:.2} Å³",
            s.vertices,
            s.triangles,
            s.components,
            s.cavities,
            if s.cavities == 1 { "y" } else { "ies" },
            s.area,
            s.volume,
        );
        if a.timings {
            let t = &s.timings;
            eprintln!(
                "sesmesh: time: read {}, geometry {}, classify {}, mesh {}, assemble {}, measure {}, write {} ({}, {} kernel)",
                ms(read_time),
                ms(t.geometry),
                ms(t.classify),
                ms(t.mesh),
                ms(t.assemble),
                ms(t.measure),
                ms(write_time),
                ms(start.elapsed()),
                s.simd,
            );
        }
        if !to_stdout {
            eprintln!(
                "sesmesh: wrote {} in {:.2} s",
                output.display(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

/// Progress display on standard error (FR-CLI-4). Reports may arrive from
/// any worker thread; the line only moves forward.
struct ProgressLine {
    tty: bool,
    /// Current stage, last percentage shown, whether a live line is open.
    state: Mutex<(Option<Stage>, u64, bool)>,
}

impl ProgressLine {
    fn new(tty: bool) -> Self {
        Self {
            tty,
            state: Mutex::new((None, 0, false)),
        }
    }

    fn update(&self, p: Progress) {
        let Ok(mut st) = self.state.lock() else {
            return;
        };
        let pct = (p.done * 100).checked_div(p.total).unwrap_or(100);
        let new_stage = st.0 != Some(p.stage);
        if !new_stage && pct <= st.1 {
            return;
        }
        if new_stage {
            *st = (Some(p.stage), 0, st.2);
        }
        st.1 = pct;
        if self.tty {
            eprint!("\rsesmesh: {:<9} {pct:>3}%", p.stage.name());
            st.2 = true;
        } else if p.done == p.total {
            eprintln!("sesmesh: {} done", p.stage.name());
        }
    }

    fn finish(&self) {
        if let Ok(mut st) = self.state.lock()
            && st.2
        {
            eprintln!();
            st.2 = false;
        }
    }
}

/// OBJ header comments (FR-OUT-1).
fn header(
    a: &SurfaceArgs,
    structure: &Structure,
    atoms: &AtomSet<'_>,
    params: &SurfaceParams,
    out: &SurfaceOutput,
) -> Vec<String> {
    let s = &out.stats;
    vec![
        format!(
            "input: {}  format: {}  model: {}  atoms: {} selected / {} read",
            a.input.input.display(),
            structure.source.format.name(),
            atoms.model().number,
            atoms.len(),
            atoms.model().atoms.len(),
        ),
        format!(
            "surface: {}  probe_radius: {:.4}  grid_spacing: {:.4}  radii: {}  refine: {}",
            kind_name(s.kind),
            params.effective_probe(),
            s.spacing,
            a.radius.radii,
            params.refine.name(),
        ),
        format!(
            "vertices: {}  triangles: {}  components: {}  cavities: {}  euler: {}",
            s.vertices, s.triangles, s.components, s.cavities, s.euler
        ),
        format!("area_A2: {:.4}  volume_A3: {:.4}", s.area, s.volume),
        "normals: analytic, pointing to the solvent; faces counter-clockwise seen from outside"
            .to_string(),
    ]
}

fn info(a: &InfoArgs) -> CliResult<()> {
    let structure = read(&a.input)?;
    let mut out = String::new();
    let line = |out: &mut String, s: String| {
        out.push_str(&s);
        out.push('\n');
    };
    let src = &structure.source;
    line(&mut out, format!("file: {}", a.input.input.display()));
    line(
        &mut out,
        format!(
            "format: {}{}",
            src.format.name(),
            src.data_block
                .as_deref()
                .map_or_else(String::new, |b| format!(" (data block {b})"))
        ),
    );
    let numbers: Vec<String> = structure
        .models
        .iter()
        .take(10)
        .map(|m| m.number.to_string())
        .collect();
    line(
        &mut out,
        format!(
            "models: {} ({}{})",
            structure.models.len(),
            numbers.join(", "),
            if structure.models.len() > 10 {
                ", ..."
            } else {
                ""
            }
        ),
    );
    line(
        &mut out,
        format!("atoms (all models): {}", structure.atom_count()),
    );

    let mut diags = Diagnostics::new();
    let selection = selection(&a.input, &a.select)?;
    let options = radius_options(&a.radius)?;
    match structure.select(&selection, &mut diags) {
        Ok(atoms) => {
            let model = atoms.model();
            line(
                &mut out,
                format!("model {}: {} atoms", model.number, model.atoms.len()),
            );
            summarize_model(&mut out, &structure, model);
            line(&mut out, format!("selected: {} atoms", atoms.len()));
            let positions = atoms.positions();
            if let Some(first) = positions.first() {
                let (lo, hi) = positions.iter().fold((*first, *first), |(lo, hi), p| {
                    (
                        [0, 1, 2].map(|k| lo[k].min(p[k])),
                        [0, 1, 2].map(|k| hi[k].max(p[k])),
                    )
                });
                let f = |v: [f64; 3]| format!("[{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2]);
                line(
                    &mut out,
                    format!(
                        "bounding box: min {} max {} size {}",
                        f(lo),
                        f(hi),
                        f([0, 1, 2].map(|k| hi[k] - lo[k]))
                    ),
                );
            }
            let mut rdiags = Diagnostics::new();
            match sesmesh::radii::assign(&atoms, &options, &mut rdiags) {
                Ok(radii) => {
                    let unknown = rdiags.count(sesmesh::WarningKind::UnknownRadius);
                    let (lo, hi) = radii.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &r| {
                        (lo.min(r), hi.max(r))
                    });
                    line(
                        &mut out,
                        format!(
                            "radii ({}): {} of {} atoms covered, {} fallback ({:.2} Å); range {:.2}–{:.2} Å",
                            a.radius.radii,
                            atoms.len() as u64 - unknown.min(atoms.len() as u64),
                            atoms.len(),
                            unknown,
                            options.unknown_radius,
                            if radii.is_empty() { 0.0 } else { lo },
                            hi
                        ),
                    );
                }
                Err(e) => line(&mut out, format!("radii ({}): {e}", a.radius.radii)),
            }
            diags.merge(rdiags);
        }
        Err(e) => line(&mut out, format!("selection: {e}")),
    }
    let mut all = structure.diagnostics.clone();
    all.merge(diags);
    for w in all.warnings() {
        line(&mut out, format!("warning: {w}"));
    }
    io::stdout()
        .write_all(out.as_bytes())
        .map_err(|e| Failure::Runtime(format!("standard output: {e}")))
}

fn summarize_model(out: &mut String, structure: &Structure, model: &sesmesh::Model) {
    let mut chains: BTreeMap<&str, usize> = BTreeMap::new();
    let mut residues: BTreeSet<(&str, i32, Option<char>)> = BTreeSet::new();
    let mut elements: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut het, mut hydrogens, mut waters, mut altloc) = (0, 0, 0, 0);
    for atom in &model.atoms {
        let chain = structure.str(atom.chain);
        *chains.entry(chain).or_default() += 1;
        residues.insert((chain, atom.res_seq, atom.ins_code));
        let symbol = atom.element.map_or("?", |e| e.symbol());
        *elements.entry(symbol).or_default() += 1;
        het += usize::from(atom.het);
        hydrogens += usize::from(atom.element.is_some_and(sesmesh::Element::is_hydrogen));
        let res = structure.str(atom.res_name);
        waters += usize::from(WATER_RESIDUES.iter().any(|w| w.eq_ignore_ascii_case(res)));
        altloc += usize::from(atom.alt_loc.is_some());
    }
    let chain_list: Vec<String> = chains
        .iter()
        .map(|(c, n)| format!("{}:{n}", if c.is_empty() { "-" } else { c }))
        .collect();
    out.push_str(&format!(
        "chains: {} ({})\nresidues: {}\n",
        chains.len(),
        chain_list.join(" "),
        residues.len()
    ));
    let mut by_count: Vec<(&str, usize)> = elements.into_iter().collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let el: Vec<String> = by_count.iter().map(|(e, n)| format!("{e}:{n}")).collect();
    out.push_str(&format!(
        "elements: {}\nhetero atoms: {het}  hydrogens: {hydrogens}  water atoms: {waters}  alt-loc atoms: {altloc}\n",
        el.join(" ")
    ));
}

fn radii(a: &RadiiArgs) -> CliResult<()> {
    let set = match a.set {
        SetArg::Bondi => BuiltinRadii::Bondi,
        SetArg::Alvarez => BuiltinRadii::Alvarez,
        SetArg::Protor => BuiltinRadii::ProtOr,
    };
    let mut out = String::new();
    let atoms = set.atom_table();
    if !atoms.is_empty() {
        out.push_str(&format!(
            "# {} per-atom radii (Å): residue atom class radius\n",
            set.name()
        ));
        for (res, atom, class, r) in atoms {
            out.push_str(&format!("{res}\t{atom}\t{class}\t{r:.2}\n"));
        }
        out.push_str("# other atoms: element radii\n");
    }
    out.push_str(&format!(
        "# {} element radii (Å): Z symbol radius\n",
        set.name()
    ));
    for (e, r) in set.element_table() {
        out.push_str(&format!("{}\t{}\t{r:.2}\n", e.atomic_number(), e.symbol()));
    }
    io::stdout()
        .write_all(out.as_bytes())
        .map_err(|e| Failure::Runtime(format!("standard output: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn default_subcommand() {
        let args = |v: &[&str]| -> Vec<OsString> { v.iter().map(OsString::from).collect() };
        assert_eq!(
            with_default_subcommand(args(&["sesmesh", "1crn.pdb"])),
            args(&["sesmesh", "surface", "1crn.pdb"])
        );
        assert_eq!(
            with_default_subcommand(args(&["sesmesh", "info", "x"])),
            args(&["sesmesh", "info", "x"])
        );
        assert_eq!(
            with_default_subcommand(args(&["sesmesh", "--help"])),
            args(&["sesmesh", "--help"])
        );
    }

    #[test]
    fn output_names() {
        assert_eq!(
            default_output(Path::new("a/1crn.cif.gz")),
            PathBuf::from("1crn.obj")
        );
        assert_eq!(default_output(Path::new("x.pdb")), PathBuf::from("x.obj"));
        assert_eq!(
            default_output(Path::new("noext")),
            PathBuf::from("noext.obj")
        );
    }
}
