//! `sesmesh` command-line tool.
//!
//! Scaffold only (Phase 0): `--help` and `--version` work; surface
//! computation arrives with the MVP (Plan Phase 4).

use std::process::ExitCode;

use clap::Parser;

/// Compute Solvent Excluded Surfaces (Connolly surfaces) of molecules and
/// write them as triangle meshes.
#[derive(Debug, Parser)]
#[command(name = "sesmesh", version, about, long_about = None)]
struct Cli {}

fn main() -> ExitCode {
    let _cli = Cli::parse();
    eprintln!(
        "sesmesh {}: surface computation is not implemented yet",
        sesmesh::VERSION
    );
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
