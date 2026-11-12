//! Error type shared by every fallible sesmesh operation.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::io::Format;

/// Result alias using [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors returned by sesmesh.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Reading a file or stream failed.
    #[error("{}: {source}", display_path(path.as_deref()))]
    Io {
        /// File being read, if any.
        path: Option<PathBuf>,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The input could not be parsed.
    #[error("{}:{line}:{column}: invalid {format} input: {message}", display_path(path.as_deref()))]
    Parse {
        /// Format being parsed.
        format: Format,
        /// File being parsed, if any.
        path: Option<PathBuf>,
        /// 1-based line number.
        line: usize,
        /// 1-based column number (in bytes).
        column: usize,
        /// Description of the problem.
        message: String,
    },

    /// The input format could not be determined or is not supported by this
    /// build (for example gzip input without the `gzip` feature).
    #[error("unsupported input: {0}")]
    Unsupported(String),

    /// An argument or input value is invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// An atom selection matched no atoms.
    #[error("no atoms selected: {0}")]
    EmptySelection(String),

    /// A configured resource limit would be exceeded.
    #[error("{what} exceeds the configured limit of {limit} bytes")]
    ResourceLimit {
        /// What hit the limit.
        what: String,
        /// The limit in bytes.
        limit: u64,
    },
}

fn display_path(path: Option<&Path>) -> String {
    path.map_or_else(|| "<input>".to_owned(), |p| p.display().to_string())
}

/// Parse failure produced by the format readers before the file path and
/// format are attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParseError {
    pub(crate) line: usize,
    pub(crate) column: usize,
    pub(crate) message: String,
}

impl ParseError {
    pub(crate) fn new(line: usize, column: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            column,
            message: message.into(),
        }
    }

    pub(crate) fn into_error(self, format: Format, path: Option<PathBuf>) -> Error {
        Error::Parse {
            format,
            path,
            line: self.line,
            column: self.column,
            message: self.message,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}
