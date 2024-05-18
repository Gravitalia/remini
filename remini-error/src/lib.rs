#![forbid(unsafe_code)]
#![deny(
    dead_code,
    unused_imports,
    unused_mut,
    missing_docs,
    missing_debug_implementations
)]
//! internal library to provide structures for errors in Remini.
//!
//! # Examples
//! ```rust
//! use remini_error::Result;
//!
//! fn main() -> Result<()> {
//!     Ok(())
//! }
//! ```

use std::error::Error as StdError;
use std::fmt;

/// Boxed error to bypass specific [Error](StdError).
type BError = Box<dyn StdError + Send + Sync>;
/// anyhow-like error handler.
pub type Result<T> = core::result::Result<T, BError>;

/// The struct that represents an error
#[derive(Debug)]
pub struct Error {
    /// The error type.
    pub etype: ErrorType,
    /// The cause of this error.
    pub cause: Option<BError>,
    /// Explains the context in which the error occurs.
    pub context: Option<String>,
}

impl Error {
    /// Throw an [`Error`].
    pub fn new(
        etype: ErrorType,
        cause: Option<BError>,
        context: Option<String>,
    ) -> Self {
        Error {
            etype,
            cause,
            context,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.etype)
    }
}
impl StdError for Error {}

/// Errors in Squid.
#[derive(Debug)]
pub enum ErrorType {
    /// Generic error that returns no additional information.
    Unspecified,
    /// Errors related to [`tract-onnx`] and models.
    Algorithms,
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorType::Unspecified => {
                write!(f, "An error has occurred, but no further information is provided.")
            },
            ErrorType::Algorithms => write!(
                f,
                "Machine learning algorithms cannot achieve calculations."
            ),
        }
    }
}
impl StdError for ErrorType {}
