#![forbid(unsafe_code)]
//! internal library to provide structures for errors in Remini.

/// Custom `Result` type for remini.
pub type Result<T> = std::result::Result<T, Error>;

/// List of remini errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot permute axes")]
    Permutation,
    #[error("model returned nothing")]
    Execution,
    #[error(transparent)]
    Tract(#[from] anyhow::Error),

    #[error("image conversion error: {0}")]
    Image(#[from] image::error::ImageError),

    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[error("unknown error")]
    Unspecified,
}
