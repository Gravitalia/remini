#![forbid(unsafe_code)]
#![deny(dead_code, unused_imports, unused_mut)]
//! remini grpc server.

use remini::remini_server::{Remini as Rem, ReminiServer};
use remini::{NudityReply, NudityRequest};
use std::path::Path;
use tonic::{
    codec::CompressionEncoding, transport::Server, Request, Response, Status,
};
use tracing::{error, Level};
use tracing_subscriber::fmt;

pub mod remini {
    tonic::include_proto!("remini");
}

struct Remini {
    /// [`corpus::Corpus`] model structure.
    corpus: corpus::Corpus,
}

#[tonic::async_trait]
impl Rem for Remini {
    async fn nudity(
        &self,
        request: Request<NudityRequest>,
    ) -> Result<Response<NudityReply>, Status> {
        let (confidence, is_nude) = self
            .corpus
            .predict(&request.into_inner().image)
            .map_err(|error| {
                error!("Failed to predicts image nudity: {:?}", error);
                Status::invalid_argument("prediction failed")
            })?;

        Ok(Response::new(NudityReply {
            result: is_nude.into(),
            confidence,
        }))
    }
}

#[tokio::main]
async fn main() -> remini_error::Result<()> {
    #[cfg(not(debug_assertions))]
    fmt()
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    #[cfg(debug_assertions)]
    fmt()
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_max_level(Level::TRACE)
        .init();

    let addr = format!(
        "0.0.0.0:{}",
        std::env::var("port").unwrap_or("50051".to_string())
    )
    .parse()?;

    // Init every models.
    let remini = Remini {
        corpus: corpus::Corpus::load(
            Path::new("./corpus/model.onnx").to_path_buf(),
        )?,
    };

    Server::builder()
        .add_service(
            ReminiServer::new(remini)
                .accept_compressed(CompressionEncoding::Zstd)
                .max_decoding_message_size(1024 * 1024) // 1MB file maximum.
                .max_encoding_message_size(1 * 1024), // Only send 1kB.
        )
        .serve(addr)
        .await?;

    Ok(())
}
