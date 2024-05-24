#![forbid(unsafe_code)]
#![deny(dead_code, unused_imports, unused_mut)]
//! remini grpc server.

use remini::remini_server::{Remini as Rem, ReminiServer};
use remini::{BytesRequest, NudityReply, StringRequest, ToxicityReply};
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
    // [`superego::Superego`] model structure.
    superego: superego::Superego,
}

#[tonic::async_trait]
impl Rem for Remini {
    async fn nudity(
        &self,
        request: Request<BytesRequest>,
    ) -> Result<Response<NudityReply>, Status> {
        let (confidence, is_nude) = self
            .corpus
            .predict(&request.into_inner().bytes)
            .map_err(|error| {
                error!("Failed to predicts image nudity: {:?}", error);
                Status::invalid_argument("prediction failed")
            })?;

        Ok(Response::new(NudityReply {
            result: is_nude.into(),
            confidence,
        }))
    }

    async fn toxicity(
        &self,
        request: Request<StringRequest>,
    ) -> Result<Response<ToxicityReply>, Status> {
        println!("{:?}", self.superego.predict(request.into_inner().string));

        Ok(Response::new(ToxicityReply {
            identity_hate: 0.0,
            insult: 0.0,
            neutral: 1.0,
            obscene: 0.0,
            severe_toxic: 0.0,
            threat: 0.0,
            toxic: 0.0,
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
        superego: superego::Superego::load(
            Path::new("./superego/model.onnx").to_path_buf(),
            Path::new("./superego/tokenizer.json").to_path_buf(),
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
