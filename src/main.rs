pub mod corpus;
pub mod helpers;

use corpus::CorpusManager;
use remini::remini_server::{Remini as Rem, ReminiServer};
use remini::{Reply as ReminiReply, Request as ReminiRequest};
use tonic::{transport::Server, Request, Response, Status};

pub type Model = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::Graph<
        tract_onnx::prelude::TypedFact,
        Box<dyn tract_onnx::prelude::TypedOp>,
    >,
>;

pub mod remini {
    tonic::include_proto!("remini");
}

struct Remini {
    /// Corpus model to detect nodity on a content.
    corpus: corpus::Corpus,
}

#[tonic::async_trait]
impl Rem for Remini {
    async fn predict(
        &self,
        request: Request<ReminiRequest>, // Accept request of type HelloRequest
    ) -> Result<Response<ReminiReply>, Status> {
        let content = request.into_inner();

        match content.model.as_str() {
            "corpus" => match self.corpus.predict(&content.data) {
                Ok(result) => Ok(Response::new(ReminiReply {
                    model: content.model,
                    message: result,
                    error: false,
                })),
                Err(error) => {
                    log::error!("Corpus model got an error; {}", error);

                    Ok(Response::new(ReminiReply {
                        model: content.model,
                        message: "Internal server error".to_string(),
                        error: true,
                    }))
                }
            },
            _ => Ok(Response::new(ReminiReply {
                model: content.model,
                message: "Unknown model".to_string(),
                error: true,
            })),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set logger with Fern.
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {}] {}",
                helpers::format_rfc3339(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_secs()
                ),
                record.level(),
                message
            ))
        })
        .level(if cfg!(debug_assertions) {
            log::LevelFilter::Trace
        } else {
            log::LevelFilter::Info
        })
        .chain(std::io::stdout())
        .apply()
        .unwrap();

    let addr = format!(
        "[::1]:{}",
        std::env::var("port").unwrap_or("50051".to_string())
    )
    .parse()?;

    // Init every models.
    let remini = Remini {
        corpus: corpus::Corpus {
            model: corpus::init()?,
        },
    };

    log::info!("Server started on {}", addr);

    Server::builder()
        .add_service(ReminiServer::new(remini))
        .serve(addr)
        .await?;

    Ok(())
}
