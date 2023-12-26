pub mod helpers;

use remini::remini_server::{Remini as Rem, ReminiServer};
use remini::{Reply as ReminiReply, Request as ReminiRequest};
use tonic::{transport::Server, Request, Response, Status};

pub mod remini {
    tonic::include_proto!("remini");
}

#[derive(Debug, Default)]
pub struct Remini {}

#[tonic::async_trait]
impl Rem for Remini {
    async fn predict(
        &self,
        request: Request<ReminiRequest>, // Accept request of type HelloRequest
    ) -> Result<Response<ReminiReply>, Status> {
        let content = request.into_inner();

        Ok(Response::new(ReminiReply {
            model: content.model,
            message: "OK".to_string(),
            error: false,
        }))
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
    let remini = Remini::default();

    log::info!("Server started on {}", addr);

    Server::builder()
        .add_service(ReminiServer::new(remini))
        .serve(addr)
        .await?;

    Ok(())
}
