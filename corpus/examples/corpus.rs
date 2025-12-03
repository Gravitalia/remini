use corpus::Corpus;
use tracing_subscriber::prelude::*;

use std::fs::read;
use std::path::Path;
use std::time::Instant;

const MODELS: [&str; 2] = [
    "corpus/models/model-large.onnx",
    "corpus/models/small-dynamic.onnx",
];
const IMG_PATH: &str = "corpus/examples/image.jpg";

fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    format!("{}=debug", env!("CARGO_CRATE_NAME")).into()
                }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let buffer = read(IMG_PATH).expect("image does not exists");
    // `corpus-small-dynamic` works but tract don't optimize it,
    // which leads to slower execution time.
    let path = Path::new(".").join(MODELS[0]);
    let corpus = Corpus::load(path.to_path_buf()).expect("model not found");
    let start = Instant::now();
    let score = corpus
        .predict(buffer)
        .expect("failed to predict image nudity");
    let end = start.elapsed().as_millis();
    tracing::info!(?score, time = end, "image inferred");
}
