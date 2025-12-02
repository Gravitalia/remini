use corpus::Corpus;

use std::fs::read;
use std::path::Path;
use std::time::SystemTime;

const MODELS: [&str; 2] = [
    "corpus/models/model-large.onnx",
    "corpus/models/small-dynamic.onnx",
];
const IMG_PATH: &str = "corpus/examples/image.jpg";

fn main() {
    let buffer = read(IMG_PATH).expect("image does not exists");
    // `corpus-small-dynamic` works but tract don't optimize it,
    // which leads to slower execution time.
    let path = Path::new(".").join(MODELS[0]);
    let corpus = Corpus::load(path.to_path_buf()).expect("model not found");
    let start = SystemTime::now();
    let prediction = corpus
        .predict(buffer)
        .expect("failed to predict image nudity");
    let end = SystemTime::now().duration_since(start).unwrap().as_millis();
    let score = prediction * 100.0;
    println!("Probability to be a nude is {score:?}% in {end:?}ms");
}
