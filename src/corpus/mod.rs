use anyhow::Result;
use tract_onnx::prelude::*;

const IMAGE_WIDTH: u32 = 224;
const IMAGE_HEIGHT: u32 = 224;
const RESULT: [&str; 2] = [
    "not_nude",
    "nude",
];

/// Define a structure to manage the Corpus model.
#[derive(Debug)]
pub struct Corpus {
    pub model: super::Model,
}

/// Define a trait for the CorpusManager with methods to interact with the model.
pub trait CorpusManager {
    /// Predict label of the entry.
    fn predict(&self, buffer: &[u8]) -> Result<String>;
}

impl CorpusManager for Corpus {
    /// Predicts the possible label of the input image.
    fn predict(&self, buffer: &[u8]) -> Result<String> {
        let img = image::load_from_memory(buffer)?;

        // If image well-sized use it as it is, otherwise, resize it.
        let resized = if img.width() == 224 && img.height() == 224 {
            img.to_rgba8()
        } else {
            /*image::load_from_memory(image_processor::resizer::resize(
                buffer,
                Some(IMAGE_WIDTH),
                Some(IMAGE_WIDTH),
            )?)?
            .to_rgba8();*/
            image::imageops::resize(&img, IMAGE_WIDTH, IMAGE_HEIGHT, ::image::imageops::FilterType::Nearest)
        };

        let img_array: Tensor =
            tract_ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
                resized.get_pixel(x as u32, y as u32)[c] as f32
            })
            .into();

        let outputs = self
            .model
            .run(tvec!(img_array.permute_axes(&[0, 2, 3, 1])?.into()))?;

        let best = outputs[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .zip(1..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Ok(RESULT[best.unwrap().1 - 1].to_string())
    }
}

/// Start Corpus model with optimization and return it.
pub fn init() -> Result<super::Model> {
    let model = tract_onnx::onnx()
        .model_for_path("./src/corpus/model.onnx")?
        .with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?
        .with_output_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![1, RESULT.len()]))?
        .into_optimized()?
        .into_runnable()?;

    Ok(model)
}
