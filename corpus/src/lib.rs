#![forbid(unsafe_code)]
#![deny(dead_code, unused_imports, unused_mut, missing_docs)]
//! # corpus
//!
//! detects whether image contains nude or not.
//! only support: png, jpeg and webp.

use remini_error::{Error, ErrorType};
use std::{cmp::Ordering, path::PathBuf};
use tracing::trace;
use tract_onnx::prelude::*;

type Model = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::Graph<
        tract_onnx::prelude::TypedFact,
        Box<dyn tract_onnx::prelude::TypedOp>,
    >,
>;

const IMAGE_WIDTH: u32 = 224;
const IMAGE_HEIGHT: u32 = 224;
const RESULT: [&str; 2] = ["not_nude", "nude"];

/// Define a structure to manage the Corpus model.
#[derive(Debug)]
pub struct Corpus {
    /// `Corpus` ONNX model.
    model: Model,
}

impl Corpus {
    /// Loads the ONNX model from a file path.
    pub fn load(path: PathBuf) -> TractResult<Self> {
        let model = init(path)?;
        Ok(Corpus { model })
    }

    /// Predicts the possible label of the input image.
    pub fn predict(&self, buffer: &[u8]) -> Result<(f32, u8), Error> {
        let img = image::load_from_memory(buffer).map_err(|error| {
            Error::new(
                ErrorType::Unspecified,
                Some(Box::new(error)),
                Some("while loading image from memory buffer".to_string()),
            )
        })?;

        // If image well-sized use it as it is, otherwise, resize it.
        let resized =
            if img.width() == IMAGE_WIDTH && img.height() == IMAGE_HEIGHT {
                img.to_rgba8()
            } else {
                image::imageops::resize(
                    &img,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                    ::image::imageops::FilterType::Nearest,
                )
            };

        let img_array: Tensor = tract_ndarray::Array::from_shape_fn(
            (1, 3, 224, 224),
            |(_, c, y, x)| resized.get_pixel(x as u32, y as u32)[c] as f32,
        )
        .into();

        let outputs = self
            .model
            .run(tvec!(img_array
                .permute_axes(&[0, 2, 3, 1])
                .map_err(|error| {
                    Error::new(
                        ErrorType::Algorithms,
                        Some(error.into()),
                        Some("cannot permute axes:".to_string()),
                    )
                })?
                .into()))
            .map_err(|error| {
                Error::new(ErrorType::Algorithms, Some(error.into()), None)
            })?;

        let best = outputs[0]
            .to_array_view::<f32>()
            .map_err(|error| {
                Error::new(
                    ErrorType::Algorithms,
                    Some(error.into()),
                    Some(
                        "while transforming data into `ndarray::Array`"
                            .to_string(),
                    ),
                )
            })?
            .iter()
            .cloned()
            .zip(1..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let (confidence, result) = best.unwrap_or_default();
        trace!(
            image_size = img.as_bytes().len(),
            confidence,
            result,
            "Does image contains nudity? {:?}",
            result == 2
        );

        Ok((confidence, result))
    }
}

/// Inits ONNX `Corpus` model.
fn init(path: PathBuf) -> TractResult<Model> {
    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?
        .with_output_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec![1, RESULT.len()]),
        )?
        .into_optimized()?
        .into_runnable()?;

    Ok(model)
}
