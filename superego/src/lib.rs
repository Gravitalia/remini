#![forbid(unsafe_code)]
//#![deny(dead_code, unused_imports, unused_mut, missing_docs)]
//! # superego
//!
//! returns the probability that the message contains:
//! identity hate, insult, neutral (nothing), obscene, serious toxicity,
//! threat and toxicity.

use remini_error::{Error, ErrorType};
use std::path::PathBuf;
use tract_onnx::prelude::*;

type Model = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::Graph<
        tract_onnx::prelude::TypedFact,
        Box<dyn tract_onnx::prelude::TypedOp>,
    >,
>;

const RESULT: [&str; 7] = [
    "identity_hate",
    "insult",
    "neutral",
    "obscene",
    "severe_toxic",
    "threat",
    "toxic",
];

/// Define a structure to manage the Superego model.
#[derive(Debug)]
pub struct Superego {
    /// `Superego` ONNX model.
    model: Model,
}

impl Superego {
    /// Loads the ONNX model from a file path.
    pub fn load(path: PathBuf) -> TractResult<Self> {
        let model = init(path)?;
        Ok(Superego { model })
    }

    /// Predicts the possible label of the input image.
    pub fn predict(
        &self,
        _text: String,
    ) -> Result<(f32, f32, f32, f32, f32, f32, f32), Error> {
        let output_tokenizer = [
            108, 29, 112, 23, 2, 84, 5, 166, 4, 11, 5, 182, 742, 13, 44, 4106,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        .to_vec();

        let input_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, output_tokenizer.len()),
            output_tokenizer.iter().map(|&x| x as f32).collect(),
        )
        .map_err(|error| {
            Error::new(ErrorType::Unspecified, Some(Box::new(error)), None)
        })?
        .into();

        let outputs: Vec<f32> = self
            .model
            .run(tvec!(input_tensor.into()))
            .map_err(|error| {
                Error::new(ErrorType::Algorithms, Some(error.into()), None)
            })?[0]
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
            .copied()
            .collect();

        Ok((
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4],
            outputs[5], outputs[6],
        ))
    }
}

/// Inits ONNX `Corpus` model.
fn init(path: PathBuf) -> TractResult<Model> {
    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .with_output_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec![1, RESULT.len()]),
        )?
        .into_optimized()?
        .into_runnable()?;

    Ok(model)
}
