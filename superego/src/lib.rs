#![forbid(unsafe_code)]
//#![deny(dead_code, unused_imports, unused_mut, missing_docs)]
//! # superego
//!
//! returns the probability that the message contains:
//! identity hate, insult, neutral (nothing), obscene, serious toxicity,
//! threat and toxicity.

use remini_error::{Error, Result};
use std::path::PathBuf;
use tokenizers::tokenizer::Tokenizer;
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
    /// BERT tokenizer.
    tokenizer: Tokenizer,
}

impl Superego {
    /// Loads the ONNX model from a file path.
    pub fn load(model: PathBuf, tokenizer: PathBuf) -> TractResult<Self> {
        let model = init_model(model)?;
        let tokenizer = init_tokenizer(tokenizer)?;

        Ok(Superego { model, tokenizer })
    }

    /// Predicts the possible label of the input image.
    pub fn predict(
        &self,
        text: String,
    ) -> Result<(f32, f32, f32, f32, f32, f32, f32)> {
        let tokenizer = self
            .tokenizer
            .encode(text, true)
            .map_err(|_| Error::Unspecified)?;
        let output_tokenizer = tokenizer.get_ids();

        let input_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, output_tokenizer.len()),
            output_tokenizer.iter().map(|&x| x as f32).collect(),
        )
        .map_err(|_| Error::Unspecified)?
        .into();

        let outputs: Vec<f32> = self.model.run(tvec!(input_tensor.into()))?[0]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .collect();

        Ok((
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4],
            outputs[5], outputs[6],
        ))
    }
}

/// Inits ONNX `Superego` model.
fn init_model(path: PathBuf) -> TractResult<Model> {
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

/// Inits tokenizer.
fn init_tokenizer(path: PathBuf) -> Result<Tokenizer> {
    let tokenizer =
        Tokenizer::from_file(path).map_err(|_| Error::Unspecified)?;

    Ok(tokenizer)
}
