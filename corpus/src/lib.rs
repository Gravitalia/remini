#![forbid(unsafe_code)]
//! # corpus
//!
//! `corpus` detects probability of nudity in images.
//! Only support:
//! - PNG,
//! - JPEG,
//! - WebP.
//!
//! It returns a probability (between 0 and 1). 1 means model is 100% sure image
//! is a nude.
//!
//! **Warning** models are statistics. Despite high confidence, they can be
//! wrong.

use remini_error::{Error, Result};
use tract_onnx::prelude::*;

use std::fmt::Debug;
use std::path::Path;

type Model =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// Models are trained with TF and uses NHWC format.
const NHWC: [u64; 4] = [1, 224, 224, 3];
const NCHW: (usize, usize, usize, usize) = (1, 3, 224, 224);
const IMAGE_WIDTH: u32 = 224;
const IMAGE_HEIGHT: u32 = 224;

/// In-memory `corpus` ONNX model.
#[derive(Debug)]
pub struct Corpus(Model);

impl Corpus {
    /// Inits ONNX model with optimizations.
    fn init(path: impl AsRef<Path>) -> TractResult<Model> {
        tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact(NHWC).into())?
            .with_output_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec![1, 1]),
            )?
            .into_optimized()?
            .into_runnable()
    }

    /// Loads the ONNX model from a file path.
    pub fn load(path: impl AsRef<Path> + Debug) -> TractResult<Self> {
        tracing::trace!(?path, "corpus model is loading...");
        let model = Self::init(&path)?;
        tracing::trace!(?path, "corpus model loaded");
        Ok(Corpus(model))
    }

    /// Resize image to model's shape.
    /// If image is well-sized, return it as RGB.
    #[inline]
    #[must_use]
    fn resize(
        img: &image::DynamicImage,
    ) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
        tracing::trace!(
            width = img.width(),
            height = img.height(),
            "image resizing"
        );
        if img.width() == IMAGE_WIDTH && img.height() == IMAGE_HEIGHT {
            img.to_rgba8()
        } else {
            image::imageops::resize(
                img,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                ::image::imageops::FilterType::Nearest,
            )
        }
    }

    /// Predicts the probabilty of nudity in image.
    pub fn predict(&self, buffer: impl AsRef<[u8]>) -> Result<f32> {
        let img = image::load_from_memory(buffer.as_ref())?;
        let resized = Self::resize(&img);

        let img_array: Tensor =
            tract_ndarray::Array::from_shape_fn(NCHW, |(_, c, y, x)| {
                resized.get_pixel(x as u32, y as u32)[c] as f32
            })
            .into();

        let outputs = self.0.run(tvec!(
            img_array
                .permute_axes(&[0, 2, 3, 1])
                .map_err(|_| { Error::Permutation })?
                .into()
        ))?;

        Ok(*outputs[0]
            .to_array_view::<f32>()?
            .first()
            .ok_or(Error::Execution)?)
    }
}
