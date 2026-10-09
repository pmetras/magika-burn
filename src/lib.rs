//! Google's [Magika](https://github.com/google/magika) content-type detection
//! model in pure Rust: its forward pass is written out for f32, with the weights
//! embedded in the binary. Works natively and on `wasm32-unknown-unknown`, where
//! the `wasm` module provides JavaScript bindings.
//!
//! ```
//! use magika_burn::{ContentType, Magika};
//!
//! let magika = Magika::new();
//! let prediction = magika.identify_bytes(b"fn main() {\n    println!(\"Hello, world!\");\n}\n");
//! println!("{} ({})", prediction.output, prediction.info().mime_type);
//!
//! assert_eq!(magika.identify_bytes(b"").output, ContentType::Empty);
//! assert_eq!(ContentType::Pdf.info().mime_type, "application/pdf");
//! ```

mod config;
mod content_type;
mod features;
mod model;
mod prediction;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm;

use std::{
    fmt,
    io::{self, Read, Seek},
};

pub use crate::{
    config::{INPUT_SIZE, NUM_LABELS, PADDING_TOKEN},
    content_type::{ContentType, ContentTypeInfo},
    features::Features,
    prediction::{OverwriteReason, Prediction, PredictionMode},
};

/// Runs the Rust examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// Invalid model input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The number of tokens is not a positive multiple of [`INPUT_SIZE`].
    InvalidLength(usize),
    /// A token is outside `0..=PADDING_TOKEN`.
    InvalidToken { index: usize, token: i32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength(len) => write!(
                f,
                "expected a positive multiple of {INPUT_SIZE} tokens, got {len}"
            ),
            Self::InvalidToken { index, token } => write!(
                f,
                "token {token} at index {index} is outside 0..={PADDING_TOKEN}"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// The Magika model. Its weights are embedded in the binary, so creating one is
/// free.
pub struct Magika {
    prediction_mode: PredictionMode,
}

impl Magika {
    /// Creates the model, using [`PredictionMode::HighConfidence`].
    pub fn new() -> Self {
        Self {
            prediction_mode: PredictionMode::default(),
        }
    }

    /// Sets the prediction mode used by the `identify_*` methods.
    pub fn with_prediction_mode(mut self, prediction_mode: PredictionMode) -> Self {
        self.prediction_mode = prediction_mode;
        self
    }

    /// Returns the prediction mode used by the `identify_*` methods.
    pub fn prediction_mode(&self) -> PredictionMode {
        self.prediction_mode
    }

    /// Identifies the content type of in-memory content.
    pub fn identify_bytes(&self, content: &[u8]) -> Prediction {
        self.identify(Features::extract(content))
    }

    /// Identifies the content type of seekable content, e.g. a [`std::fs::File`].
    ///
    /// Reads at most the first and the last 4 KiB, so this is cheap for large
    /// files. See [`Features::extract_reader`].
    pub fn identify_reader(&self, reader: impl Read + Seek) -> io::Result<Prediction> {
        Ok(self.identify(Features::extract_reader(reader)?))
    }

    pub(crate) fn identify(&self, features: Features) -> Prediction {
        match features {
            Features::Ruled(output) => Prediction::ruled(output),
            Features::Tokens(tokens) => {
                let tokens = tokens
                    .as_slice()
                    .try_into()
                    .expect("extracted features have INPUT_SIZE tokens");
                let mut scores = [0.0; NUM_LABELS];
                model::forward(tokens, &mut scores);
                Prediction::from_scores(&scores, self.prediction_mode)
            }
        }
    }

    /// Scores a batch of samples.
    ///
    /// `features` holds `n` samples of [`INPUT_SIZE`] tokens each, back to back.
    /// A token is a byte value (`0..=255`) or [`PADDING_TOKEN`]. Returns `n` rows
    /// of [`NUM_LABELS`] softmax scores, also back to back.
    pub fn scores(&self, features: &[i32]) -> Result<Vec<f32>, Error> {
        if features.is_empty() || !features.len().is_multiple_of(INPUT_SIZE) {
            return Err(Error::InvalidLength(features.len()));
        }
        if let Some((index, &token)) = features
            .iter()
            .enumerate()
            .find(|(_, token)| !(0..=PADDING_TOKEN).contains(*token))
        {
            return Err(Error::InvalidToken { index, token });
        }

        let mut scores = vec![0.0; features.len() / INPUT_SIZE * NUM_LABELS];
        let (samples, _) = features.as_chunks::<INPUT_SIZE>();
        let (rows, _) = scores.as_chunks_mut::<NUM_LABELS>();
        for (tokens, row) in samples.iter().zip(rows) {
            model::forward(tokens, row);
        }
        Ok(scores)
    }
}

impl Default for Magika {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Magika {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Magika")
            .field("prediction_mode", &self.prediction_mode)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One model can be shared by several threads, which the model's lazily built
    /// tables must not break.
    #[test]
    fn is_shareable_between_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Magika>();

        let magika = Magika::new();
        let expected = magika.identify_bytes(b"#!/bin/sh\necho 'Hello, world!'\n");
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let prediction = magika.identify_bytes(b"#!/bin/sh\necho 'Hello, world!'\n");
                    assert_eq!(prediction, expected);
                });
            }
        });
    }
}
