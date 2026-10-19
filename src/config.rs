//! Client configuration and Python output representation.

use crate::image_processing::ImageTransformConfig;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Db,
    File,
    WebDataset,
    Invalid,
}

fn default_source_type() -> SourceType {
    SourceType::Db
}

/// How image payloads are represented when returned as Python objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    /// PIL image (materializes the pixels into PIL's own storage).
    #[default]
    #[serde(alias = "PIL")]
    Pil,
    /// Read-only, zero-copy ndarray view over the Rust-owned pixels.
    #[serde(alias = "NumPy", alias = "NUMPY")]
    Numpy,
}

#[derive(Deserialize)]
pub struct DatagoClientConfig {
    #[serde(default = "default_source_type")]
    pub source_type: SourceType,

    pub source_config: serde_json::Value,
    pub image_config: Option<ImageTransformConfig>,
    pub limit: usize,
    pub samples_buffer_size: usize,

    /// Output representation for images (`"pil"` by default, `"numpy"` for
    /// zero-copy arrays).
    #[serde(default)]
    pub image_format: ImageFormat,
}
