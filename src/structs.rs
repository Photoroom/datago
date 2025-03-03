use crate::image_processing::ImageTransformConfig;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Db,
    File,
}

fn default_source_type() -> SourceType {
    SourceType::Db
}

#[derive(Deserialize)]
pub struct DatagoClientConfig {
    #[serde(default = "default_source_type")]
    pub source_type: SourceType,

    pub source_config: serde_json::Value,
    pub image_config: Option<ImageTransformConfig>,
    pub limit: usize,
    pub rank: usize,
    pub world_size: usize,
    pub samples_buffer_size: usize,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentPayload {
    #[pyo3(get, set)]
    pub data: Vec<u8>,
    #[pyo3(get, set)]
    pub len: usize,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImagePayload {
    #[pyo3(get, set)]
    pub data: Vec<u8>,
    #[pyo3(get, set)]
    pub original_height: usize, // Good indicator of the image frequency dbResponse at the current resolution
    #[pyo3(get, set)]
    pub original_width: usize,
    #[pyo3(get, set)]
    pub height: usize, // Useful to decode the current payload
    #[pyo3(get, set)]
    pub width: usize,
    #[pyo3(get, set)]
    pub channels: i8,
    #[pyo3(get, set)]
    pub bit_depth: usize,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    #[pyo3(get, set)]
    pub id: String,

    #[pyo3(get, set)]
    pub source: String,

    #[doc(hidden)]
    pub attributes: std::collections::HashMap<String, serde_json::Value>,

    #[pyo3(get, set)]
    pub duplicate_state: i32,

    #[pyo3(get, set)]
    pub image: ImagePayload,

    #[pyo3(get, set)]
    pub masks: std::collections::HashMap<String, ImagePayload>,

    #[pyo3(get, set)]
    pub additional_images: std::collections::HashMap<String, ImagePayload>,

    #[pyo3(get, set)]
    pub latents: std::collections::HashMap<String, LatentPayload>,

    #[pyo3(get, set)]
    pub coca_embedding: Vec<f32>,

    #[pyo3(get, set)]
    pub tags: Vec<String>,
}

#[pymethods]
impl Sample {
    #[getter]
    pub fn attributes(&self) -> String {
        serde_json::to_string(&self.attributes).unwrap_or("".to_string())
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct CocaEmbedding {
    pub vector: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UrlLatent {
    pub file_direct_url: String,
    pub latent_type: String,
    pub is_mask: bool,
}
