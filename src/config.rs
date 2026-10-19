use crate::image_processing::ImageTransformConfig;
use serde::Deserialize;

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

#[derive(Deserialize)]
pub struct DatagoClientConfig {
    #[serde(default = "default_source_type")]
    pub source_type: SourceType,

    pub source_config: serde_json::Value,
    pub image_config: Option<ImageTransformConfig>,
    pub limit: usize,
    pub samples_buffer_size: usize,
}
