//! Client configuration and Python output representation.

use crate::image_processing::ImageTransformConfig;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    /// PIL image (materializes the pixels into PIL's own storage).
    #[default]
    Pil,
    /// Read-only, zero-copy ndarray view over the Rust-owned pixels.
    Numpy,
}

impl<'de> Deserialize<'de> for ImageFormat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)
            .map_err(|error| serde::de::Error::custom(format!("image_format: {error}")))?;
        match value.to_ascii_lowercase().as_str() {
            "pil" => Ok(Self::Pil),
            "numpy" => Ok(Self::Numpy),
            _ => Err(serde::de::Error::custom(format!(
                "image_format must be 'pil' or 'numpy', got {value:?}"
            ))),
        }
    }
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

impl DatagoClientConfig {
    pub(crate) fn parse(input: &str) -> Result<Self, String> {
        // Collect unused top-level keys without rejecting existing configurations.
        #[derive(Deserialize)]
        struct Parsed {
            #[serde(flatten)]
            config: DatagoClientConfig,
            #[serde(flatten)]
            unknown: BTreeMap<String, serde_json::Value>,
        }
        let parsed: Parsed = serde_json::from_str(input).map_err(|error| error.to_string())?;
        for key in parsed.unknown.keys() {
            log::warn!("Unknown client configuration field {key:?} is ignored");
        }
        let config = parsed.config;
        if config.samples_buffer_size == 0 {
            return Err("samples_buffer_size must be greater than 0".into());
        }
        if config.limit == 0 {
            return Err("limit must be greater than 0".into());
        }
        let world_size = config
            .source_config
            .get("world_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1);
        let rank = config
            .source_config
            .get("rank")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if world_size == 0 {
            return Err("world_size must be greater than 0".into());
        }
        if rank >= world_size {
            return Err("rank must be less than world_size".into());
        }
        Ok(config)
    }
}
