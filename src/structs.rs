use crate::image_processing::ImageTransformConfig;
use pyo3::prelude::*;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

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

#[derive(Debug)]
pub struct DatagoEngine {
    pub samples_rx: kanal::Receiver<Option<Sample>>,
    pub feeder: Option<thread::JoinHandle<()>>,
    pub worker: Option<thread::JoinHandle<()>>,
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
    pub attributes: HashMap<String, serde_json::Value>,

    #[pyo3(get, set)]
    pub duplicate_state: i32,

    #[pyo3(get, set)]
    pub image: ImagePayload,

    #[pyo3(get, set)]
    pub masks: HashMap<String, ImagePayload>,

    #[pyo3(get, set)]
    pub additional_images: HashMap<String, ImagePayload>,

    #[pyo3(get, set)]
    pub latents: HashMap<String, LatentPayload>,

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

// We'll share a single connection pool across all worker threads
#[derive(Clone)]
pub struct SharedClient {
    pub client: ClientWithMiddleware,
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

pub fn new_shared_client(max_connections: usize) -> SharedClient {
    let retry_policy = ExponentialBackoff::builder()
        .retry_bounds(
            std::time::Duration::from_millis(100), // min_retry_interval
            std::time::Duration::from_secs(3),
        )
        .build_with_max_retries(3);

    let client = ClientBuilder::new(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap(),
    )
    .with(RetryTransientMiddleware::new_with_policy(retry_policy))
    .build();

    SharedClient {
        client,
        semaphore: Arc::new(tokio::sync::Semaphore::new(max_connections)),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryFile {
    pub filename: String,
    pub buffer: Vec<u8>,
}

pub struct TarballSample {
    pub name: String,
    pub content: Vec<BinaryFile>,
}

impl TarballSample {
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn add(&mut self, file: BinaryFile) {
        self.content.push(file);
    }

    pub fn iter(&self) -> impl Iterator<Item = &BinaryFile> {
        self.content.iter()
    }

    pub fn new(name: String) -> Self {
        TarballSample {
            name,
            content: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_latent_payload_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let payload = LatentPayload {
            data: data.clone(),
            len: data.len(),
        };

        assert_eq!(payload.data, data);
        assert_eq!(payload.len, 5);
    }

    #[test]
    fn test_image_payload_creation() {
        let payload = ImagePayload {
            data: vec![255, 0, 128],
            original_height: 100,
            original_width: 100,
            height: 50,
            width: 50,
            channels: 3,
            bit_depth: 8,
        };

        assert_eq!(payload.original_height, 100);
        assert_eq!(payload.original_width, 100);
        assert_eq!(payload.height, 50);
        assert_eq!(payload.width, 50);
        assert_eq!(payload.channels, 3);
        assert_eq!(payload.bit_depth, 8);
        assert_eq!(payload.data.len(), 3);
    }

    #[test]
    fn test_sample_attributes_json() {
        let mut attributes = HashMap::new();
        attributes.insert(
            "caption".to_string(),
            serde_json::Value::String("test caption".to_string()),
        );
        attributes.insert(
            "score".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.85).unwrap()),
        );

        let sample = Sample {
            id: "test_id".to_string(),
            source: "test_source".to_string(),
            attributes,
            duplicate_state: 0,
            image: ImagePayload {
                data: vec![],
                original_height: 100,
                original_width: 100,
                height: 100,
                width: 100,
                channels: 3,
                bit_depth: 8,
            },
            masks: HashMap::new(),
            additional_images: HashMap::new(),
            latents: HashMap::new(),
            coca_embedding: vec![],
            tags: vec!["tag1".to_string(), "tag2".to_string()],
        };

        let attributes_json = sample.attributes();
        assert!(attributes_json.contains("caption"));
        assert!(attributes_json.contains("test caption"));
        assert!(attributes_json.contains("score"));
        assert!(attributes_json.contains("0.85"));
    }

    #[test]
    fn test_sample_empty_attributes() {
        let sample = Sample {
            id: "test_id".to_string(),
            source: "test_source".to_string(),
            attributes: HashMap::new(),
            duplicate_state: 0,
            image: ImagePayload {
                data: vec![],
                original_height: 100,
                original_width: 100,
                height: 100,
                width: 100,
                channels: 3,
                bit_depth: 8,
            },
            masks: HashMap::new(),
            additional_images: HashMap::new(),
            latents: HashMap::new(),
            coca_embedding: vec![],
            tags: vec![],
        };

        let attributes_json = sample.attributes();
        assert_eq!(attributes_json, "{}");
    }

    #[test]
    fn test_coca_embedding_default() {
        let embedding = CocaEmbedding::default();
        assert!(embedding.vector.is_empty());
    }

    #[test]
    fn test_url_latent_creation() {
        let url_latent = UrlLatent {
            file_direct_url: "https://example.com/image.jpg".to_string(),
            latent_type: "masked_image".to_string(),
            is_mask: false,
        };

        assert_eq!(url_latent.file_direct_url, "https://example.com/image.jpg");
        assert_eq!(url_latent.latent_type, "masked_image");
        assert!(!url_latent.is_mask);
    }

    #[test]
    fn test_shared_client_creation() {
        let client = new_shared_client(10);
        assert_eq!(client.semaphore.available_permits(), 10);
    }

    #[test]
    fn test_binary_file_creation() {
        let file = BinaryFile {
            filename: "test.txt".to_string(),
            buffer: vec![72, 101, 108, 108, 111], // "Hello" in bytes
        };

        assert_eq!(file.filename, "test.txt");
        assert_eq!(file.buffer, vec![72, 101, 108, 108, 111]);
    }

    #[test]
    fn test_tarball_sample_empty() {
        let sample = TarballSample::new("test_sample".to_string());
        assert!(sample.is_empty());
        assert_eq!(sample.name, "test_sample");
        assert_eq!(sample.content.len(), 0);
    }

    #[test]
    fn test_tarball_sample_add_file() {
        let mut sample = TarballSample::new("test_sample".to_string());
        let file = BinaryFile {
            filename: "test.txt".to_string(),
            buffer: vec![1, 2, 3, 4, 5],
        };

        sample.add(file);
        assert!(!sample.is_empty());
        assert_eq!(sample.content.len(), 1);
        assert_eq!(sample.content[0].filename, "test.txt");
    }

    #[test]
    fn test_tarball_sample_iterator() {
        let mut sample = TarballSample::new("test_sample".to_string());

        let file1 = BinaryFile {
            filename: "file1.txt".to_string(),
            buffer: vec![1, 2, 3],
        };
        let file2 = BinaryFile {
            filename: "file2.txt".to_string(),
            buffer: vec![4, 5, 6],
        };

        sample.add(file1);
        sample.add(file2);

        let files: Vec<&BinaryFile> = sample.iter().collect();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].filename, "file1.txt");
        assert_eq!(files[1].filename, "file2.txt");
    }

    #[test]
    fn test_source_type_deserialization() {
        let json_db = r#""db""#;
        let json_file = r#""file""#;
        let json_webdataset = r#""webdataset""#;

        let source_db: SourceType = serde_json::from_str(json_db).unwrap();
        let source_file: SourceType = serde_json::from_str(json_file).unwrap();
        let source_wds: SourceType = serde_json::from_str(json_webdataset).unwrap();

        matches!(source_db, SourceType::Db);
        matches!(source_file, SourceType::File);
        matches!(source_wds, SourceType::WebDataset);
    }

    #[test]
    fn test_datago_client_config_deserialization() {
        let config_json = r#"{
            "source_type": "file",
            "source_config": {"root_path": "/tmp"},
            "limit": 100,
            "samples_buffer_size": 10
        }"#;

        let config: DatagoClientConfig = serde_json::from_str(config_json).unwrap();
        matches!(config.source_type, SourceType::File);
        assert_eq!(config.limit, 100);
        assert_eq!(config.samples_buffer_size, 10);
        assert!(config.image_config.is_none());
    }

    #[test]
    fn test_datago_client_config_default_source_type() {
        let config_json = r#"{
            "source_config": {"root_path": "/tmp"},
            "limit": 100,
            "samples_buffer_size": 10
        }"#;

        let config: DatagoClientConfig = serde_json::from_str(config_json).unwrap();
        matches!(config.source_type, SourceType::Db);
    }
}
