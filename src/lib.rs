pub mod client;
pub mod generator_files;
pub mod generator_http;
pub mod image_processing;
pub mod structs;
pub mod worker_files;
pub mod worker_http;

pub use client::DatagoClient;
pub use generator_files::SourceFileConfig;
pub use generator_http::SourceDBConfig;
pub use image_processing::ImageTransformConfig;
pub use structs::{DatagoClientConfig, ImagePayload, LatentPayload, Sample};

use pyo3::prelude::*;

#[pymodule]
fn datago(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatagoClient>()?;
    Ok(())
}
