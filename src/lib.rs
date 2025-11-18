pub mod client;
pub mod generator_files;
pub mod generator_http;
pub mod generator_wds;

pub mod image_processing;
pub mod structs;
pub mod worker_files;
pub mod worker_http;
pub mod worker_wds;

pub use client::{initialize_logging, DatagoClient};
pub use generator_files::SourceFileConfig;
pub use generator_http::SourceDBConfig;
pub use image_processing::ImageTransformConfig;
pub use structs::{DatagoClientConfig, ImagePayload, LatentPayload, Sample};

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn datago(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatagoClient>()?;
    m.add_class::<ImagePayload>()?;
    m.add_function(wrap_pyfunction!(initialize_logging, m)?)?;
    Ok(())
}
