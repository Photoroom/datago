pub mod client;
mod config;
mod error;
pub mod generator_files;
pub mod generator_http;
pub mod generator_wds;
mod http_client;
mod image_payload;
mod pipeline;
mod python_conversion;

pub mod image_processing;
pub mod structs;
pub mod worker_files;
pub mod worker_http;
mod worker_utils;
pub mod worker_wds;

pub use client::{initialize_logging, live_cleanup_threads, DatagoClient};
pub use error::DatagoError;
pub use generator_files::SourceFileConfig;
pub use generator_http::SourceDBConfig;
pub use image_processing::ImageTransformConfig;
pub use structs::{DatagoClientConfig, LatentPayload, PythonImagePayload, Sample};

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn datago(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatagoClient>()?;
    m.add_class::<PythonImagePayload>()?;
    m.add_function(wrap_pyfunction!(initialize_logging, m)?)?;
    m.add_function(wrap_pyfunction!(live_cleanup_threads, m)?)?;
    Ok(())
}
