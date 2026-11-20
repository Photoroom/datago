//! Image data loading with native workers and a Python consumer.
//!
//! `source` is the seam between front ends and the back end. A front end is a
//! `Source`: its generator enumerates work items and its worker prepares one
//! sample per item. The back end wires any source into the same pipeline:
//! bounded queues, a feeder thread and a pool of loading tasks. `worker_utils`
//! alone publishes samples, enforces the successful-sample limit, and observes
//! cancellation at every asynchronous wait. `pipeline` records how a pass ended.
//! `client` owns engine lifecycle and Python signal handling.
//!
//! Image pixels live in immutable Arc allocations (`image_payload`). Their
//! validated layout drives metadata and conversion. `python_buffer` retains an
//! allocation owner for every read-only export; `python_conversion` builds Python
//! sample dictionaries and propagates Python exceptions. `structs` retains the
//! original public import paths through re-exports.

pub mod client;
mod config;
mod error;
pub mod generator_files;
pub mod generator_http;
pub mod generator_wds;
mod http_client;
mod image_payload;
mod pipeline;
mod python_buffer;
mod python_conversion;
mod source;

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
