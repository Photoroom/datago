//! Errors which end a streaming pass. Bad individual samples remain skippable.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatagoError {
    /// Invalid client setup or read arguments.
    Configuration(String),
    /// Failed source enumeration, metadata request, or shard download.
    Source(String),
    /// A processing task failed or panicked.
    Worker(String),
    /// No sample arrived before the read deadline; the pass is cancelled.
    Timeout,
}

impl fmt::Display for DatagoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) => write!(f, "invalid configuration: {message}"),
            Self::Source(message) => write!(f, "source failed: {message}"),
            Self::Worker(message) => write!(f, "worker failed: {message}"),
            Self::Timeout => f.write_str("timed out waiting for a sample"),
        }
    }
}

impl std::error::Error for DatagoError {}

impl From<DatagoError> for pyo3::PyErr {
    fn from(error: DatagoError) -> Self {
        use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
        match error {
            DatagoError::Configuration(_) => PyValueError::new_err(error.to_string()),
            DatagoError::Timeout => PyTimeoutError::new_err(error.to_string()),
            _ => PyRuntimeError::new_err(error.to_string()),
        }
    }
}

pub(crate) fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "panic with a non-string payload".to_owned()
    }
}
