use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

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
    #[pyo3(get, set)]
    pub is_encoded: bool, // Indicates if image is already encoded (JPEG/PNG)
}

#[pyclass(name = "ImagePayload")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonImagePayload {
    inner: ImagePayload,
}

impl AsRef<ImagePayload> for PythonImagePayload {
    fn as_ref(&self) -> &ImagePayload {
        &self.inner
    }
}

impl Default for ImagePayload {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl ImagePayload {
    #[new]
    pub fn new() -> Self {
        ImagePayload {
            data: Vec::new(),
            original_height: 0,
            original_width: 0,
            height: 0,
            width: 0,
            channels: 0,
            bit_depth: 0,
            is_encoded: false,
        }
    }

    /// Convert this ImagePayload to a PIL Image directly in Rust
    /// This avoids the need for Python-side conversion and reduces data copying
    pub fn to_pil_image(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.is_encoded {
            // For encoded images (JPEG, PNG), create PIL image directly from bytes
            let pil = py.import("PIL.Image")?;
            let bytes_io = py
                .import("io")?
                .getattr("BytesIO")?
                .call1((self.data.as_slice(),))?;
            let image = pil.call_method1("open", (bytes_io,))?;
            Ok(image.into())
        } else {
            // For raw images, create numpy array first then convert to PIL
            let numpy = py.import("numpy")?;
            let shape: (usize, usize, usize) = if self.channels == 1 {
                (self.height, self.width, 1)
            } else {
                (self.height, self.width, self.channels as usize)
            };

            let np_array = numpy
                .call_method1(
                    "frombuffer",
                    (self.data.as_slice(), numpy.getattr("uint8")?),
                )?
                .call_method1("reshape", (shape,))?;

            let pil = py.import("PIL.Image")?;
            if self.channels == 1 {
                // Greyscale image - use 2D shape and create directly
                let shape_2d = (self.height, self.width);
                let np_array_2d = numpy
                    .call_method1(
                        "frombuffer",
                        (self.data.as_slice(), numpy.getattr("uint8")?),
                    )?
                    .call_method1("reshape", (shape_2d,))?;
                let image = pil.call_method1("fromarray", (np_array_2d,))?;
                Ok(image.call_method1("convert", ("L",))?.into())
            } else if self.channels == 4 {
                // RGBA image
                let image = pil.call_method1("fromarray", (np_array,))?;
                Ok(image.call_method1("convert", ("RGBA",))?.into())
            } else {
                // RGB image (assuming 3 channels)
                let image = pil.call_method1("fromarray", (np_array,))?;
                Ok(image.into())
            }
        }
    }

    /// Get the image as a numpy array (zero-copy when possible)
    pub fn to_numpy_array(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let numpy = py.import("numpy")?;

        if self.is_encoded {
            // For encoded images, we need to decode first
            // This is not zero-copy but necessary for encoded data
            let pil = py.import("PIL.Image")?;
            let bytes_io = py
                .import("io")?
                .getattr("BytesIO")?
                .call1((self.data.as_slice(),))?;
            let image = pil.call_method1("open", (bytes_io,))?;
            let np_array = image
                .call_method0("convert")?
                .call_method1("RGB", ())?
                .call_method0("to_numpy")?;
            Ok(np_array.into())
        } else {
            // For raw images, create zero-copy numpy array
            let shape: (usize, usize, usize) = if self.channels == 1 {
                (self.height, self.width, 1)
            } else {
                (self.height, self.width, self.channels as usize)
            };

            let np_array = numpy
                .call_method1(
                    "frombuffer",
                    (self.data.as_slice(), numpy.getattr("uint8")?),
                )?
                .call_method1("reshape", (shape,))?;

            Ok(np_array.into())
        }
    }
}

#[pymethods]
impl PythonImagePayload {
    #[new]
    pub fn new(inner: ImagePayload) -> Self {
        PythonImagePayload { inner }
    }

    /// Convert to PIL image (this is the main method that gets called)
    pub fn __call__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.to_pil_image(py)
    }

    /// Get the underlying ImagePayload data
    pub fn get_payload(&self) -> ImagePayload {
        self.inner.clone()
    }

    /// Make it behave like a PIL image by delegating all attribute access
    pub fn __getattr__(&self, attr: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Convert to PIL image and delegate all attribute access
        let pil_image = self.inner.to_pil_image(py)?;
        pil_image.getattr(py, attr)
    }

    /// Convert to PIL image (exposed directly for convenience)
    pub fn to_pil_image(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.to_pil_image(py)
    }

    /// Convert to numpy array (exposed directly for convenience)
    pub fn to_numpy_array(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.to_numpy_array(py)
    }
}
