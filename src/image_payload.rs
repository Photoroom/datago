//! Shared image pixels, metadata, and Python image access.

use crate::python_buffer::{export_bytes, SharedImageBuffer};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use serde::{Deserialize, Serialize};
use std::ffi::c_int;
use std::io::Cursor;
use std::sync::Arc;

#[pyclass]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImagePayload {
    pub data: Arc<Vec<u8>>,
    #[pyo3(get, set)]
    pub original_height: usize,
    #[pyo3(get, set)]
    pub original_width: usize,
    #[pyo3(get, set)]
    pub height: usize,
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

#[pymethods]
impl ImagePayload {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Explicit access to bytes returns an owned Python copy. Cloning the
    /// ImagePayload wrapper itself shares the underlying Rust allocation.
    #[getter]
    pub fn data(&self) -> Vec<u8> {
        self.data.as_ref().to_vec()
    }

    #[setter]
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = Arc::new(data);
    }

    /// Convert this ImagePayload to a PIL Image directly in Rust
    /// This avoids the need for Python-side conversion and reduces data copying
    pub fn to_pil_image(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.is_encoded {
            // Decode outside the GIL, then use the same validated conversion as
            // raw images. PIL must not silently narrow an encoded RGB16 image.
            self.decode(py)?.to_pil_image(py)
        } else {
            let layout = self.raw_layout()?;
            layout.pil_mode()?;
            let array = numpy_from_buffer(py, self.data.clone(), layout)?;
            let pil = py.import("PIL.Image")?;
            let image = pil.call_method1("fromarray", (array,))?;
            // Pillow shares some modes (including I;16). Copy only when needed
            // to return writable PIL-owned storage, independent of Rust pixels.
            if image.getattr("readonly")?.is_truthy()? {
                Ok(image.call_method1("copy", ())?.into())
            } else {
                Ok(image.into())
            }
        }
    }

    /// Get the image as a numpy array. Raw payloads return a read-only zero-copy
    /// view over the shared pixels; encoded payloads are decoded first (not
    /// zero-copy). The dtype follows the source (`uint8`/`uint16`/`float32`);
    /// single-channel images are shaped `(H, W)`, multi-channel `(H, W, C)`.
    pub fn to_numpy_array(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.is_encoded {
            // Native decoding preserves alpha and wider channel types.
            self.decode(py)?.to_numpy_array(py)
        } else {
            numpy_from_buffer(py, self.data.clone(), self.raw_layout()?)
        }
    }

    // Buffer-protocol export: exposes the stored bytes read-only. For encoded
    // payloads this is the compressed stream, not decoded pixels; use
    // `to_numpy_array()` / `to_pil_image()` to decode.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        // Retain the Arc rather than the mutable wrapper, so an outstanding view
        // stays valid even if `data` is reassigned through the setter afterwards.
        let py = slf.py();
        let data = slf
            .try_borrow()
            .map_err(|error| PyBufferError::new_err(error.to_string()))?
            .data
            .clone();
        // SAFETY: PyO3 supplies CPython's writable buffer descriptor.
        unsafe { export_bytes(py, data, view, flags) }
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

    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width
    }

    #[getter]
    pub fn height(&self) -> usize {
        self.inner.height
    }

    #[getter]
    pub fn size(&self) -> (usize, usize) {
        (self.inner.width, self.inner.height)
    }

    #[getter]
    pub fn mode(&self, py: Python<'_>) -> PyResult<String> {
        let layout = if self.inner.is_encoded {
            self.inner.encoded_layout(py)?
        } else {
            self.inner.raw_layout()?
        };
        Ok(layout.pil_mode()?.to_owned())
    }

    /// Native decoding expands palette formats to pixel channels before storage.
    #[getter]
    pub fn palette(&self) -> Option<Py<PyAny>> {
        None
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

    /// Convert to numpy array. Raw payloads return a read-only zero-copy view;
    /// encoded payloads are decoded first (not zero-copy). The dtype follows the
    /// source (`uint8`/`uint16`/`float32`); single-channel images are shaped
    /// `(H, W)`, multi-channel `(H, W, C)`.
    pub fn to_numpy_array(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.to_numpy_array(py)
    }

    // Buffer-protocol export: exposes the stored bytes read-only. For encoded
    // payloads this is the compressed stream, not decoded pixels; use
    // `to_numpy_array()` / `to_pil_image()` to decode.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        // Retain the Arc via a dedicated owner so the view lifetime is tied to
        // the pixels, independent of this wrapper object.
        let py = slf.py();
        let data = slf
            .try_borrow()
            .map_err(|error| PyBufferError::new_err(error.to_string()))?
            .inner
            .data
            .clone();
        // SAFETY: PyO3 supplies CPython's writable buffer descriptor.
        unsafe { export_bytes(py, data, view, flags) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SampleType {
    U8,
    U16,
    F32,
}

impl SampleType {
    fn from_depth(depth: usize) -> PyResult<Self> {
        match depth {
            8 => Ok(Self::U8),
            16 => Ok(Self::U16),
            32 => Ok(Self::F32),
            _ => Err(PyValueError::new_err(format!(
                "unsupported bit depth: {depth}"
            ))),
        }
    }

    fn dtype(self, numpy: &NumpyApi) -> &Py<PyAny> {
        match self {
            Self::U8 => &numpy.uint8,
            Self::U16 => &numpy.uint16,
            Self::F32 => &numpy.float32,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Validated interpretation of the public metadata. Deriving this on access
/// preserves the existing serialization format and Python metadata setters.
#[derive(Clone, Copy)]
struct PixelLayout {
    width: usize,
    height: usize,
    channels: usize,
    sample_type: SampleType,
}

impl PixelLayout {
    fn pil_mode(self) -> PyResult<&'static str> {
        Ok(match (self.channels, self.sample_type) {
            (1, SampleType::U8) => "L",
            (1, SampleType::U16) => "I;16",
            (1, SampleType::F32) => "F",
            (2, SampleType::U8) => "LA",
            (3, SampleType::U8) => "RGB",
            (4, SampleType::U8) => "RGBA",
            (2..=4, _) => {
                return Err(PyValueError::new_err(
                    "PIL cannot represent multi-channel images with bit_depth > 8; \
                 use image_format=\"numpy\" to preserve precision, or set \
                 image_config.image_to_rgb8=true to convert to RGB8",
                ))
            }
            _ => unreachable!("validated channel count"),
        })
    }
}

impl ImagePayload {
    /// Exact sentinel used for a sample with no image, distinct from bad pixels.
    pub(crate) fn is_absent(&self) -> bool {
        !self.is_encoded
            && self.data.is_empty()
            && self.width == 0
            && self.height == 0
            && self.original_width == 0
            && self.original_height == 0
            && self.channels == 0
            && self.bit_depth == 0
    }

    fn raw_layout(&self) -> PyResult<PixelLayout> {
        if self.is_encoded {
            return Err(PyValueError::new_err(
                "expected raw pixels, got encoded bytes",
            ));
        }
        let sample_type = SampleType::from_depth(self.bit_depth)?;
        if !(1..=4).contains(&self.channels) {
            return Err(PyValueError::new_err(format!(
                "invalid raw image channel count: {}",
                self.channels
            )));
        }
        let layout = PixelLayout {
            width: self.width,
            height: self.height,
            channels: self.channels as usize,
            sample_type,
        };
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|size| size.checked_mul(layout.channels))
            .and_then(|size| size.checked_mul(sample_type.bytes()));
        if expected != Some(self.data.len()) {
            return Err(PyValueError::new_err(
                "image dimensions and dtype do not match the pixel buffer length",
            ));
        }
        Ok(layout)
    }

    fn encoded_layout(&self, py: Python<'_>) -> PyResult<PixelLayout> {
        let (dimensions, color) = py
            .detach(|| {
                crate::image_processing::decode_image(|| {
                    use image::ImageDecoder;
                    let decoder = image::ImageReader::new(Cursor::new(self.data.as_slice()))
                        .with_guessed_format()?
                        .into_decoder()?;
                    Ok((decoder.dimensions(), decoder.color_type()))
                })
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PixelLayout {
            width: dimensions.0 as usize,
            height: dimensions.1 as usize,
            channels: color.channel_count() as usize,
            sample_type: SampleType::from_depth(
                (color.bits_per_pixel() / u16::from(color.channel_count())) as usize,
            )?,
        })
    }

    fn decode(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            crate::image_processing::decode_image(|| image::load_from_memory(&self.data))
                .map(Self::from_decoded_image)
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn from_decoded_image(image: image::DynamicImage) -> Self {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let color = image.color();
        Self {
            width,
            height,
            original_width: width,
            original_height: height,
            channels: color.channel_count() as i8,
            bit_depth: (color.bits_per_pixel() / color.channel_count() as u16) as usize,
            is_encoded: false,
            data: Arc::new(image.into_bytes()),
        }
    }
}

/// Python-aware initialization avoids holding the interpreter while waiting for
/// another thread's NumPy import. Failed imports leave the cache uninitialized.
struct NumpyApi {
    frombuffer: Py<PyAny>,
    uint8: Py<PyAny>,
    uint16: Py<PyAny>,
    float32: Py<PyAny>,
}

static NUMPY_API: PyOnceLock<NumpyApi> = PyOnceLock::new();

fn numpy_from_buffer(
    py: Python<'_>,
    data: Arc<Vec<u8>>,
    layout: PixelLayout,
) -> PyResult<Py<PyAny>> {
    let owner = Bound::new(py, SharedImageBuffer { data })?;
    let numpy = NUMPY_API.get_or_try_init(py, || -> PyResult<_> {
        let module = py.import("numpy")?;
        Ok(NumpyApi {
            frombuffer: module.getattr("frombuffer")?.unbind(),
            uint8: module.getattr("uint8")?.unbind(),
            uint16: module.getattr("uint16")?.unbind(),
            float32: module.getattr("float32")?.unbind(),
        })
    })?;
    let array = numpy
        .frombuffer
        .bind(py)
        .call1((owner, layout.sample_type.dtype(numpy)))?;
    let array = if layout.channels == 1 {
        array.call_method1("reshape", ((layout.height, layout.width),))?
    } else {
        array.call_method1("reshape", ((layout.height, layout.width, layout.channels),))?
    };
    Ok(array.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_exports_report_mutable_borrow_conflicts() {
        Python::initialize();
        Python::attach(|py| {
            let memoryview = py
                .import("builtins")
                .unwrap()
                .getattr("memoryview")
                .unwrap();
            let payload = Bound::new(py, ImagePayload::default()).unwrap();
            let borrow = payload.borrow_mut();
            let error = memoryview.call1((&payload,)).unwrap_err();
            assert!(error.is_instance_of::<PyBufferError>(py));
            drop(borrow);
            assert!(memoryview.call1((&payload,)).is_ok());

            let wrapper = Bound::new(py, PythonImagePayload::new(ImagePayload::default())).unwrap();
            let borrow = wrapper.borrow_mut();
            let error = memoryview.call1((&wrapper,)).unwrap_err();
            assert!(error.is_instance_of::<PyBufferError>(py));
            drop(borrow);
            assert!(memoryview.call1((&wrapper,)).is_ok());
        });
    }

    #[test]
    fn only_the_exact_default_payload_is_absent() {
        assert!(ImagePayload::default().is_absent());
        assert!(!ImagePayload {
            data: Arc::new(vec![1]),
            ..Default::default()
        }
        .is_absent());
        assert!(!ImagePayload {
            is_encoded: true,
            ..Default::default()
        }
        .is_absent());
        assert!(!ImagePayload {
            bit_depth: 8,
            ..Default::default()
        }
        .is_absent());
        assert!(!ImagePayload {
            original_width: 1,
            ..Default::default()
        }
        .is_absent());
    }
}
