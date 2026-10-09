use crate::image_processing::ImageTransformConfig;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::{PyBytes, PyDict, PyList};
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{c_int, c_void, CString};
use std::ptr;
use std::sync::Arc;
use std::thread;
use tokio_util::sync::CancellationToken;

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
pub enum MetadataReceiver {
    Json(kanal::Receiver<serde_json::Value>),
    Tarball(kanal::Receiver<TarballSample>),
}

impl MetadataReceiver {
    pub fn close(&self) {
        match self {
            Self::Json(receiver) => {
                let _ = receiver.close();
            }
            Self::Tarball(receiver) => {
                let _ = receiver.close();
            }
        }
    }
}

#[derive(Debug)]
pub struct DatagoEngine {
    pub samples_rx: kanal::Receiver<Option<Sample>>,
    pub metadata_rx: MetadataReceiver,
    /// Flipped when the engine is torn down (stop, end-of-stream, or interrupt)
    /// so workers can abort in-flight tasks promptly.
    pub cancel: CancellationToken,
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
    #[serde(with = "shared_bytes")]
    pub data: Arc<[u8]>,
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

mod shared_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub fn serialize<S>(data: &Arc<[u8]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        data.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<[u8]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<u8>::deserialize(deserializer).map(Arc::from)
    }
}

#[pyclass(name = "ImagePayload")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonImagePayload {
    inner: ImagePayload,
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
            data: Arc::from(Vec::<u8>::new()),
            original_height: 0,
            original_width: 0,
            height: 0,
            width: 0,
            channels: 0,
            bit_depth: 0,
            is_encoded: false,
        }
    }

    /// Explicit access to bytes returns an owned Python copy. Cloning the
    /// ImagePayload wrapper itself shares the underlying Rust allocation.
    #[getter]
    pub fn data(&self) -> Vec<u8> {
        self.data.as_ref().to_vec()
    }

    #[setter]
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = Arc::from(data);
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
                .call1((self.data.as_ref(),))?;
            let image = pil.call_method1("open", (bytes_io,))?;
            Ok(image.into())
        } else {
            // For raw images, expose the Rust allocation through the buffer
            // protocol so numpy can view it without an intermediate Python bytes
            // copy; PIL then materializes a writable image.
            let numpy = py.import("numpy")?;
            let pil = py.import("PIL.Image")?;
            let owner = Py::new(
                py,
                SharedImageBuffer {
                    data: self.data.clone(),
                },
            )?;

            if self.channels == 1 {
                // Greyscale image - use a 2D shape and create directly
                let shape = (self.height, self.width);
                let np_array = numpy
                    .call_method1(
                        "frombuffer",
                        (owner.bind(py).as_any(), numpy.getattr("uint8")?),
                    )?
                    .call_method1("reshape", (shape,))?;
                let image = pil.call_method1("fromarray", (np_array,))?;
                Ok(image.call_method1("convert", ("L",))?.into())
            } else {
                let shape = (self.height, self.width, self.channels as usize);
                let np_array = numpy
                    .call_method1(
                        "frombuffer",
                        (owner.bind(py).as_any(), numpy.getattr("uint8")?),
                    )?
                    .call_method1("reshape", (shape,))?;
                if self.channels == 4 {
                    let image = pil.call_method1("fromarray", (np_array,))?;
                    Ok(image.call_method1("convert", ("RGBA",))?.into())
                } else {
                    Ok(pil.call_method1("fromarray", (np_array,))?.into())
                }
            }
        }
    }

    /// Get the image as a numpy array. Raw payloads return a read-only zero-copy
    /// view over the shared pixels; encoded payloads are decoded first (not
    /// zero-copy).
    pub fn to_numpy_array(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.is_encoded {
            // For encoded images, we need to decode first
            // This is not zero-copy but necessary for encoded data
            let pil = py.import("PIL.Image")?;
            let bytes_io = py
                .import("io")?
                .getattr("BytesIO")?
                .call1((self.data.as_ref(),))?;
            let image = pil.call_method1("open", (bytes_io,))?;
            let rgb = image.call_method1("convert", ("RGB",))?;
            let np_array = py.import("numpy")?.call_method1("asarray", (rgb,))?;
            Ok(np_array.into())
        } else {
            let owner = Py::new(
                py,
                SharedImageBuffer {
                    data: self.data.clone(),
                },
            )?;
            numpy_from_buffer(py, owner.bind(py).as_any(), self)
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
        let data = slf.borrow().data.clone();
        let owner = Py::new(py, SharedImageBuffer { data: data.clone() })?;
        unsafe { fill_readonly_buffer(view, flags, data.as_ref(), owner.into_bound(py).into_any()) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_buffer_format(view) };
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
        if self.inner.channels == -1 {
            return self
                .inner
                .to_pil_image(py)?
                .bind(py)
                .getattr("mode")?
                .extract();
        }
        Ok(image_mode(self.inner.channels, self.inner.bit_depth).to_owned())
    }

    /// DynamicImage expands palette formats to pixel channels when decoding.
    /// Encoded images are rare here; consult PIL for their exact palette state.
    #[getter]
    pub fn palette(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.inner.channels == -1 {
            let image = self.inner.to_pil_image(py)?;
            let palette = image.bind(py).getattr("palette")?;
            if !palette.is_none() {
                return Ok(Some(palette.unbind()));
            }
        }
        Ok(None)
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
    /// encoded payloads are decoded first (not zero-copy).
    pub fn to_numpy_array(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let payload = &slf.borrow().inner;
        if payload.is_encoded {
            return payload.to_numpy_array(py);
        }
        let owner = Py::new(
            py,
            SharedImageBuffer {
                data: payload.data.clone(),
            },
        )?;
        numpy_from_buffer(py, owner.bind(py).as_any(), payload)
    }

    /// NumPy's frombuffer retains a memoryview of this object as its base,
    /// keeping the immutable Arc-backed allocation alive after the sample drops.
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
        let data = slf.borrow().inner.data.clone();
        let owner = Py::new(py, SharedImageBuffer { data: data.clone() })?;
        unsafe { fill_readonly_buffer(view, flags, data.as_ref(), owner.into_bound(py).into_any()) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_buffer_format(view) };
    }
}

/// Internal buffer owner used when callers start from the Rust ImagePayload
/// pyclass instead of PythonImagePayload. Its Arc is retained by NumPy's base.
/// The wrapped bytes are raw pixels or an encoded stream, mirroring the payload.
#[pyclass]
struct SharedImageBuffer {
    data: Arc<[u8]>,
}

#[pymethods]
impl SharedImageBuffer {
    // Exports whatever bytes were wrapped (raw pixels or an encoded stream).
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let owner = slf.borrow();
        unsafe { fill_readonly_buffer(view, flags, owner.data.as_ref(), slf.into_any()) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_buffer_format(view) };
    }
}

fn image_mode(channels: i8, bit_depth: usize) -> &'static str {
    match (channels, bit_depth) {
        (1, depth) if depth > 8 => "I;16",
        (1, _) => "L",
        (3, _) => "RGB",
        (4, _) => "RGBA",
        _ => "RGB",
    }
}

fn numpy_from_buffer(
    py: Python<'_>,
    owner: &Bound<'_, PyAny>,
    payload: &ImagePayload,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let shape = (
        payload.height,
        payload.width,
        payload.channels.max(0) as usize,
    );
    let array = numpy
        .call_method1("frombuffer", (owner, numpy.getattr("uint8")?))?
        .call_method1("reshape", (shape,))?;
    Ok(array.into())
}

/// Export a read-only one-dimensional byte buffer. The Python owner is retained
/// in Py_buffer.obj, so its Arc-backed data stays alive for every outstanding view.
unsafe fn fill_readonly_buffer(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    data: &[u8],
    owner: Bound<'_, PyAny>,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("buffer view is null"));
    }
    if flags & ffi::PyBUF_WRITABLE == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err("image payload buffer is read-only"));
    }
    unsafe {
        let buffer = &mut *view;
        buffer.obj = owner.into_ptr();
        buffer.buf = data.as_ptr() as *mut c_void;
        buffer.len = data.len() as isize;
        buffer.readonly = 1;
        buffer.itemsize = 1;
        buffer.format = if flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT {
            CString::new("B").unwrap().into_raw()
        } else {
            ptr::null_mut()
        };
        buffer.ndim = 1;
        buffer.shape = if flags & ffi::PyBUF_ND == ffi::PyBUF_ND {
            ptr::addr_of_mut!((*view).len)
        } else {
            ptr::null_mut()
        };
        buffer.strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
            ptr::addr_of_mut!((*view).itemsize)
        } else {
            ptr::null_mut()
        };
        buffer.suboffsets = ptr::null_mut();
        buffer.internal = ptr::null_mut();
    }
    Ok(())
}

unsafe fn release_buffer_format(view: *mut ffi::Py_buffer) {
    if !view.is_null() {
        let format = unsafe { (*view).format };
        if !format.is_null() {
            drop(unsafe { CString::from_raw(format) });
        }
    }
}

/// Helper function to convert ImagePayload to PythonImagePayload
pub fn to_python_image_payload(payload: ImagePayload) -> PythonImagePayload {
    PythonImagePayload::new(payload)
}

/// Helper function to convert a HashMap of ImagePayload to HashMap of PythonImagePayload
pub fn to_python_image_payload_map(
    map: HashMap<String, ImagePayload>,
) -> HashMap<String, PythonImagePayload> {
    map.into_iter()
        .map(|(k, v)| (k, PythonImagePayload::new(v)))
        .collect()
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
    pub image: PythonImagePayload,

    #[pyo3(get, set)]
    pub masks: HashMap<String, PythonImagePayload>,

    #[pyo3(get, set)]
    pub additional_images: HashMap<String, PythonImagePayload>,

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

pub fn sample_to_python_types(sample: Sample, py: Python<'_>) -> Option<Py<PyAny>> {
    // Convert the sample to a Python dict with PIL images
    let sample_dict = PyDict::new(py);

    // Add basic fields
    sample_dict.set_item("id", sample.id).unwrap();
    sample_dict.set_item("source", sample.source).unwrap();
    sample_dict
        .set_item("duplicate_state", sample.duplicate_state)
        .unwrap();

    // Convert attributes to python dict
    let attributes_dict = PyDict::new(py);
    for (key, value) in sample.attributes {
        // If value is a string it can be passed as is, else we pass the json-encoded version
        let value_serialized: String = if value.is_string() {
            value.to_string()
        } else {
            serde_json::to_string(&value).unwrap_or("{}".to_string())
        };
        attributes_dict.set_item(key, value_serialized).unwrap();
    }
    sample_dict.set_item("attributes", attributes_dict).unwrap();

    // Convert tags
    let tags_list = PyList::new(py, &sample.tags).unwrap();
    sample_dict.set_item("tags", tags_list).unwrap();

    // Convert coca_embedding
    let coca_array = PyList::new(py, &sample.coca_embedding).unwrap();
    sample_dict.set_item("coca_embedding", coca_array).unwrap();

    // Convert latents
    let latents_dict = PyDict::new(py);
    for (key, latent) in sample.latents {
        let latent_dict = PyDict::new(py);
        latent_dict
            .set_item("data", PyBytes::new(py, &latent.data))
            .unwrap();
        latent_dict.set_item("len", latent.len).unwrap();
        latents_dict.set_item(key, latent_dict).unwrap();
    }
    sample_dict.set_item("latents", latents_dict).unwrap();

    // Convert images to PIL images
    let image_pil = sample.image.to_pil_image(py).unwrap();
    sample_dict.set_item("image", image_pil).unwrap();

    // Convert masks to PIL images
    let masks_dict = PyDict::new(py);
    for (key, mask) in sample.masks {
        let mask_pil = mask.to_pil_image(py).unwrap();
        masks_dict.set_item(key, mask_pil).unwrap();
    }
    sample_dict.set_item("masks", masks_dict).unwrap();

    // Convert additional images to PIL images
    let additional_images_dict = PyDict::new(py);
    for (key, additional_image) in sample.additional_images {
        let additional_image_pil = additional_image.to_pil_image(py).unwrap();
        additional_images_dict
            .set_item(key, additional_image_pil)
            .unwrap();
    }
    sample_dict
        .set_item("additional_images", additional_images_dict)
        .unwrap();

    Some(sample_dict.into())
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
            data: Arc::from(vec![255, 0, 128]),
            original_height: 100,
            original_width: 100,
            height: 50,
            width: 50,
            channels: 3,
            bit_depth: 8,
            is_encoded: false,
        };

        assert_eq!(payload.original_height, 100);
        assert_eq!(payload.original_width, 100);
        assert_eq!(payload.height, 50);
        assert_eq!(payload.width, 50);
        assert_eq!(payload.channels, 3);
        assert_eq!(payload.bit_depth, 8);
        assert_eq!(payload.data.len(), 3);
        assert!(!payload.is_encoded);
    }

    #[test]
    fn test_image_payload_clones_share_pixels_and_serde_keeps_byte_array_format() {
        let payload = ImagePayload {
            data: Arc::from(vec![1, 2, 3]),
            original_height: 1,
            original_width: 1,
            height: 1,
            width: 1,
            channels: 3,
            bit_depth: 8,
            is_encoded: false,
        };
        let clone = payload.clone();
        assert!(Arc::ptr_eq(&payload.data, &clone.data));

        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(json["data"], serde_json::json!([1, 2, 3]));
        let restored: ImagePayload = serde_json::from_value(json).unwrap();
        assert_eq!(restored.data.as_ref(), &[1, 2, 3]);
        assert!(!Arc::ptr_eq(&payload.data, &restored.data));
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
            image: to_python_image_payload(ImagePayload {
                data: Arc::from(Vec::<u8>::new()),
                original_height: 100,
                original_width: 100,
                height: 100,
                width: 100,
                channels: 3,
                bit_depth: 8,
                is_encoded: false,
            }),
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
            image: to_python_image_payload(ImagePayload {
                data: Arc::from(Vec::<u8>::new()),
                original_height: 100,
                original_width: 100,
                height: 100,
                width: 100,
                channels: 3,
                bit_depth: 8,
                is_encoded: false,
            }),
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
