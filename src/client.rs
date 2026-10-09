use crate::generator_files;
use crate::generator_http;
use crate::generator_wds;
use crate::image_processing::ARAwareTransform;
use crate::structs::{DatagoClientConfig, ImageFormat, Sample, SourceType};

use crate::structs::sample_to_python_types;
use crate::structs::DatagoEngine;
use log::{error, warn};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
const SIGNAL_INTERVAL: Duration = Duration::from_millis(100);

/// Number of background cleanup threads currently joining native workers whose
/// blocking IO outlived the stop grace period. Exposed so repeated stop/restart
/// cycles cannot silently accumulate threads.
static LIVE_CLEANUP_THREADS: AtomicUsize = AtomicUsize::new(0);

enum ReceiveFailure<E> {
    Timeout,
    Closed,
    Interrupted(E),
}

/// Mutable engine state, kept behind a mutex so Python callers never touch PyO3's
/// per-object borrow flag. `stop()` can therefore always make progress while a
/// reader is waiting, and concurrent calls serialize instead of raising
/// `RuntimeError: Already borrowed`.
struct ClientState {
    engine: Option<DatagoEngine>,
    is_started: bool,
    // After EOS/stop, reads return None rather than implicitly starting a new pass.
    stopped: bool,
    generation: u64,
}

#[pyclass(frozen)]
pub struct DatagoClient {
    source_type: SourceType,
    pub source_config: serde_json::Value,
    pub samples_buffer: usize,
    pub limit: usize,

    // Perf settings
    pub max_connections: usize,

    // Sample processing
    pub image_transform: Option<ARAwareTransform>,
    pub encode_images: bool,
    pub img_to_rgb8: bool,
    pub encode_format: crate::image_processing::EncodeFormat,
    pub jpeg_quality: u8,
    pub image_format: ImageFormat,

    state: Mutex<ClientState>,

    is_valid: bool,
}

/// Number of cleanup threads joining native workers whose blocking IO outlived
/// the stop grace period. Useful from Python to detect thread accumulation.
#[pyfunction]
#[allow(dead_code)] // only referenced through the Python module in the lib target
pub fn live_cleanup_threads() -> usize {
    LIVE_CLEANUP_THREADS.load(Ordering::SeqCst)
}

fn check_config(str_config: &str) -> Option<DatagoClientConfig> {
    match serde_json::from_str::<DatagoClientConfig>(str_config) {
        Ok(config) => {
            if config.samples_buffer_size == 0 {
                error!("Samples buffer size must be greater than 0");
                return None;
            }

            if config.limit == 0 {
                error!("Limit must be greater than 0");
                return None;
            }

            // Check that a distributed config is valid, and error out early if not
            let world_size = config
                .source_config
                .get("world_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            let rank = config
                .source_config
                .get("rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if world_size == 0 {
                error!("World size must be greater than 0");
                return None;
            }

            if rank >= world_size {
                error!("Rank must be less than world size");
                return None;
            }
            Some(config)
        }
        Err(e) => {
            error!("Failed to parse config: {e}");
            None
        }
    }
}

#[pymethods]
impl DatagoClient {
    #[new]
    pub fn new(str_config: String) -> Self {
        match check_config(&str_config) {
            Some(config) => {
                let mut image_transform: Option<ARAwareTransform> = None;
                let mut encode_images = false;
                let mut img_to_rgb8 = false;
                let mut encode_format = crate::image_processing::EncodeFormat::default();
                let mut jpeg_quality = 92u8;

                if let Some(image_config) = config.image_config {
                    if image_config.crop_and_resize {
                        image_transform = Some(image_config.get_ar_aware_transform());
                    }
                    encode_images = image_config.pre_encode_images;
                    img_to_rgb8 = image_config.image_to_rgb8;
                    encode_format = image_config.encode_format;
                    jpeg_quality = image_config.jpeg_quality;
                }

                DatagoClient {
                    source_type: config.source_type,
                    source_config: config.source_config,
                    samples_buffer: config.samples_buffer_size,
                    limit: config.limit,
                    max_connections: 128,
                    image_transform,
                    encode_images,
                    img_to_rgb8,
                    encode_format,
                    jpeg_quality,
                    image_format: config.image_format,
                    state: Mutex::new(ClientState {
                        engine: None,
                        is_started: false,
                        stopped: false,
                        generation: 0,
                    }),
                    is_valid: true,
                }
            }
            None => {
                error!("Failed to parse config");
                DatagoClient {
                    source_type: SourceType::Invalid,
                    source_config: serde_json::Value::Null,
                    samples_buffer: 0,
                    limit: 0,
                    max_connections: 0,
                    image_transform: None,
                    encode_images: false,
                    img_to_rgb8: false,
                    encode_format: crate::image_processing::EncodeFormat::default(),
                    jpeg_quality: 92,
                    image_format: ImageFormat::default(),
                    state: Mutex::new(ClientState {
                        engine: None,
                        is_started: false,
                        stopped: false,
                        generation: 0,
                    }),
                    is_valid: false,
                }
            }
        }
    }

    /// Start streaming. Safe to call again after the stream ends or after `stop()`.
    #[pyo3(name = "start")]
    fn py_start(&self) {
        self.start();
    }

    /// Return the next sample, or `None` once the stream is exhausted or stopped.
    /// Reads do not silently restart: call `start()` again for a new pass.
    #[pyo3(name = "get_sample")]
    fn py_get_sample(&self, py: Python<'_>) -> PyResult<Option<Sample>> {
        self.get_sample_python(py)
    }

    /// Return the next sample using native Python types, with the same semantics
    /// as `get_sample()`.
    #[pyo3(name = "get_sample_auto_convert")]

    fn py_get_sample_auto_convert(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let sample = match self.get_sample_python(py)? {
            Some(sample) => sample,
            None => return Ok(None),
        };
        Ok(sample_to_python_types(sample, py, self.image_format))
    }

    /// Stop streaming and release the worker threads.
    #[pyo3(name = "stop")]
    fn py_stop(&self, py: Python<'_>) {
        self.stop_python(py);
    }
}

impl DatagoClient {
    fn state(&self) -> MutexGuard<'_, ClientState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Explicit start permits an intentional restart after stop/EOS.
    pub fn start(&self) {
        let mut state = self.state();
        self.start_locked(&mut state);
    }

    fn start_locked(&self, state: &mut ClientState) {
        if state.is_started {
            return;
        }
        state.stopped = false;
        if !self.is_valid {
            return;
        }
        initialize_logging(Some("warn".to_string()));
        state.engine = match &self.source_type {
            SourceType::Db => Some(generator_http::orchestrate(self)),
            SourceType::File => Some(generator_files::orchestrate(self)),
            SourceType::WebDataset => {
                warn!("WebDataset source type is new and experimental, use with caution!\nPlease report any issues you encounter to https://github.com/Photoroom/datago/issues.");
                Some(generator_wds::orchestrate(self))
            }
            SourceType::Invalid => {
                error!("Client ill-defined, probably a config error. Cannot start");
                None
            }
        };
        state.is_started = state.engine.is_some();
        if state.is_started {
            state.generation = state.generation.wrapping_add(1);
        }
    }

    /// Clone the channel under a short lock. The receiver is shared; stop() can
    /// close it while Python or Rust waits without this lock held.
    fn sample_receiver(&self) -> Option<(kanal::Receiver<Option<Sample>>, u64)> {
        let mut state = self.state();
        if state.stopped {
            return None;
        }
        if !state.is_started {
            self.start_locked(&mut state);
        }
        state
            .engine
            .as_ref()
            .map(|engine| (engine.samples_rx.clone(), state.generation))
    }

    fn take_engine(&self) -> Option<DatagoEngine> {
        self.take_engine_if(None)
    }

    fn take_engine_for(&self, generation: u64) -> Option<DatagoEngine> {
        self.take_engine_if(Some(generation))
    }

    /// Take ownership of the running engine (only if the optional generation
    /// still matches) and close both pipeline queues so producers observe the
    /// cancellation.
    fn take_engine_if(&self, generation: Option<u64>) -> Option<DatagoEngine> {
        let mut state = self.state();
        if let Some(generation) = generation {
            if state.generation != generation {
                return None;
            }
        }
        state.is_started = false;
        state.stopped = true;
        let engine = state.engine.take();
        if let Some(engine) = &engine {
            let _ = engine.samples_rx.close();
            engine.metadata_rx.close();
            engine.cancel.cancel();
        }
        engine
    }

    /// Blocking sample read used by the Rust API and the CLI. Waits up to
    /// [`TIMEOUT`] and joins the worker threads before returning `None`. Unlike
    /// the Python binding it applies neither a stop grace period nor a background
    /// reaper. After end-of-stream or [`stop`](Self::stop), call
    /// [`start`](Self::start) to begin a new pass.
    pub fn get_sample(&self) -> Option<Sample> {
        let (samples_rx, generation) = self.sample_receiver()?;
        match receive_until::<std::convert::Infallible>(
            &samples_rx,
            Instant::now() + TIMEOUT,
            |timeout| {
                samples_rx
                    .recv_timeout(timeout)
                    .map_err(|error| match error {
                        kanal::ReceiveErrorTimeout::Timeout => ReceiveFailure::Timeout,
                        _ => ReceiveFailure::Closed,
                    })
            },
        ) {
            Ok(Some(sample)) => Some(sample),
            Ok(None) | Err(ReceiveFailure::Timeout | ReceiveFailure::Closed) => {
                if let Some(engine) = self.take_engine_for(generation) {
                    join_engine(engine);
                }
                None
            }
            Err(ReceiveFailure::Interrupted(never)) => match never {},
        }
    }

    pub fn get_sample_auto_convert(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let sample = match self.get_sample() {
            Some(sample) => sample,
            None => return Ok(None),
        };
        Ok(sample_to_python_types(sample, py, self.image_format))
    }

    /// Blocking stop: closes both pipeline queues and joins the worker threads.
    /// The Python binding uses a short grace period plus a background reaper
    /// instead, so it never blocks object teardown on stalled native IO.
    pub fn stop(&self) {
        if let Some(engine) = self.take_engine() {
            join_engine(engine);
        }
    }

    fn get_sample_python(&self, py: Python<'_>) -> PyResult<Option<Sample>> {
        let Some((samples_rx, generation)) = self.sample_receiver() else {
            return Ok(None);
        };
        let result = receive_until::<PyErr>(&samples_rx, Instant::now() + TIMEOUT, |timeout| {
            if let Err(error) = py.check_signals() {
                return Err(ReceiveFailure::Interrupted(error));
            }
            py.detach(|| samples_rx.recv_timeout(timeout))
                .map_err(|error| match error {
                    kanal::ReceiveErrorTimeout::Timeout => ReceiveFailure::Timeout,
                    _ => ReceiveFailure::Closed,
                })
        });
        match result {
            Ok(Some(sample)) => Ok(Some(sample)),
            Ok(None) | Err(ReceiveFailure::Timeout | ReceiveFailure::Closed) => {
                self.stop_python_for(py, generation);
                Ok(None)
            }
            Err(ReceiveFailure::Interrupted(error)) => {
                // Cancel and reap without calling back into Python: the error is
                // already set, and stop_python_for never runs signal checks.
                self.stop_python_for(py, generation);
                Err(error)
            }
        }
    }

    fn stop_python_for(&self, py: Python<'_>, generation: u64) {
        if let Some(engine) = self.take_engine_for(generation) {
            wait_or_reap_engine(py, engine);
        }
    }

    fn stop_python(&self, py: Python<'_>) {
        if let Some(engine) = self.take_engine() {
            wait_or_reap_engine(py, engine);
        }
    }
}

fn receive_until<E>(
    samples_rx: &kanal::Receiver<Option<Sample>>,
    deadline: Instant,
    mut wait: impl FnMut(Duration) -> Result<Option<Sample>, ReceiveFailure<E>>,
) -> Result<Option<Sample>, ReceiveFailure<E>> {
    loop {
        match samples_rx.try_recv() {
            Ok(Some(Some(sample))) => return Ok(Some(sample)),
            Ok(None) => {}
            Ok(Some(None)) | Err(_) => return Ok(None),
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        match wait(remaining.min(SIGNAL_INTERVAL)) {
            Ok(Some(sample)) => return Ok(Some(sample)),
            Ok(None) => return Ok(None),
            Err(ReceiveFailure::Timeout) if Instant::now() < deadline => continue,
            Err(ReceiveFailure::Timeout) => {
                warn!("Timeout waiting for sample after {TIMEOUT:?}");
                return Err(ReceiveFailure::Timeout);
            }
            Err(error) => return Err(error),
        }
    }
}

/// Wait up to a short grace period for the engine to wind down, then hand the
/// join off to a background cleanup thread. The GIL is released while polling
/// and joining so other Python threads keep running.
fn wait_or_reap_engine(py: Python<'_>, engine: DatagoEngine) {
    let deadline = Instant::now() + Duration::from_millis(100);
    while !engine_finished(&engine) && Instant::now() < deadline {
        py.detach(|| std::thread::sleep(Duration::from_millis(2)));
    }
    if engine_finished(&engine) {
        py.detach(|| join_engine(engine));
    } else {
        reap_engine(engine);
    }
}

fn join_engine(mut engine: DatagoEngine) {
    for thread in [&mut engine.feeder, &mut engine.worker] {
        if let Some(handle) = thread.take() {
            if let Err(error) = handle.join() {
                error!("Failed to join datago thread: {:?}", error);
            }
        }
    }
}

/// Join a still-running engine on a detached thread. This is the escape hatch
/// for blocking IO that cannot be cancelled; `LIVE_CLEANUP_THREADS` keeps the
/// number of such abandoned joiners observable.
fn reap_engine(engine: DatagoEngine) {
    let live = LIVE_CLEANUP_THREADS.fetch_add(1, Ordering::SeqCst) + 1;
    warn!(
        "datago: native workers still busy after the stop grace period; joining them \
         on a background cleanup thread (live cleanup threads: {live})"
    );
    let spawned = std::thread::Builder::new()
        .name("datago-cleanup".to_string())
        .spawn(move || {
            join_engine(engine);
            LIVE_CLEANUP_THREADS.fetch_sub(1, Ordering::SeqCst);
        });
    if let Err(error) = spawned {
        // If the OS refuses another thread, dropping JoinHandles detaches them;
        // do not block Python object destruction as a fallback.
        LIVE_CLEANUP_THREADS.fetch_sub(1, Ordering::SeqCst);
        error!("Failed to start datago cleanup reaper: {error}");
    }
}

fn engine_finished(engine: &DatagoEngine) -> bool {
    let feeder_finished = match &engine.feeder {
        Some(thread) => thread.is_finished(),
        None => true,
    };
    let worker_finished = match &engine.worker {
        Some(thread) => thread.is_finished(),
        None => true,
    };
    feeder_finished && worker_finished
}

// Ensure cleanup happens even if stop() wasn't called
impl Drop for DatagoClient {
    fn drop(&mut self) {
        // PyO3 may destroy the object while holding the GIL. Cancel immediately
        // and let an owned reaper join workers without blocking object teardown.
        if let Some(engine) = self.take_engine() {
            if engine_finished(&engine) {
                join_engine(engine);
            } else {
                reap_engine(engine);
            }
        }
    }
}

#[pyfunction(signature = (log_level=None))]
pub fn initialize_logging(log_level: Option<String>) -> bool {
    // Try to initialize logging, return false if it fails, e.g. if this function is called multiple times.
    if let Some(level) = log_level {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
            .try_init()
            .is_ok()
    } else {
        env_logger::try_init().is_ok()
    }
}

// -------- Unit tests --------

mod tests {
    #[cfg(test)]
    use crate::client::DatagoClient;

    #[cfg(test)]
    use crate::structs::DatagoEngine;

    #[cfg(test)]
    use std::collections::HashSet;

    #[cfg(test)]
    use serde_json::json;

    #[cfg(test)]
    use crate::image_processing::ImageTransformConfig;

    #[cfg(test)]
    use crate::structs::PythonImagePayload;

    #[test]
    fn stale_reader_cleanup_cannot_take_a_restarted_engine() {
        let client = DatagoClient::new("{}".to_string());
        let (_samples_tx, samples_rx) = kanal::bounded(1);
        let (_metadata_tx, metadata_rx) = kanal::bounded(1);
        {
            let mut state = client.state();
            state.engine = Some(DatagoEngine {
                samples_rx,
                metadata_rx: crate::structs::MetadataReceiver::Json(metadata_rx),
                cancel: tokio_util::sync::CancellationToken::new(),
                feeder: None,
                worker: None,
            });
            state.generation = 2;
            state.is_started = true;
            state.stopped = false;
        }

        assert!(client.take_engine_for(1).is_none());
        assert!(client.state().is_started);
        assert!(client.take_engine_for(2).is_some());
        assert!(!client.state().is_started);
    }

    #[cfg(test)]
    fn get_test_source() -> String {
        std::env::var("DATAROOM_TEST_SOURCE")
            .expect("DATAROOM_TEST_SOURCE environment variable not set")
    }

    #[cfg(test)]
    fn get_test_config() -> serde_json::Value {
        json!({
            "source_config": {
                "sources": get_test_source(),
                "sources_ne": "",
                "require_images": false,
                "require_embeddings": false,
                "tags": "",
                "tags_ne": "",
                "tags_all": "",
                "tags_ne_all": "",
                "tags_empty": "",
                "has_attributes": "",
                "lacks_attributes": "",
                "has_masks": "",
                "lacks_masks": "",
                "has_latents": "",
                "lacks_latents": "",
                "min_short_edge": 0,
                "max_short_edge": 0,
                "min_pixel_count": -1,
                "max_pixel_count": -1,
                "duplicate_state": -1,
                "attributes": "",
                "random_sampling": false,
                "page_size": 10,
            },
            "limit": 2,
            "rank": 0,
            "world_size": 1,
            "num_threads": 1,
            "max_connections": 1,
            "samples_buffer_size": 1
        })
    }

    #[test]
    fn test_start_stop() {
        let config = get_test_config();
        let client = DatagoClient::new(config.to_string());

        client.start();
        client.stop();
    }

    #[test]
    fn test_no_start() {
        let config = get_test_config();
        let client = DatagoClient::new(config.to_string());
        client.stop();
    }

    #[test]
    fn test_no_stop() {
        let config = get_test_config();
        let client = DatagoClient::new(config.to_string());
        client.start();
    }

    #[test]
    fn test_get_sample() {
        let config = get_test_config();
        let client = DatagoClient::new(config.to_string());
        let sample = client.get_sample();

        assert!(sample.is_some());
        assert!(!sample.unwrap().id.is_empty());
    }

    #[test]
    fn test_limit() {
        let limit = 10; // limit < page_size
        let mut config = get_test_config();
        config["limit"] = json!(limit);
        let client = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());
            assert!(!sample.unwrap().id.is_empty());
        }

        let limit = 100; // limit > page_size
        let mut config = get_test_config();
        config["limit"] = json!(limit);
        let client = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());
            assert!(!sample.unwrap().id.is_empty());
        }
    }

    #[cfg(test)]
    use log::debug;

    #[cfg(test)]
    fn check_image(img: &PythonImagePayload) {
        let payload = img.get_payload();
        assert!(!payload.data.is_empty());

        if payload.channels > 0 {
            // Raw image
            assert!(payload.channels == 3 || payload.channels == 1);
            assert!(payload.width > 0);
            assert!(payload.height > 0);
            assert!(
                payload.data.len() * 8
                    == payload.width
                        * payload.height
                        * payload.bit_depth
                        * payload.channels as usize
            );
        } else {
            // Encoded image
            assert!(payload.width > 0);
            assert!(payload.height > 0);
            assert!(!payload.data.is_empty());
            assert!(payload.channels == -1);

            // Check that we can decode the image
            let _img = image::load_from_memory(&payload.data).unwrap();
        }
    }

    #[test]
    fn test_fetch_image() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);
    }

    #[test]
    fn test_extra_fields() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        config["source_config"]["has_latents"] = "masked_image".into();
        config["source_config"]["has_masks"] = "segmentation_mask".into();

        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);

        assert!(sample.additional_images.contains_key("masked_image"));
        check_image(&sample.additional_images["masked_image"]);

        assert!(sample.masks.contains_key("segmentation_mask"));
        check_image(&sample.masks["segmentation_mask"]);
    }

    #[test]
    fn test_crop_resize() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        config["source_config"]["has_latents"] = "masked_image".into();
        config["source_config"]["has_masks"] = "segmentation_mask".into();

        config["image_transform"] = json!(ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 224,
            downsampling_ratio: 16,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: false,
            image_to_rgb8: false,
            encode_format: crate::image_processing::EncodeFormat::default(),
            jpeg_quality: 92
        });

        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);

        assert!(sample.additional_images.contains_key("masked_image"));
        check_image(&sample.additional_images["masked_image"]);

        assert!(sample.masks.contains_key("segmentation_mask"));
        check_image(&sample.masks["segmentation_mask"]);
    }

    #[test]
    fn test_img_compression() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        config["source_config"]["has_latents"] = "masked_image".into();
        config["source_config"]["has_masks"] = "segmentation_mask".into();

        config["image_transform"] = json!(ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 224,
            downsampling_ratio: 16,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: true, // new part being tested
            image_to_rgb8: false,
            encode_format: crate::image_processing::EncodeFormat::default(),
            jpeg_quality: 92
        });

        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);

        assert!(sample.additional_images.contains_key("masked_image"));
        check_image(&sample.additional_images["masked_image"]);

        assert!(sample.masks.contains_key("segmentation_mask"));
        check_image(&sample.masks["segmentation_mask"]);
    }

    #[test]
    fn test_tags() {
        let mut config = get_test_config();
        let tag = "v4_trainset_hq";

        // Test positive tags
        config["source_config"]["tags"] = tag.into();
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        assert!(sample.tags.contains(&tag.to_string()));
        client.stop();

        // Test negative tags
        config["source_config"]["tags"] = "".into();
        config["source_config"]["tags_ne"] = "v4_trainset_hq".into();
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        assert!(!sample.tags.contains(&tag.to_string()));
        client.stop();
    }

    #[test]
    fn test_tags_all() {
        let mut config = get_test_config();
        let tags = "v4_trainset_hq,photo";
        config["source_config"]["tags_all"] = tags.into();
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        // Check that sample.tags contains all the tags in the tags string
        for tag in tags.split(',') {
            assert!(sample.tags.contains(&tag.to_string()));
        }
        client.stop();
    }

    #[test]
    fn test_tags_ne() {
        let mut config = get_test_config();
        let tags = "v4_trainset_hq,photo";
        config["source_config"]["tags_ne"] = tags.into();
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        // Check that sample.tags does not contain any of the tags in the tags string
        debug!("{:?}", sample.tags);
        for tag in tags.split(',') {
            assert!(!sample.tags.contains(&tag.to_string()));
        }
        client.stop();
    }

    #[test]
    fn test_tags_empty() {
        let mut config = get_test_config();
        config["source_config"]["tags_empty"] = "true".into();
        config["source_config"]["tags_ne"] = "".into();
        config["source_config"]["tags"] = "".into();

        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some(), "Sample should be present");

        let sample = sample.unwrap();
        assert!(sample.tags.is_empty(), "Tags should be empty");
        client.stop();
    }

    #[test]
    fn test_tags_ne_all() {
        let mut config = get_test_config();
        let tag1 = "photo";
        let tag2 = "graphic";
        config["source_config"]["tags_ne_all"] = format!("{tag1},{tag2}").into();
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        // Assert that the sample does not contain both tags at the same time
        let has_first = sample.tags.contains(&tag1.to_string());
        let has_second = sample.tags.contains(&tag2.to_string());
        assert!(
            !(has_first && has_second),
            "Sample should not contain both tags at the same time"
        );
        client.stop();
    }

    #[test]
    fn test_attributes_filter() {
        let mut config = get_test_config();
        config["source_config"]["attributes"] = "aesthetic_score__gte:0.5".into();
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        assert!(sample.attributes.contains_key("aesthetic_score"));
        assert!(sample.attributes["aesthetic_score"].as_f64().unwrap() >= 0.5);
        client.stop();
    }

    #[test]
    fn test_pixel_count_filter() {
        let mut config = get_test_config();
        config["source_config"]["min_pixel_count"] = 1000000.into();
        config["source_config"]["max_pixel_count"] = 2000000.into();
        config["source_config"]["require_images"] = json!(true);
        let client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        let payload = sample.image.get_payload();
        assert!(payload.width * payload.height >= 1000000);
        assert!(payload.width * payload.height <= 2000000);
        client.stop();
    }

    #[test]
    fn test_multiple_sources() {
        let limit = 10;
        let mut config = get_test_config();
        config["source_config"]["sources"] = "LAION_ART,LAION_AESTHETICS".into();
        config["limit"] = json!(limit);

        let client = DatagoClient::new(config.to_string());
        let sources = config["source_config"]["sources"]
            .as_str()
            .unwrap()
            .split(",")
            .collect::<Vec<&str>>();

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());

            let sample = sample.unwrap();
            assert!(!sample.id.is_empty());
            debug!("{}", sample.source);
            assert!(sources.contains(&sample.source.as_str()));
        }
    }

    #[test]
    fn test_sources_ne() {
        let limit = 10;
        let mut config = get_test_config();
        config["source_config"]["sources"] = "LAION_ART,LAION_AESTHETICS".into();
        config["source_config"]["sources_ne"] = "LAION_ART".into();
        config["limit"] = json!(limit);

        debug!("{config}");
        let client = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());

            let sample = sample.unwrap();
            assert!(!sample.id.is_empty());
            assert!(sample.source == "LAION_AESTHETICS");
        }
    }

    #[test]
    fn test_random_sampling() {
        let limit = 10;
        let mut config = get_test_config();
        config["source_config"]["random_sampling"] = json!(true);
        config["limit"] = json!(limit);

        // Fill in two sets with some results, and check that they are different
        let mut sample_set_1: HashSet<String> = HashSet::new();
        let mut sample_set_2: HashSet<String> = HashSet::new();

        let client_1 = DatagoClient::new(config.to_string());
        let client_2 = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            sample_set_1.insert(client_1.get_sample().unwrap().id);
        }

        for _ in 0..limit {
            sample_set_2.insert(client_2.get_sample().unwrap().id);
        }
        assert!(sample_set_1 != sample_set_2);
    }

    #[test]
    fn test_ranks() {
        let mut config = get_test_config();
        let limit = 100;
        config["source_config"]["require_images"] = json!(false);
        config["source_config"]["world_size"] = json!(2);
        config["limit"] = json!(limit);

        // Fill in two sets with some results, and check that they are completely different
        let mut sample_set_1: HashSet<String> = HashSet::new();
        let mut sample_set_2: HashSet<String> = HashSet::new();

        config["source_config"]["rank"] = json!(0);
        let client_1 = DatagoClient::new(config.to_string());

        config["source_config"]["rank"] = json!(1);
        let client_2 = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            sample_set_1.insert(client_1.get_sample().unwrap().id);
        }

        for _ in 0..limit {
            sample_set_2.insert(client_2.get_sample().unwrap().id);
        }

        // Check that the two sets are completely different
        assert!(sample_set_1.intersection(&sample_set_2).count() == 0);
    }
}
