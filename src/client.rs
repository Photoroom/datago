use crate::generator_files;
use crate::generator_http;
use crate::image_processing::ARAwareTransform;
use crate::structs::{DatagoClientConfig, Sample, SourceType};
use crate::worker_files;
use crate::worker_http;

use kanal::bounded;
use log::{error, info, warn};
use pyo3::prelude::*;
use std::sync::Arc;
use std::thread;

#[pyclass]
pub struct DatagoClient {
    pub is_started: bool,
    source_type: SourceType,
    source_config: serde_json::Value,
    limit: usize,

    // Perf settings
    max_connections: usize,
    rank: usize,
    world_size: usize,

    // Channels
    pages_tx: kanal::Sender<serde_json::Value>,
    pages_rx: kanal::Receiver<serde_json::Value>,
    samples_meta_tx: kanal::Sender<serde_json::Value>,
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    samples_rx: kanal::Receiver<Option<Sample>>,

    // Sample processing
    image_transform: Option<ARAwareTransform>,
    encode_images: bool,
    image_to_rgb8: bool, // Convert all images to RGB 8 bits format

    // Threads
    pinger: Option<thread::JoinHandle<()>>,
    feeder: Option<thread::JoinHandle<()>>,
    worker: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl DatagoClient {
    #[new]
    pub fn new(str_config: String) -> Self {
        let config: DatagoClientConfig = serde_json::from_str(&str_config).unwrap(); // Ok to panic here, no way we can recover

        let (pages_tx, pages_rx) = bounded(2);

        // Logic here is that the final results buffer should be around the threadpool size, so that threads are not blocked in the worst case
        // metadata channel is cheap, no payload yet, so we can overprovision it in comparison
        let (samples_meta_tx, samples_meta_rx) = bounded(config.samples_buffer_size * 2);
        let (samples_tx, samples_rx) = bounded(config.samples_buffer_size);

        let mut image_transform: Option<ARAwareTransform> = None;
        let mut encode_images = false;
        let mut image_to_rgb8 = false;
        if let Some(image_config) = config.image_config {
            if image_config.crop_and_resize {
                image_transform = Some(image_config.get_ar_aware_transform());
            }
            encode_images = image_config.pre_encode_images;
            image_to_rgb8 = image_config.image_to_rgb8;
        }

        DatagoClient {
            is_started: false,
            source_type: config.source_type,
            source_config: config.source_config,
            limit: config.limit,
            max_connections: 256,
            rank: config.rank,
            world_size: config.world_size,
            pages_tx,
            pages_rx,
            samples_meta_tx,
            samples_meta_rx,
            samples_tx,
            samples_rx,
            image_transform,
            encode_images,
            image_to_rgb8,
            pinger: None,
            feeder: None,
            worker: None,
        }
    }

    pub fn start(&mut self) {
        if self.is_started {
            return;
        }

        // In Python, by default the log level is set to "warn", so we do the same here
        // This has no effect, in case the user has previously called initialize_logging().
        initialize_logging(Some("warn".to_string()));

        // Spawn a new thread which will query the DB and send the pages
        let pages_tx = self.pages_tx.clone();
        let limit = self.limit;
        let rank = self.rank;
        let world_size = self.world_size;

        assert!(
            world_size == 0 || rank < world_size,
            "Rank cannot be greater than or equal to world size"
        );

        let http_client = Arc::new(worker_http::new_shared_client(self.max_connections));

        match self.source_type {
            SourceType::Db => {
                // convert the source_config to a SourceDBConfig
                let source_db_config: generator_http::SourceDBConfig =
                    serde_json::from_value(self.source_config.clone()).unwrap();

                info!("Using DB as source");
                let http_client = http_client.clone();

                self.pinger = Some(thread::spawn(move || {
                    generator_http::ping_pages(
                        http_client,
                        pages_tx,
                        source_db_config,
                        rank,
                        world_size,
                        limit,
                    );
                }));
            }
            SourceType::File => {
                // convert the source_config to a SourceFileConfig
                let source_file_config: generator_files::SourceFileConfig =
                    serde_json::from_value(self.source_config.clone()).unwrap();

                info!("Using file as source {}", source_file_config.root_path);

                self.pinger = Some(thread::spawn(move || {
                    generator_files::ping_files(
                        pages_tx,
                        source_file_config,
                        rank,
                        world_size,
                        limit,
                    );
                }));
            }
        }

        // Spawn a new thread which will pull the pages and send the sample metadata
        let pages_rx = self.pages_rx.clone();
        let samples_meta_tx = self.samples_meta_tx.clone();
        let limit = self.limit;
        self.feeder = Some(thread::spawn(move || {
            generator_http::dispatch_pages(pages_rx, samples_meta_tx, limit);
        }));

        // Spawn a thread which will handle the async workers
        // Need to clone all these trivial values to move them into the thread
        // FIXME: this is a bit ugly, there must be a better way
        let samples_meta_rx_local = self.samples_meta_rx.clone();
        let samples_tx_local = self.samples_tx.clone();
        let local_image_transform = self.image_transform.clone();
        let encode_images = self.encode_images;
        let image_to_rgb8 = self.image_to_rgb8;

        match self.source_type {
            SourceType::Db => {
                // Spawn threads which will receive the pages
                self.worker = Some(thread::spawn(move || {
                    worker_http::pull_samples(
                        http_client,
                        samples_meta_rx_local,
                        samples_tx_local,
                        local_image_transform,
                        encode_images,
                        image_to_rgb8,
                        limit,
                    );
                }));
            }
            SourceType::File => {
                self.worker = Some(thread::spawn(move || {
                    worker_files::pull_samples(
                        samples_meta_rx_local,
                        samples_tx_local,
                        local_image_transform,
                        encode_images,
                        image_to_rgb8,
                        limit,
                    );
                }));
            }
        }

        self.is_started = true;
    }

    pub fn get_sample(&mut self) -> Option<Sample> {
        if !self.is_started {
            self.start();
        }

        // If no more samples and workers are closed, then wrap it up
        if self.samples_rx.is_closed() {
            info!("No more samples to process, stopping the client");
            self.stop();
            return None;
        }

        // Try to fetch a new sample from the queue
        // The client will timeout if zero sample is received in 5 minutes
        // At this point it will stop and wrap everything up
        const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

        match self.samples_rx.recv_timeout(TIMEOUT) {
            Ok(sample) => match sample {
                Some(sample) => Some(sample),
                None => {
                    info!("End of stream received, stopping the client");
                    self.stop();
                    None
                }
            },
            Err(e) => {
                warn!("Timeout waiting for sample, stopping the client. {}", e);
                self.stop();
                None
            }
        }
    }

    pub fn stop(&mut self) {
        if !self.is_started {
            return;
        }

        let _ = self.samples_meta_rx.close();
        let _ = self.pages_rx.close();
        let _ = self.samples_tx.close();

        if let Some(pinger) = self.pinger.take() {
            if pinger.join().is_err() {
                error!("Failed to join pinger thread");
            }
        }

        if let Some(feeder) = self.feeder.take() {
            if feeder.join().is_err() {
                error!("Failed to join feeder thread");
            }
        }

        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                error!("Failed to join worker thread");
            }
        }
        self.is_started = false;
    }
}

// Ensure cleanup happens even if stop() wasn't called
impl Drop for DatagoClient {
    fn drop(&mut self) {
        self.stop();
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
