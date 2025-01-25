use crate::generator_http;
use crate::image_processing::ARAwareTransform;
use crate::image_processing::ImageTransformConfig;
use crate::worker_http;
use pyo3::prelude::*;

use kanal::bounded;
use serde::Deserialize;
use std::sync::Arc;
use std::thread;
use threadpool::ThreadPool;

#[derive(Deserialize)]
struct DatagoClientConfig {
    source_config: generator_http::SourceDBConfig,
    image_config: Option<ImageTransformConfig>,
    limit: usize,
    rank: usize,
    world_size: usize,
    samples_buffer_size: usize,
}

#[pyclass]
pub struct DatagoClient {
    pub is_started: bool,
    source_config: generator_http::SourceDBConfig,
    limit: usize,

    // Perf settings
    num_threads: usize,
    max_connections: usize,
    rank: usize,
    world_size: usize,

    // Channels
    pages_tx: kanal::Sender<serde_json::Value>,
    pages_rx: kanal::Receiver<serde_json::Value>,
    samples_meta_tx: kanal::Sender<serde_json::Value>,
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<worker_http::Sample>,
    samples_rx: kanal::Receiver<worker_http::Sample>,
    worker_done_count: Arc<std::sync::atomic::AtomicUsize>,

    // Sample processing
    image_transform: Option<ARAwareTransform>,
    encode_images: bool,

    // Threads
    pinger: Option<thread::JoinHandle<()>>,
    feeder: Option<thread::JoinHandle<()>>,
    thread_pool: ThreadPool,
}

#[pymethods]
impl DatagoClient {
    #[new]
    pub fn new(str_config: String) -> Self {
        let config: DatagoClientConfig = serde_json::from_str(&str_config).unwrap();

        let (pages_tx, pages_rx) = bounded(2);
        let (samples_meta_tx, samples_meta_rx) = bounded(config.samples_buffer_size);
        let (samples_tx, samples_rx) = bounded(config.samples_buffer_size);

        let mut image_transform: Option<ARAwareTransform> = None;
        let mut encode_images = false;

        if let Some(image_config) = config.image_config {
            if image_config.crop_and_resize {
                println!("Cropping and resizing images");
                image_transform = Some(image_config.get_ar_aware_transform());
            }
            encode_images = image_config.pre_encode_images;
        }

        // We use the machine number of CPUs as max number of threads
        let num_threads = num_cpus::get();

        DatagoClient {
            is_started: false,
            source_config: config.source_config,
            limit: config.limit,
            num_threads: num_threads,
            max_connections: 512,
            rank: config.rank,
            world_size: config.world_size,
            pages_tx,
            pages_rx,
            samples_meta_tx,
            samples_meta_rx,
            samples_tx,
            samples_rx,
            worker_done_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            image_transform,
            encode_images,
            pinger: None,
            feeder: None,
            thread_pool: ThreadPool::new(num_threads),
        }
    }

    pub fn start(&mut self) {
        if self.is_started {
            return;
        }

        // Spawn a new thread which will query the DB and send the pages
        let pages_tx = self.pages_tx.clone();
        let source_config = self.source_config.clone();
        let limit = self.limit;
        let rank = self.rank;
        let world_size = self.world_size;

        assert!(
            world_size == 0 || rank < world_size,
            "Rank cannot be greater than or equal to world size"
        );

        self.pinger = Some(thread::spawn(move || {
            generator_http::ping_pages(pages_tx, source_config, rank, world_size, limit);
        }));

        // Spawn a new thread which will pull the pages and send the sample metadata
        let pages_rx = self.pages_rx.clone();
        let samples_meta_tx = self.samples_meta_tx.clone();
        let limit = self.limit;
        self.feeder = Some(thread::spawn(move || {
            generator_http::pull_pages(pages_rx, samples_meta_tx, limit);
        }));

        // Spawn threads which will receive the pages
        let http_client = worker_http::SharedClient {
            client: reqwest::blocking::Client::new(),
            semaphore: Arc::new(tokio::sync::Semaphore::new(self.max_connections)),
        };

        for _ in 0..self.num_threads {
            // Need to clone all these trivial values to move them into the thread
            // FIXME: this is a bit ugly, there must be a better way
            let samples_meta_rx_local = self.samples_meta_rx.clone();
            let samples_tx_local = self.samples_tx.clone();
            let thread_local_client = http_client.clone();
            let local_image_transform = self.image_transform.clone();
            let encode_images = self.encode_images;
            let worker_done_count = self.worker_done_count.clone();

            self.thread_pool.execute(move || {
                worker_http::pull_samples(
                    thread_local_client,
                    samples_meta_rx_local,
                    samples_tx_local,
                    worker_done_count,
                    &local_image_transform,
                    encode_images,
                );
            });
        }

        self.is_started = true;
    }

    pub fn get_sample(&mut self) -> Option<worker_http::Sample> {
        if !self.is_started {
            self.start();
        }
        const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

        // If no more samples and workers are closed, then wrap it up
        if self
            .worker_done_count
            .load(std::sync::atomic::Ordering::Relaxed)
            == self.num_threads
            && self.samples_rx.is_empty()
        {
            println!("No more samples to process, stopping the client");
            self.stop();
            return None;
        }

        // Try to fetch a new sample from the queue
        match self.samples_rx.recv_timeout(TIMEOUT) {
            Ok(sample) => {
                if sample.id != *"" {
                    Some(sample)
                } else {
                    println!("Empty sample received, stopping the client");
                    self.stop();
                    None
                }
            }
            Err(_) => {
                println!("Timeout waiting for sample, stopping the client");
                self.stop();
                None
            }
        }
    }

    pub fn stop(&mut self) {
        if !self.is_started {
            return;
        }

        self.samples_meta_rx.close();
        self.pages_rx.close();
        self.samples_rx.close();

        if let Some(pinger) = self.pinger.take() {
            pinger.join().unwrap();
        }

        if let Some(feeder) = self.feeder.take() {
            feeder.join().unwrap();
        }

        self.thread_pool.join();
        self.is_started = false;
    }
}

// Ensure cleanup happens even if stop() wasn't called
impl Drop for DatagoClient {
    fn drop(&mut self) {
        self.stop();
    }
}
