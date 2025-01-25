use crate::generator_http;
use crate::image_processing::ARAwareTransform;
use crate::image_processing::ImageTransformConfig;
use crate::worker_http;

use kanal::bounded;
use std::sync::Arc;
use std::thread;
use threadpool::ThreadPool;



pub struct DatagoClient {
    pub is_started: bool,
    source_config: generator_http::SourceDBConfig,
    limit: usize,

    // Perf settings
    page_size: usize,
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

    // Sample processing
    image_transform: Option<ARAwareTransform>,
    encode_images: bool,

    // Threads
    pinger: Option<thread::JoinHandle<()>>,
    feeder: Option<thread::JoinHandle<()>>,
    thread_pool: ThreadPool,
}

impl DatagoClient {
    pub fn new(
        source_config: generator_http::SourceDBConfig,
        image_config: Option<ImageTransformConfig>,
        limit: usize,
        page_size: usize,
        rank: usize,
        world_size: usize,
        num_threads: usize,
        max_connections: usize,
        backlog: usize,
    ) -> DatagoClient {
        let (pages_tx, pages_rx) = bounded(2);
        let (samples_meta_tx, samples_meta_rx) = bounded(backlog);
        let (samples_tx, samples_rx) = bounded(backlog);

        let mut image_transform: Option<ARAwareTransform> = None;
        let mut encode_images = false;

        if let Some(image_config) = image_config {
            println!("Cropping and resizing images");
            image_transform = Some(image_config.get_ar_aware_transform());
            encode_images = image_config.pre_encode_images;
        }

        DatagoClient {
            is_started: false,
            source_config,
            limit,
            page_size,
            num_threads,
            max_connections,
            rank,
            world_size,
            pages_tx,
            pages_rx,
            samples_meta_tx,
            samples_meta_rx,
            samples_tx,
            samples_rx,
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
        let page_size = self.page_size;
        let limit = self.limit;
        let rank = self.rank;
        let world_size = self.world_size;

        self.pinger = Some(thread::spawn(move || {
            generator_http::ping_pages(pages_tx, source_config, rank, world_size, page_size, limit);
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
            let samples_meta_rx_local = self.samples_meta_rx.clone();
            let samples_tx_local = self.samples_tx.clone();
            let thread_local_client = http_client.clone();
            let local_image_transform = self.image_transform.clone();
            let encode_images = self.encode_images.clone();

            self.thread_pool.execute(move || {
                worker_http::pull_samples(
                    thread_local_client,
                    samples_meta_rx_local,
                    samples_tx_local,
                    &local_image_transform,
                    encode_images,
                );
            });
        }

        self.is_started = true;
    }

    pub fn get_sample(&mut self) -> Option<worker_http::Sample> {
        if !self.is_started {
            return None;
        }

        // FIXME: Will hang here if no samples are available
        if let Some(sample) = self.samples_rx.recv().ok() {
            return Some(sample);
        }

        None
    }

    pub fn stop(&mut self) {
        if !self.is_started {
            return;
        }

        // -----------------------------------------------------------------
        println!("Wrapping up the datago client");
        if let Some(pinger) = self.pinger.take() {
            match pinger.join() {
                Ok(_) => {}
                Err(e) => {
                    println!("Pinger thread raised an exception: {:?}", e);
                }
            }
        }

        if let Some(feeder) = self.feeder.take() {
            match feeder.join() {
                Ok(_) => {}
                Err(e) => {
                    println!("Feeder thread raised an exception: {:?}", e);
                }
            }
        }

        println!("Shutting down the thread pool");
        self.thread_pool.join()
    }
}

// Ensure cleanup happens even if stop() wasn't called
impl Drop for DatagoClient {
    fn drop(&mut self) {
        self.stop();
    }
}
