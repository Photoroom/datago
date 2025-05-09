use crate::client::DatagoClient;
use crate::structs::DatagoEngine;
use crate::worker_files;
use kanal::bounded;
use log::debug;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFileConfig {
    pub root_path: String,

    #[serde(default)]
    pub random_order: bool,
}

// Hash function to be able to dispatch the samples to the correct rank
fn hash<T: Hash>(t: &T) -> u64 {
    use std::hash::Hasher;
    let mut s = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn enumerate_files(
    samples_metadata_tx: kanal::Sender<serde_json::Value>,
    source_config: SourceFileConfig,
    rank: usize,
    world_size: usize,
    limit: usize,
) {
    // Get an iterator over the files in the root path
    let supported_extensions = [
        "jpg", "jpeg", "png", "bmp", "gif", "JPG", "JPEG", "PNG", "BMP", "GIF",
    ];

    let files = walkdir::WalkDir::new(&source_config.root_path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok());

    let mut files_list: Vec<walkdir::DirEntry> = files
        .filter_map(|entry| {
            let path = entry.path();
            let file_name = path.to_str().unwrap().to_string();
            if supported_extensions
                .iter()
                .any(|&ext| file_name.ends_with(ext))
            {
                Some(entry)
            } else {
                None
            }
        })
        .collect();

    // If random_order is set, shuffle the files
    let files_iter = if source_config.random_order {
        let mut rng = rand::rng(); // Falls back to OsRng, which will differ over time
        files_list.shuffle(&mut rng);
        files_list.into_iter()
    } else {
        files_list.into_iter()
    };

    // Iterate over the files and send the paths as they come
    let mut count = 0;
    let max_submitted_samples = (1.1 * (limit as f64)).ceil() as usize;

    // Build a page from the files iterator
    for entry in files_iter {
        let file_name = entry.path().to_str().unwrap().to_string();

        // If world_size is not 0, we need to dispatch the samples to the correct rank
        if world_size > 1 {
            let hash = hash(&file_name);
            let target_rank = (hash % world_size as u64) as usize;
            if target_rank != rank {
                continue;
            }
        }

        if samples_metadata_tx
            .send(serde_json::Value::String(file_name))
            .is_err()
        {
            break;
        }

        count += 1;

        if count >= max_submitted_samples {
            // NOTE: This doesn´t count the samples which have actually been processed
            debug!("ping_pages: reached the limit of samples requested. Shutting down");
            break;
        }
    }

    // Either we don't have any more samples or we have reached the limit
    debug!(
        "ping_pages: total samples requested: {}. page samples served {}",
        limit, count
    );

    // Send an empty value to signal the end of the stream
    match samples_metadata_tx.send(serde_json::Value::Null) {
        Ok(_) => {}
        Err(_) => {
            debug!("ping_pages: stream already closed, all good");
        }
    };
}

pub fn orchestrate(client: &DatagoClient) -> DatagoEngine {
    // Start pulling the samples, which spread across two steps. The samples will land in the last kanal,
    // all the threads pausing when the required buffer depth is reached.
    // - A first thread will query the filesystem and get pages of filepaths back. It will dispatch the filepaths
    // to the worker pool.
    // - The worker pool will load the files, deserialize them, do the required pre-processing then commit to the ready queue.

    // TODO: Pass over an Arc ref of the client instead of doing the current member copies

    // Allocate all the message passing pipes
    let (samples_metadata_tx, samples_metadata_rx) = bounded(client.samples_buffer * 2);
    let (samples_tx, samples_rx) = bounded(client.samples_buffer);

    // Convert the source_config to a SourceFileConfig
    let source_config: SourceFileConfig =
        serde_json::from_value(client.source_config.clone()).unwrap();
    println!("Using file as source {}", source_config.root_path);

    // Create a thread which will generate work as it goes. We'll query the filesystem
    // and send the filepaths to the worker pool as we go
    let rank = client.rank;
    let limit = client.limit;
    let world_size = client.world_size;

    let generator = Some(thread::spawn(move || {
        enumerate_files(samples_metadata_tx, source_config, rank, world_size, limit);
    }));

    // Spawn a thread which will handle the async workers through a mutlithread tokio runtime
    let image_transform = client.image_transform.clone();
    let encode_images = client.encode_images;
    let img_to_rgb8 = client.image_to_rgb8;
    let limit = client.limit;
    let samples_tx_worker = samples_tx.clone();
    let samples_metadata_rx_worker = samples_metadata_rx.clone();

    let worker = Some(thread::spawn(move || {
        worker_files::pull_samples(
            samples_metadata_rx,
            samples_tx_worker,
            image_transform,
            encode_images,
            img_to_rgb8,
            limit,
        );
    }));

    DatagoEngine {
        pages_rx: samples_metadata_rx_worker,
        samples_tx,
        samples_rx,
        pinger: None,
        feeder: generator,
        worker,
    }
}
