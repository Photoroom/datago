use crate::client::DatagoClient;
use crate::structs::DatagoEngine;
use crate::worker_files;
use kanal::bounded;
use log::{debug, info};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFileConfig {
    pub root_path: String,

    #[serde(default)]
    pub random_sampling: bool,

    #[serde(default)]
    pub rank: usize,

    #[serde(default)]
    pub world_size: usize,
}

// Hash function to be able to dispatch the samples to the correct rank

// The seed ensures consistent hashing across different runs,
// essentially acting as a deterministic salt
const HASH_SEED: u64 = 0x51_73_b3_c3_7f_d9_2e_a1;

fn hash<T: Hash>(t: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    HASH_SEED.hash(&mut hasher); // Add seed first
    t.hash(&mut hasher); // Then hash the actual data
    hasher.finish()
}

fn enumerate_files(
    samples_metadata_tx: kanal::Sender<serde_json::Value>,
    source_config: SourceFileConfig,
    limit: usize,
) {
    // Get an iterator over the files in the root path
    let supported_extensions = ["jpg", "jpeg", "png", "bmp", "gif", "webp"];

    let files = walkdir::WalkDir::new(&source_config.root_path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok());

    let mut files_list: Vec<walkdir::DirEntry> = files
        .filter_map(|entry| {
            let path = entry.path();
            let file_name = path.to_string_lossy().into_owned();
            if supported_extensions
                .iter()
                .any(|&ext| file_name.to_lowercase().ends_with(ext))
            {
                Some(entry)
            } else {
                None
            }
        })
        .collect();

    // If shuffle is set, shuffle the files
    let files_iter = if source_config.random_sampling {
        let mut rng = rand::rng(); // Get a random number generator, thread local. We don´t seed, so typically won't be reproducible
        files_list.shuffle(&mut rng);
        files_list.into_iter()
    } else {
        files_list.into_iter()
    };

    // Iterate over the files and send the paths as they come
    let mut count = 0;

    // We oversubmit arbitrarily by 10% to account for the fact that some files might be corrupted or unreadable.
    // There's another mechanism to limit the number of samples processed as requested by the user, so this is just a buffer.
    let max_submitted_samples = (1.1 * (limit as f64)).ceil() as usize;

    // Build a page from the files iterator
    for entry in files_iter {
        let file_name = entry.path().to_str().unwrap().to_string();

        // If world_size is not 0, we need to dispatch the samples to the correct rank
        if source_config.world_size > 1 {
            let hash = hash(&file_name);
            let target_rank = (hash % source_config.world_size as u64) as usize;
            if target_rank != source_config.rank {
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
            debug!("ping_pages: stream already closed, wrapping up");
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
    info!("Using file as source {}", source_config.root_path);

    // Create a thread which will generate work as it goes. We'll query the filesystem
    // and send the filepaths to the worker pool as we go
    let limit = client.limit;

    let feeder = Some(thread::spawn(move || {
        enumerate_files(samples_metadata_tx, source_config, limit);
    }));

    // Spawn a thread which will handle the async workers through a mutlithread tokio runtime
    let image_transform = client.image_transform.clone();
    let encode_images = client.encode_images;
    let img_to_rgb8 = client.image_to_rgb8;
    let limit = client.limit;
    let samples_metadata_rx_worker = samples_metadata_rx.clone();

    let worker = Some(thread::spawn(move || {
        worker_files::pull_samples(
            samples_metadata_rx_worker,
            samples_tx,
            image_transform,
            encode_images,
            img_to_rgb8,
            limit,
        );
    }));

    DatagoEngine {
        samples_rx,
        feeder,
        worker,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn test_hash_function() {
        let str1 = "test_string1";
        let str2 = "test_string2";
        let str3 = "test_string1"; // Same as str1

        let hash1 = hash(&str1);
        let hash2 = hash(&str2);
        let hash3 = hash(&str3);

        // Same input should produce same hash
        assert_eq!(hash1, hash3);
        // Different inputs should likely produce different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_source_file_config_defaults() {
        let config_json = r#"{
            "root_path": "/tmp/test"
        }"#;

        let config: SourceFileConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.root_path, "/tmp/test");
        assert!(!config.random_sampling);
        assert_eq!(config.rank, 0);
        assert_eq!(config.world_size, 0);
    }

    #[test]
    fn test_source_file_config_full() {
        let config_json = r#"{
            "root_path": "/tmp/test",
            "random_sampling": true,
            "rank": 2,
            "world_size": 4
        }"#;

        let config: SourceFileConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.root_path, "/tmp/test");
        assert!(config.random_sampling);
        assert_eq!(config.rank, 2);
        assert_eq!(config.world_size, 4);
    }

    fn create_test_images(dir: &Path) -> Vec<String> {
        let extensions = ["jpg", "png", "bmp", "gif", "JPEG"];
        let mut files = Vec::new();
        for (i, ext) in extensions.iter().enumerate() {
            let filename = format!("test_image_{}.{}", i, ext);
            let filepath = dir.join(&filename);
            fs::write(&filepath, "fake_image_data").unwrap();
            files.push(filepath.to_string_lossy().to_string());
        }

        // Create a non-image file that should be ignored
        let non_image = dir.join("not_an_image.txt");
        fs::write(&non_image, "text_content").unwrap();

        files
    }

    #[test]
    fn test_enumerate_files_basic() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        let created_files = create_test_images(temp_path);

        let (tx, rx) = kanal::bounded(100);
        let config = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: false,
            rank: 0,
            world_size: 1,
        };

        std::thread::spawn(move || {
            enumerate_files(tx, config, 10);
        });

        let mut received_files = Vec::new();
        while let Ok(value) = rx.recv() {
            if value == serde_json::Value::Null {
                break;
            }
            if let Some(path) = value.as_str() {
                received_files.push(path.to_string());
            }
        }

        assert_eq!(received_files.len(), created_files.len());
        // Check that all created files were found
        for created_file in &created_files {
            assert!(received_files.contains(created_file));
        }
    }

    #[test]
    fn test_enumerate_files_with_limit() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        create_test_images(temp_path);

        let (tx, rx) = kanal::bounded(100);
        let config = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: false,
            rank: 0,
            world_size: 1,
        };

        let limit = 2;
        std::thread::spawn(move || {
            enumerate_files(tx, config, limit);
        });

        let mut received_files = Vec::new();
        while let Ok(value) = rx.recv() {
            if value == serde_json::Value::Null {
                break;
            }
            if let Some(path) = value.as_str() {
                received_files.push(path.to_string());
            }
        }

        // Should respect the limit (plus 10% buffer)
        assert!(received_files.len() <= ((limit as f64 * 1.1).ceil() as usize));
    }

    #[test]
    fn test_enumerate_files_with_world_size() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        create_test_images(temp_path);

        // Test rank 0 of world_size 2
        let (tx1, rx1) = kanal::bounded(100);
        let config1 = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: false,
            rank: 0,
            world_size: 2,
        };

        // Test rank 1 of world_size 2
        let (tx2, rx2) = kanal::bounded(100);
        let config2 = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: false,
            rank: 1,
            world_size: 2,
        };

        std::thread::spawn(move || {
            enumerate_files(tx1, config1, 10);
        });

        std::thread::spawn(move || {
            enumerate_files(tx2, config2, 10);
        });

        let mut files_rank0 = Vec::new();
        while let Ok(value) = rx1.recv() {
            if value == serde_json::Value::Null {
                break;
            }
            if let Some(path) = value.as_str() {
                files_rank0.push(path.to_string());
            }
        }

        let mut files_rank1 = Vec::new();
        while let Ok(value) = rx2.recv() {
            if value == serde_json::Value::Null {
                break;
            }
            if let Some(path) = value.as_str() {
                files_rank1.push(path.to_string());
            }
        }

        // Different ranks should get different files (no overlap)
        for file in &files_rank0 {
            assert!(!files_rank1.contains(file));
        }

        // Both ranks should have some files
        assert!(!files_rank0.is_empty());
        assert!(!files_rank1.is_empty());
    }

    #[test]
    fn test_enumerate_files_random_sampling() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        create_test_images(temp_path);

        // Run twice with random sampling to see if order changes
        let (tx1, rx1) = kanal::bounded(100);
        let config1 = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: true,
            rank: 0,
            world_size: 1,
        };

        let (tx2, rx2) = kanal::bounded(100);
        let config2 = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: true,
            rank: 0,
            world_size: 1,
        };

        std::thread::spawn(move || {
            enumerate_files(tx1, config1, 10);
        });

        std::thread::spawn(move || {
            enumerate_files(tx2, config2, 10);
        });

        let mut files1 = Vec::new();
        while let Ok(value) = rx1.recv() {
            if value == serde_json::Value::Null {
                break;
            }
            if let Some(path) = value.as_str() {
                files1.push(path.to_string());
            }
        }

        let mut files2 = Vec::new();
        while let Ok(value) = rx2.recv() {
            if value == serde_json::Value::Null {
                break;
            }
            if let Some(path) = value.as_str() {
                files2.push(path.to_string());
            }
        }

        // Should find the same files but potentially in different order
        assert_eq!(files1.len(), files2.len());

        // Check that all files from first run are in second run
        for file in &files1 {
            assert!(files2.contains(file));
        }
    }

    #[test]
    fn test_enumerate_files_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        let (tx, rx) = kanal::bounded(100);
        let config = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: false,
            rank: 0,
            world_size: 1,
        };

        std::thread::spawn(move || {
            enumerate_files(tx, config, 10);
        });

        let mut received_files = Vec::new();
        while let Ok(value) = rx.recv() {
            if value == serde_json::Value::Null {
                break;
            }
            if let Some(path) = value.as_str() {
                received_files.push(path.to_string());
            }
        }

        assert!(received_files.is_empty());
    }
}
