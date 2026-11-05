use crate::client::DatagoClient;
use crate::error::DatagoError;
use crate::structs::{DatagoEngine, MetadataReceiver};
use crate::worker_files;
use kanal::bounded;
use log::{info, warn};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFileConfig {
    pub root_path: String,

    #[serde(default)]
    pub random_sampling: bool,

    #[serde(default)]
    pub rank: usize,

    #[serde(default = "default_world_size")]
    pub world_size: usize,
}

fn default_world_size() -> usize {
    1
}

fn get_data_slice_multirank(quorum: usize, rank: usize, world_size: usize) -> (usize, usize) {
    assert!(rank < world_size, "Rank must be less than world size");

    let chunk_size = quorum / world_size; // This floors by default
    let remainder = quorum % world_size;

    let start = if rank < remainder {
        rank * (chunk_size + 1)
    } else {
        remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
    };

    let end = if (rank + 1) <= remainder {
        (rank + 1) * (chunk_size + 1)
    } else {
        remainder * (chunk_size + 1) + (rank + 1 - remainder) * chunk_size
    };
    (start, end)
}

fn enumerate_files(
    samples_metadata_tx: kanal::Sender<serde_json::Value>,
    source_config: SourceFileConfig,
) -> Result<(), String> {
    // Get an iterator over regular files only. Besides skipping directories with
    // image-like names, this avoids opening FIFOs/devices in the blocking decoder.
    let supported_extensions = ["jpg", "jpeg", "png", "bmp", "gif", "webp"];

    // Use streaming walkdir to avoid loading all files into memory at once
    let mut walk_error = None;
    let walker = walkdir::WalkDir::new(&source_config.root_path)
        .follow_links(false)
        // All ranks must enumerate the same order before taking disjoint slices.
        .sort_by_file_name()
        .into_iter()
        // Test cancellation before filtering extensions so a tree full of
        // unrelated files does not require a complete walk after stop().
        .take_while(|_| !samples_metadata_tx.is_disconnected())
        .filter_map(|entry| {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if error.depth() == 0 => {
                    walk_error = Some(error.to_string());
                    return None;
                }
                Err(error) => {
                    warn!("Skipping unreadable filesystem entry: {error}");
                    return None;
                }
            };
            // Follow symlinks for the type check: curated subsets are often
            // symlink farms. FIFOs, devices and directories are still skipped.
            let file_type = entry.file_type();
            let is_file = file_type.is_file()
                || (file_type.is_symlink()
                    && std::fs::metadata(entry.path()).is_ok_and(|meta| meta.is_file()));
            if !is_file {
                return None;
            }
            let path = entry.path();
            let extension = path.extension()?.to_str()?;
            if supported_extensions
                .iter()
                .any(|ext| extension.eq_ignore_ascii_case(ext))
            {
                Some(entry)
            } else {
                None
            }
        });

    // Sequential single-rank reads stream immediately. Global shuffling and
    // balanced contiguous rank slices require knowing the complete file list.
    let entries: Box<dyn Iterator<Item = _> + '_> = if source_config.random_sampling
        || source_config.world_size > 1
    {
        let mut files: Vec<_> = walker.collect();
        if source_config.world_size > 1 {
            let (start, end) =
                get_data_slice_multirank(files.len(), source_config.rank, source_config.world_size);
            files = files.drain(start..end).collect();
        }
        if source_config.random_sampling {
            files.shuffle(&mut rand::rng());
        }
        Box::new(files.into_iter())
    } else {
        Box::new(walker)
    };

    // The worker alone counts successful samples and closes this queue when
    // satisfied. Candidate failures must never exhaust an arbitrary allowance.
    for entry in entries {
        let path = entry.path().to_string_lossy().into_owned();
        if samples_metadata_tx
            .send(serde_json::Value::String(path))
            .is_err()
        {
            return Ok(());
        }
    }
    if let Some(error) = walk_error {
        return Err(error);
    }
    let _ = samples_metadata_tx.send(serde_json::Value::Null);
    Ok(())
}

pub fn orchestrate(client: &DatagoClient) -> Result<DatagoEngine, DatagoError> {
    // Validate before creating any queues or starting producer threads.
    let source_config: SourceFileConfig = serde_json::from_value(client.source_config.clone())
        .map_err(|error| DatagoError::Configuration(format!("file source: {error}")))?;
    // Start pulling the samples, which spread across two steps. The samples will land in the last kanal,
    // all the threads pausing when the required buffer depth is reached.
    // - A first thread will query the filesystem and get pages of filepaths back. It will dispatch the filepaths
    // to the worker pool.
    // - The worker pool will load the files, deserialize them, do the required pre-processing then commit to the ready queue.

    // TODO: Pass over an Arc ref of the client instead of doing the current member copies

    // Allocate all the message passing pipes
    let (samples_metadata_tx, samples_metadata_rx) = bounded(client.samples_buffer * 2);
    let (samples_tx, samples_rx) = bounded(client.samples_buffer);
    let cancel = CancellationToken::new();
    let control = crate::pipeline::PipelineControl::new(
        samples_rx.clone(),
        MetadataReceiver::Json(samples_metadata_rx.clone()),
        cancel.clone(),
    );

    info!("Using file as source {}", source_config.root_path);

    // Create a thread which will generate work as it goes. We'll query the filesystem
    // and send the filepaths to the worker pool as we go
    let feeder = Some(control.spawn_feeder(samples_metadata_tx.clone(), move || {
        enumerate_files(samples_metadata_tx, source_config)
    }));

    // Spawn a thread which will handle the async workers through a mutlithread tokio runtime
    let image_transform = client.image_transform.clone();
    let encoding = crate::image_processing::ImageEncoding {
        encode_images: client.encode_images,
        img_to_rgb8: client.img_to_rgb8,
        encode_format: client.encode_format,
        jpeg_quality: client.jpeg_quality,
    };
    let limit = client.limit;
    let samples_metadata_rx_worker = samples_metadata_rx.clone();
    let cancel_worker = cancel.clone();

    let worker = Some(control.spawn_worker(samples_tx.clone(), move || {
        worker_files::pull_samples(
            samples_metadata_rx_worker,
            samples_tx,
            image_transform,
            encoding,
            limit,
            cancel_worker,
        )
    }));

    Ok(DatagoEngine {
        control,
        samples_rx,
        feeder,
        worker,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn test_get_data_slice_multirank() {
        // Test case 1: Equal distribution with no remainder
        let (start, end) = get_data_slice_multirank(10, 0, 2);
        assert_eq!(start, 0);
        assert_eq!(end, 5);

        let (start, end) = get_data_slice_multirank(10, 1, 2);
        assert_eq!(start, 5);
        assert_eq!(end, 10);

        // Test case 2: Unequal distribution with remainder
        let (start, end) = get_data_slice_multirank(11, 0, 2);
        assert_eq!(start, 0);
        assert_eq!(end, 6);

        let (start, end) = get_data_slice_multirank(11, 1, 2);
        assert_eq!(start, 6);
        assert_eq!(end, 11);

        // Test case 3: Multiple ranks with remainder
        let (start, end) = get_data_slice_multirank(13, 0, 3);
        assert_eq!(start, 0);
        assert_eq!(end, 5);

        let (start, end) = get_data_slice_multirank(13, 1, 3);
        assert_eq!(start, 5);
        assert_eq!(end, 9);

        let (start, end) = get_data_slice_multirank(13, 2, 3);
        assert_eq!(start, 9);
        assert_eq!(end, 13);

        // Test case 4: Single rank
        let (start, end) = get_data_slice_multirank(10, 0, 1);
        assert_eq!(start, 0);
        assert_eq!(end, 10);

        // Test case 5: Edge case with zero quorum
        let (start, end) = get_data_slice_multirank(0, 0, 1);
        assert_eq!(start, 0);
        assert_eq!(end, 0);

        // Test case 6: Edge case with zero world size (should panic or handle gracefully)
        // Note: This test assumes the function should panic or handle the zero division gracefully
        // You may need to adjust the test based on your actual error handling
        let result = std::panic::catch_unwind(|| {
            get_data_slice_multirank(10, 0, 0);
        });
        assert!(result.is_err());
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
        assert_eq!(config.world_size, 1);
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

    fn create_test_images(dir: &Path, min_num_files: usize) -> Vec<String> {
        let extensions = ["jpg", "png", "bmp", "gif", "JPEG"];
        let mut files = Vec::new();
        let mut n_files = 0;

        while n_files < min_num_files {
            for (i, ext) in extensions.iter().enumerate() {
                let filename = format!("test_image_{n_files}_{i}.{ext}");
                let filepath = dir.join(&filename);
                fs::write(&filepath, "fake_image_data").unwrap();
                files.push(filepath.to_string_lossy().to_string());
                n_files += 1;
            }
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
        let limit = 10;
        let created_files = create_test_images(temp_path, limit);

        let (tx, rx) = kanal::bounded(100);
        let config = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: false,
            rank: 0,
            world_size: 1,
        };

        std::thread::spawn(move || {
            enumerate_files(tx, config).unwrap();
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
    #[cfg(unix)]
    fn test_enumerate_files_symlinks() {
        use std::os::unix::fs::symlink;
        use std::os::unix::net::UnixListener;

        let targets = TempDir::new().unwrap();
        let target = create_test_images(targets.path(), 1).remove(0);
        let dataset = TempDir::new().unwrap();
        let image_link = dataset.path().join("image.png");
        symlink(target, &image_link).unwrap();
        symlink(targets.path(), dataset.path().join("directory.png")).unwrap();
        symlink(
            targets.path().join("missing.png"),
            dataset.path().join("dangling.png"),
        )
        .unwrap();
        let socket_path = targets.path().join("socket.png");
        let _socket = UnixListener::bind(&socket_path).unwrap();
        symlink(socket_path, dataset.path().join("socket.png")).unwrap();

        let (sender, receiver) = kanal::bounded(10);
        enumerate_files(
            sender,
            SourceFileConfig {
                root_path: dataset.path().to_string_lossy().into_owned(),
                random_sampling: false,
                rank: 0,
                world_size: 1,
            },
        )
        .unwrap();
        assert_eq!(
            receiver.recv().unwrap(),
            serde_json::Value::String(image_link.to_string_lossy().into_owned())
        );
        assert_eq!(receiver.recv().unwrap(), serde_json::Value::Null);
        assert!(receiver.recv().is_err());
    }

    #[test]
    fn test_enumerate_files_does_not_cap_candidates() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        let limit = 10;
        let created = create_test_images(temp_path, limit * 4);

        let (tx, rx) = kanal::bounded(100);
        let config = SourceFileConfig {
            root_path: temp_path.to_string_lossy().to_string(),
            random_sampling: false,
            rank: 0,
            world_size: 1,
        };

        std::thread::spawn(move || {
            enumerate_files(tx, config).unwrap();
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

        assert_eq!(received_files.len(), created.len());
    }

    #[test]
    fn test_enumerate_files_with_world_size() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        let limit = 10;

        create_test_images(temp_path, limit * 2); // We'll check that each rank has "limit" files l416

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
            enumerate_files(tx1, config1).unwrap();
        });

        std::thread::spawn(move || {
            enumerate_files(tx2, config2).unwrap();
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
        assert!(files_rank0.len() >= limit);
        assert!(files_rank1.len() >= limit);
    }

    #[test]
    fn test_enumerate_files_random_sampling() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        let limit = 10;

        create_test_images(temp_path, limit);

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
            enumerate_files(tx1, config1).unwrap();
        });

        std::thread::spawn(move || {
            enumerate_files(tx2, config2).unwrap();
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
            enumerate_files(tx, config).unwrap();
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
