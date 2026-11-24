use crate::image_processing::{ARAwareTransform, ImageEncoding};
use crate::source::{check_partition, default_world_size, Source, SourceContext};
use crate::structs::Sample;
use crate::worker_files;
use log::warn;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    match walk_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Images below a directory, one sample per file.
pub(crate) struct FileSource {
    config: SourceFileConfig,
    image_transform: Arc<Option<ARAwareTransform>>,
    encoding: ImageEncoding,
}

impl Source for FileSource {
    const NAME: &'static str = "file";
    type Config = SourceFileConfig;
    /// The path of one image, as a JSON string.
    type Item = serde_json::Value;

    fn new(config: Self::Config, context: &SourceContext) -> Result<Self, String> {
        check_partition(config.rank, config.world_size)?;
        Ok(Self {
            config,
            image_transform: context.image_transform.clone(),
            encoding: context.encoding,
        })
    }

    async fn enumerate(&self, items: kanal::Sender<Self::Item>) -> Result<(), String> {
        // The walk blocks; it stops on its own once `items` closes.
        let config = self.config.clone();
        tokio::task::spawn_blocking(move || enumerate_files(items, config))
            .await
            .map_err(|error| format!("file walk failed: {error}"))?
    }

    async fn load(&self, item: Self::Item) -> Result<Sample, ()> {
        worker_files::pull_sample(item, self.image_transform.clone(), self.encoding).await
    }
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

    const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    fn create_decodable_images(dir: &Path, count: usize) {
        for i in 0..count {
            image::RgbImage::new(1, 1)
                .save(dir.join(format!("test_{i}.png")))
                .unwrap();
        }
    }

    /// Run the whole pipeline over `root` and read it to exhaustion.
    fn load_all(root: &Path, limit: usize) -> Vec<Sample> {
        let context = SourceContext {
            image_transform: Arc::new(None),
            encoding: ImageEncoding::default(),
            max_connections: 1,
        };
        let config = serde_json::json!({"root_path": root});
        let engine = crate::source::launch::<FileSource>(&config, &context, 20, limit).unwrap();
        let mut samples = Vec::new();
        while let Ok(Some(sample)) = engine.samples_rx.recv_timeout(READ_TIMEOUT) {
            samples.push(sample);
        }
        // A clean pass closes the output after its last sample.
        assert!(engine.samples_rx.recv().is_err());
        assert!(engine.control.error().is_none());
        samples
    }

    #[test]
    fn test_file_source_delivers_every_image() {
        let temp_dir = TempDir::new().unwrap();
        create_decodable_images(temp_dir.path(), 3);

        let samples = load_all(temp_dir.path(), 10);

        assert_eq!(samples.len(), 3);
        for sample in &samples {
            assert_eq!(sample.source, "filesystem");
            let payload = sample.image.get_payload();
            assert_eq!(payload.width, 1);
            assert_eq!(payload.height, 1);
        }
        // The identifier is the path of the image, as a JSON string.
        let mut ids: Vec<_> = samples.iter().map(|sample| sample.id.clone()).collect();
        ids.sort();
        let expected: Vec<_> = (0..3)
            .map(|i| serde_json::json!(temp_dir.path().join(format!("test_{i}.png"))).to_string())
            .collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn test_file_source_with_limit() {
        let temp_dir = TempDir::new().unwrap();
        create_decodable_images(temp_dir.path(), 10);

        assert_eq!(load_all(temp_dir.path(), 3).len(), 3);
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
        // Returning drops the sender: exhaustion is the queue closing.
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
            if let Some(path) = value.as_str() {
                files_rank0.push(path.to_string());
            }
        }

        let mut files_rank1 = Vec::new();
        while let Ok(value) = rx2.recv() {
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
            if let Some(path) = value.as_str() {
                files1.push(path.to_string());
            }
        }

        let mut files2 = Vec::new();
        while let Ok(value) = rx2.recv() {
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
            if let Some(path) = value.as_str() {
                received_files.push(path.to_string());
            }
        }

        assert!(received_files.is_empty());
    }
}
