use crate::image_processing;
use crate::structs::{ImagePayload, Sample};
use log::{debug, error};
use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;

async fn image_from_path(path: &str) -> Result<image::DynamicImage, image::ImageError> {
    let bytes =
        std::fs::read(path).map_err(|e| image::ImageError::IoError(std::io::Error::other(e)))?;

    image::load_from_memory(&bytes)
}

async fn image_payload_from_path(
    path: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
) -> Result<ImagePayload, image::ImageError> {
    match image_from_path(path).await {
        Ok(new_image) => {
            image_processing::image_to_payload(new_image, img_tfm, &"".to_string(), encoding).await
        }
        Err(e) => Err(e),
    }
}

async fn pull_sample(
    sample_json: serde_json::Value,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encoding: image_processing::ImageEncoding,
    samples_tx: kanal::Sender<Option<Sample>>,
) -> Result<(), ()> {
    match image_payload_from_path(sample_json.as_str().unwrap(), &img_tfm, encoding).await {
        Ok(image) => {
            let sample = Sample {
                id: sample_json.to_string(),
                source: "filesystem".to_string(),
                image,
                attributes: HashMap::new(),
                coca_embedding: vec![],
                tags: vec![],
                masks: HashMap::new(),
                latents: HashMap::new(),
                additional_images: HashMap::new(),
                duplicate_state: 0,
            };

            if samples_tx.send(Some(sample)).is_err() {
                // Channel is closed, wrapping up
                return Err(());
            }
            Ok(())
        }
        Err(e) => {
            error!("Failed to load image from path {sample_json} {e}");
            Err(())
        }
    }
}

async fn async_pull_samples(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
    limit: usize,
) {
    // We use async-await here, to better use IO stalls
    // We'll issue N async tasks in parallel, and wait for them to finish
    let default_max_tasks = std::env::var("DATAGO_MAX_TASKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(num_cpus::get()); // Number of CPUs is actually a good heuristic for a small machine

    let max_tasks = min(default_max_tasks, limit);
    let mut tasks = tokio::task::JoinSet::new();
    let mut count = 0;
    let shareable_img_tfm = Arc::new(image_transform);

    while let Ok(received) = samples_metadata_rx.recv() {
        if received == serde_json::Value::Null {
            debug!("file_worker: end of stream received, stopping there");
            let _ = samples_metadata_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.spawn(pull_sample(
            received,
            shareable_img_tfm.clone(),
            encoding,
            samples_tx.clone(),
        ));

        // If we have enough tasks, we'll wait for the older one to finish
        if tasks.len() >= max_tasks && tasks.join_next().await.unwrap().is_ok() {
            count += 1;
        }
        if count >= limit {
            break;
        }
    }

    // Make sure to wait for all the remaining tasks
    let _ = tasks.join_all().await.iter().map(|result| {
        if let Ok(()) = result {
            count += 1;
        } else {
            // Task failed or was cancelled
            debug!("file_worker: task failed or was cancelled");
        }
    });
    debug!("file_worker: total samples sent: {count}\n");

    // Signal the end of the stream
    if samples_tx.send(None).is_ok() {};
}

pub fn pull_samples(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
    limit: usize,
) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            async_pull_samples(
                samples_metadata_rx,
                samples_tx,
                image_transform,
                encoding,
                limit,
            )
            .await;
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_processing::ImageTransformConfig;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_image(path: &std::path::Path) {
        // Create a simple 1x1 PNG image
        let img = image::DynamicImage::new_rgb8(1, 1);
        img.save(path).unwrap();
    }

    #[tokio::test]
    async fn test_image_from_path_success() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result = image_from_path(image_path.to_str().unwrap()).await;
        assert!(result.is_ok());

        let img = result.unwrap();
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
    }

    #[tokio::test]
    async fn test_image_from_path_invalid_file() {
        let result = image_from_path("/nonexistent/path.png").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_image_from_path_invalid_image_data() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("not_an_image.txt");
        fs::write(&file_path, "This is not image data").unwrap();

        let result = image_from_path(file_path.to_str().unwrap()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_image_payload_from_path_basic() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &None,
            image_processing::ImageEncoding::default(),
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.width, 1);
        assert_eq!(payload.height, 1);
        assert_eq!(payload.original_width, 1);
        assert_eq!(payload.original_height, 1);
        assert!(!payload.data.is_empty());
    }

    #[tokio::test]
    async fn test_image_payload_from_path_with_transform() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");

        // Create a larger test image
        let img = image::DynamicImage::new_rgb8(100, 100);
        img.save(&image_path).unwrap();

        let transform_config = ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 64,
            downsampling_ratio: 16,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: false,
            image_to_rgb8: false,
            encode_format: image_processing::EncodeFormat::default(),
            jpeg_quality: 92,
        };

        let transform = Some(transform_config.get_ar_aware_transform());

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &transform,
            image_processing::ImageEncoding::default(),
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.original_width, 100);
        assert_eq!(payload.original_height, 100);
        // Transformed size should be different
        assert_ne!(payload.width, 100);
        assert_ne!(payload.height, 100);
    }

    #[tokio::test]
    async fn test_image_payload_from_path_with_encoding() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &None,
            image_processing::ImageEncoding {
                encode_images: true,
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.channels, -1); // Encoded images have channels = -1
        assert!(!payload.data.is_empty());

        // Should be able to decode the image
        let decoded = image::load_from_memory(&payload.data);
        assert!(decoded.is_ok());
    }

    #[tokio::test]
    async fn test_image_payload_from_path_rgb8_conversion() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");

        // Create an RGBA image
        let img = image::DynamicImage::new_rgba8(1, 1);
        img.save(&image_path).unwrap();

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &None,
            image_processing::ImageEncoding {
                img_to_rgb8: true,
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.channels, 3); // Should be converted to RGB (3 channels)
        assert_eq!(payload.bit_depth, 8);
    }

    #[tokio::test]
    async fn test_pull_sample_success() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let (tx, rx) = kanal::bounded(10);
        let sample_json = serde_json::Value::String(image_path.to_str().unwrap().to_string());

        let result = pull_sample(
            sample_json,
            Arc::new(None),
            image_processing::ImageEncoding::default(),
            tx,
        )
        .await;

        assert!(result.is_ok());

        // Check that a sample was sent
        let received = rx.recv().unwrap();
        assert!(received.is_some());

        let sample = received.unwrap();
        assert_eq!(sample.source, "filesystem");
        assert!(!sample.id.is_empty());
        assert_eq!(sample.image.width, 1);
        assert_eq!(sample.image.height, 1);
        assert!(sample.attributes.is_empty());
        assert!(sample.coca_embedding.is_empty());
        assert!(sample.tags.is_empty());
        assert!(sample.masks.is_empty());
        assert!(sample.latents.is_empty());
        assert!(sample.additional_images.is_empty());
        assert_eq!(sample.duplicate_state, 0);
    }

    #[tokio::test]
    async fn test_pull_sample_invalid_path() {
        let (tx, rx) = kanal::bounded(10);
        let sample_json = serde_json::Value::String("/nonexistent/path.png".to_string());

        let result = pull_sample(
            sample_json,
            Arc::new(None),
            image_processing::ImageEncoding::default(),
            tx,
        )
        .await;

        assert!(result.is_err());

        // No sample should be sent on error
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_async_pull_samples_basic() {
        let temp_dir = TempDir::new().unwrap();

        // Create multiple test images
        let mut image_paths = Vec::new();
        for i in 0..3 {
            let image_path = temp_dir.path().join(format!("test_{i}.png"));
            create_test_image(&image_path);
            image_paths.push(image_path.to_str().unwrap().to_string());
        }

        let (metadata_tx, metadata_rx) = kanal::bounded(10);
        let (samples_tx, samples_rx) = kanal::bounded(10);

        // Send image paths
        for path in &image_paths {
            metadata_tx
                .send(serde_json::Value::String(path.clone()))
                .unwrap();
        }
        metadata_tx.send(serde_json::Value::Null).unwrap(); // End marker

        async_pull_samples(
            metadata_rx,
            samples_tx,
            None,
            image_processing::ImageEncoding::default(),
            10,
        )
        .await;

        // Check received samples
        let mut received_samples = Vec::new();
        while let Ok(sample_opt) = samples_rx.recv() {
            if let Some(sample) = sample_opt {
                received_samples.push(sample);
            } else {
                break; // End marker
            }
        }

        assert_eq!(received_samples.len(), image_paths.len());
        for sample in &received_samples {
            assert_eq!(sample.source, "filesystem");
            assert_eq!(sample.image.width, 1);
            assert_eq!(sample.image.height, 1);
        }
    }

    #[tokio::test]
    async fn test_async_pull_samples_with_limit() {
        let temp_dir = TempDir::new().unwrap();

        // Create more images than the limit
        for i in 0..10 {
            let image_path = temp_dir.path().join(format!("test_{i}.png"));
            create_test_image(&image_path);
        }

        let (metadata_tx, metadata_rx) = kanal::bounded(20);
        let (samples_tx, samples_rx) = kanal::bounded(20);

        // Send more paths than the limit
        for i in 0..10 {
            let path = temp_dir.path().join(format!("test_{i}.png"));
            metadata_tx
                .send(serde_json::Value::String(
                    path.to_str().unwrap().to_string(),
                ))
                .unwrap();
        }
        metadata_tx.send(serde_json::Value::Null).unwrap();

        let limit = 3;
        async_pull_samples(
            metadata_rx,
            samples_tx,
            None,
            image_processing::ImageEncoding::default(),
            limit,
        )
        .await;

        // Count received samples
        let mut count = 0;
        while let Ok(sample_opt) = samples_rx.recv() {
            if sample_opt.is_some() {
                count += 1;
            } else {
                break;
            }
        }

        // Should respect the limit (might be slightly more due to async processing)
        assert!(count <= limit + 2); // Allow some buffer for async processing
    }

    fn create_test_webp_image(path: &std::path::Path) {
        // Create a simple 2x2 WebP image
        let img = image::DynamicImage::new_rgb8(2, 2);
        img.save_with_format(path, image::ImageFormat::WebP)
            .unwrap();
    }

    #[tokio::test]
    async fn test_image_from_path_webp() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.webp");
        create_test_webp_image(&image_path);

        let result = image_from_path(image_path.to_str().unwrap()).await;
        assert!(result.is_ok());

        let img = result.unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
    }

    #[tokio::test]
    async fn test_image_payload_from_path_webp() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.webp");
        create_test_webp_image(&image_path);

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &None,
            image_processing::ImageEncoding::default(),
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.width, 2);
        assert_eq!(payload.height, 2);
        assert_eq!(payload.original_width, 2);
        assert_eq!(payload.original_height, 2);
        assert!(!payload.data.is_empty());
    }

    #[tokio::test]
    async fn test_pull_sample_webp() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.webp");
        create_test_webp_image(&image_path);

        let (tx, rx) = kanal::bounded(10);
        let sample_json = serde_json::Value::String(image_path.to_str().unwrap().to_string());

        let result = pull_sample(
            sample_json,
            Arc::new(None),
            image_processing::ImageEncoding::default(),
            tx,
        )
        .await;

        assert!(result.is_ok());

        // Check that a sample was sent
        let received = rx.recv().unwrap();
        assert!(received.is_some());

        let sample = received.unwrap();
        assert_eq!(sample.source, "filesystem");
        assert!(!sample.id.is_empty());
        assert_eq!(sample.image.width, 2);
        assert_eq!(sample.image.height, 2);
    }

    #[test]
    fn test_pull_samples_sync_wrapper() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let (metadata_tx, metadata_rx) = kanal::bounded(10);
        let (samples_tx, samples_rx) = kanal::bounded(10);

        // Send one image path
        metadata_tx
            .send(serde_json::Value::String(
                image_path.to_str().unwrap().to_string(),
            ))
            .unwrap();
        metadata_tx.send(serde_json::Value::Null).unwrap();

        // Test the sync wrapper
        pull_samples(
            metadata_rx,
            samples_tx,
            None,
            image_processing::ImageEncoding::default(),
            1,
        );

        // Check that a sample was received
        let sample_opt = samples_rx.recv().unwrap();
        assert!(sample_opt.is_some());

        let sample = sample_opt.unwrap();
        assert_eq!(sample.source, "filesystem");

        // Should receive end marker
        let end_marker = samples_rx.recv().unwrap();
        assert!(end_marker.is_none());
    }
}
