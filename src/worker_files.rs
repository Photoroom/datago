use crate::image_processing;
use crate::structs::{to_python_image_payload, ImagePayload, Sample};
use crate::worker_utils::{max_tasks_from_env, run_worker};
use log::{debug, error};
use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

async fn image_from_path(path: &str) -> Result<image::DynamicImage, image::ImageError> {
    // The image decoder and std file reads are blocking. Keep them off Tokio's
    // async worker threads so cancellation/channel handling can still progress.
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(path)
            .map_err(|error| image::ImageError::IoError(std::io::Error::other(error)))?;
        let reader = std::io::BufReader::new(file);
        image::ImageReader::new(reader)
            .with_guessed_format()?
            .decode()
    })
    .await
    .map_err(|error| image::ImageError::IoError(std::io::Error::other(error)))?
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
) -> Result<Sample, ()> {
    let path = sample_json.as_str().unwrap();
    debug!("Starting to process file: {}", path);

    match image_payload_from_path(path, &img_tfm, encoding).await {
        Ok(image) => {
            debug!("Successfully processed file: {}", path);
            let sample = Sample {
                id: sample_json.to_string(),
                source: "filesystem".to_string(),
                image: to_python_image_payload(image),
                attributes: HashMap::new(),
                coca_embedding: vec![],
                tags: vec![],
                masks: HashMap::new(),
                latents: HashMap::new(),
                additional_images: HashMap::new(),
                duplicate_state: 0,
            };

            Ok(sample)
        }
        Err(e) => {
            error!("Failed to load image from path {}: {}", path, e);
            // Add more specific error handling based on error type
            if let image::ImageError::IoError(io_err) = e {
                error!("IO Error for file {}: {}", path, io_err);
            }
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
    cancel: CancellationToken,
) {
    let image_transform = Arc::new(image_transform);
    let result = run_worker(
        samples_metadata_rx,
        samples_tx,
        min(max_tasks_from_env(num_cpus::get()), limit),
        limit,
        cancel,
        |item| *item == serde_json::Value::Null,
        move |item| pull_sample(item, image_transform.clone(), encoding),
    )
    .await;
    match result {
        Ok(count) => debug!("file_worker: samples delivered: {count}"),
        Err(error) => error!("file_worker: {error}"),
    }
}

pub fn pull_samples(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
    limit: usize,
    cancel: CancellationToken,
) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        async_pull_samples(
            samples_metadata_rx,
            samples_tx,
            image_transform,
            encoding,
            limit,
            cancel,
        )
        .await;
    });
    // A blocking filesystem syscall already in progress is not cancellable.
    // Do not make this worker's join wait indefinitely for Tokio's blocking pool.
    runtime.shutdown_timeout(std::time::Duration::from_millis(100));
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
    async fn test_rgba_to_rgb8_compositing() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test_rgba_composite.png");

        // Create a 3x1 RGBA image with three different alpha values
        let mut img = image::RgbaImage::new(3, 1);
        img.put_pixel(0, 0, image::Rgba([255u8, 100u8, 50u8, 255u8])); // Fully opaque
        img.put_pixel(1, 0, image::Rgba([200u8, 100u8, 50u8, 128u8])); // Semi-transparent (50%)
        img.put_pixel(2, 0, image::Rgba([255u8, 0u8, 0u8, 0u8])); // Fully transparent
        let dyn_img = image::DynamicImage::ImageRgba8(img);
        dyn_img.save(&image_path).unwrap();

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
        assert_eq!(payload.channels, 3);
        assert_eq!(payload.bit_depth, 8);
        assert_eq!(payload.width, 3);
        assert_eq!(payload.height, 1);

        // Pixel 0: Fully opaque - should preserve RGB values
        assert_eq!(payload.data[0], 255, "Pixel 0 R");
        assert_eq!(payload.data[1], 100, "Pixel 0 G");
        assert_eq!(payload.data[2], 50, "Pixel 0 B");

        // Pixel 1: Semi-transparent - should be composited with gray background
        // With alpha=128/255 ≈ 0.502, we expect roughly:
        // R: 200 * 0.502 + 128 * 0.498 ≈ 164
        // G: 100 * 0.502 + 128 * 0.498 ≈ 114
        // B: 50 * 0.502 + 128 * 0.498 ≈ 89
        assert!(
            (payload.data[3] as i32 - 164).abs() <= 2,
            "Pixel 1 R mismatch: {}",
            payload.data[3]
        );
        assert!(
            (payload.data[4] as i32 - 114).abs() <= 2,
            "Pixel 1 G mismatch: {}",
            payload.data[4]
        );
        assert!(
            (payload.data[5] as i32 - 89).abs() <= 2,
            "Pixel 1 B mismatch: {}",
            payload.data[5]
        );

        // Pixel 2: Fully transparent - should result in gray background (128, 128, 128)
        assert_eq!(payload.data[6], 128, "Pixel 2 R");
        assert_eq!(payload.data[7], 128, "Pixel 2 G");
        assert_eq!(payload.data[8], 128, "Pixel 2 B");
    }

    #[tokio::test]
    async fn test_grayscale_to_rgb8_default_conversion() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test_gray.png");

        // Create a grayscale image
        let mut img = image::GrayImage::new(1, 1);
        img.put_pixel(0, 0, image::Luma([100u8]));
        let dyn_img = image::DynamicImage::ImageLuma8(img);
        dyn_img.save(&image_path).unwrap();

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
        assert_eq!(payload.channels, 3);

        // Grayscale should convert to RGB with same value across channels
        assert_eq!(payload.data[0], 100);
        assert_eq!(payload.data[1], 100);
        assert_eq!(payload.data[2], 100);
    }

    #[tokio::test]
    async fn test_rgb_to_rgb8_passthrough() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test_rgb.png");

        // Create an RGB image
        let mut img = image::RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([255u8, 100u8, 50u8]));
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        dyn_img.save(&image_path).unwrap();

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
        assert_eq!(payload.channels, 3);

        // RGB should pass through unchanged
        assert_eq!(payload.data[0], 255);
        assert_eq!(payload.data[1], 100);
        assert_eq!(payload.data[2], 50);
    }

    #[tokio::test]
    async fn test_pull_sample_success() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let sample_json = serde_json::Value::String(image_path.to_str().unwrap().to_string());

        let result = pull_sample(
            sample_json,
            Arc::new(None),
            image_processing::ImageEncoding::default(),
        )
        .await;

        assert!(result.is_ok());

        let sample = result.unwrap();
        assert_eq!(sample.source, "filesystem");
        assert!(!sample.id.is_empty());
        let payload = sample.image.get_payload();
        assert_eq!(payload.width, 1);
        assert_eq!(payload.height, 1);
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
        let sample_json = serde_json::Value::String("/nonexistent/path.png".to_string());

        let result = pull_sample(
            sample_json,
            Arc::new(None),
            image_processing::ImageEncoding::default(),
        )
        .await;

        assert!(result.is_err());
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
            CancellationToken::new(),
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
            let payload = sample.image.get_payload();
            assert_eq!(payload.width, 1);
            assert_eq!(payload.height, 1);
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
            CancellationToken::new(),
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

        assert_eq!(count, limit);
    }

    fn create_test_webp_image(path: &std::path::Path) {
        // Create a simple 2x2 WebP image
        let img = image::DynamicImage::new_rgb8(2, 2);
        img.save_with_format(path, image::ImageFormat::WebP)
            .unwrap();
    }

    fn create_test_jpeg_image(path: &std::path::Path) {
        // Create a simple 3x3 JPEG image
        let img = image::DynamicImage::new_rgb8(3, 3);
        img.save_with_format(path, image::ImageFormat::Jpeg)
            .unwrap();
    }

    #[tokio::test]
    async fn test_image_from_path_jpeg() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.jpg");
        create_test_jpeg_image(&image_path);

        let result = image_from_path(image_path.to_str().unwrap()).await;
        if let Err(e) = &result {
            eprintln!("Error loading JPEG: {}", e);
        }
        assert!(result.is_ok(), "Failed to load JPEG image");

        let img = result.unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 3);
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
    async fn test_image_payload_from_path_jpeg() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.jpg");
        create_test_jpeg_image(&image_path);

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &None,
            image_processing::ImageEncoding::default(),
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.width, 3);
        assert_eq!(payload.height, 3);
        assert_eq!(payload.original_width, 3);
        assert_eq!(payload.original_height, 3);
        assert!(!payload.data.is_empty());
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

        let sample_json = serde_json::Value::String(image_path.to_str().unwrap().to_string());

        let result = pull_sample(
            sample_json,
            Arc::new(None),
            image_processing::ImageEncoding::default(),
        )
        .await;

        assert!(result.is_ok());

        let sample = result.unwrap();
        assert_eq!(sample.source, "filesystem");
        assert!(!sample.id.is_empty());
        let payload = sample.image.get_payload();
        assert_eq!(payload.width, 2);
        assert_eq!(payload.height, 2);
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
            CancellationToken::new(),
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
