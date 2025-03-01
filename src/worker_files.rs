use crate::image_processing;
use crate::structs::{ImagePayload, Sample};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

fn image_from_path(path: &str) -> Result<image::DynamicImage, image::ImageError> {
    // Load bytes from the file
    let bytes = std::fs::read(path).map_err(|e| {
        image::ImageError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e))
    })?;

    // Decode the image
    image::load_from_memory(&bytes)
}

fn image_payload_from_path(
    path: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
) -> Result<ImagePayload, image::ImageError> {
    match image_from_path(path) {
        Ok(mut new_image) => {
            let original_height = new_image.height() as usize;
            let original_width = new_image.width() as usize;
            let mut channels = new_image.color().channel_count() as i8;
            let bit_depth = new_image.color().bits_per_pixel() as usize;

            // Optionally transform the additional image in the same way the main image was
            if let Some(img_tfm) = img_tfm {
                new_image = img_tfm.crop_and_resize(&new_image, None);
            }

            let height = new_image.height() as usize;
            let width = new_image.width() as usize;

            // Encode the image if needed
            let mut image_bytes: Vec<u8> = Vec::new();
            if encode_images {
                if new_image
                    .write_to(&mut Cursor::new(&mut image_bytes), image::ImageFormat::Png)
                    .is_err()
                {
                    return Err(image::ImageError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to encode image",
                    )));
                }

                channels = -1; // Signal the fact that the image is encoded
            } else {
                image_bytes = new_image.into_bytes();
            }

            Ok(ImagePayload {
                data: image_bytes,
                original_height,
                original_width,
                height,
                width,
                channels,
                bit_depth,
            })
        }
        Err(e) => Err(e),
    }
}

fn pull_sample(
    sample_json: &serde_json::Value,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
) -> Option<Sample> {
    let image_payload =
        image_payload_from_path(sample_json.as_str().unwrap(), img_tfm, encode_images);

    if let Ok(image) = image_payload {
        Some(Sample {
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
        })
    } else {
        println!("Failed to load image from path {}", sample_json);
        None
    }
}

pub fn pull_samples(
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Sample>,
    worker_done_count: Arc<std::sync::atomic::AtomicUsize>,
    image_transform: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
) {
    while let Ok(received) = samples_meta_rx.recv() {
        if received == serde_json::Value::Null {
            println!("http_worker: end of stream received, stopping there");
            samples_meta_rx.close();
            break;
        }

        if let Some(sample) = pull_sample(&received, image_transform, encode_images) {
            if samples_tx.send(sample).is_err() {
                println!("http_worker: failed to send a sample");
                break;
            }
        } else {
            break;
        }
    }

    worker_done_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}
