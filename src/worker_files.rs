use crate::image_processing;
use crate::structs::{ImagePayload, Sample};
use std::cmp::min;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

async fn image_from_path(path: &str) -> Result<image::DynamicImage, image::ImageError> {
    // Load bytes from the file
    let bytes = std::fs::read(path).map_err(|e| {
        image::ImageError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e))
    })?;

    // Decode the image
    image::load_from_memory(&bytes)
}

async fn image_payload_from_path(
    path: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
) -> Result<ImagePayload, image::ImageError> {
    match image_from_path(path).await {
        Ok(mut new_image) => {
            let original_height = new_image.height() as usize;
            let original_width = new_image.width() as usize;
            let mut channels = new_image.color().channel_count() as i8;
            let mut bit_depth = (new_image.color().bits_per_pixel()
                / new_image.color().channel_count() as u16)
                as usize;

            // Optionally transform the additional image in the same way the main image was
            if let Some(img_tfm) = img_tfm {
                new_image = img_tfm.crop_and_resize(&new_image, None).await;
            }

            let height = new_image.height() as usize;
            let width = new_image.width() as usize;

            // Encode the image if needed
            let mut image_bytes: Vec<u8> = Vec::new();
            if encode_images {
                if new_image.color() != image::ColorType::Rgb8 {
                    new_image = image::DynamicImage::ImageRgb8(new_image.to_rgb8());
                    bit_depth = (new_image.color().bits_per_pixel()
                        / new_image.color().channel_count() as u16)
                        as usize;
                }

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

async fn pull_sample(
    sample_json: serde_json::Value,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encode_images: bool,
    samples_tx: kanal::Sender<Option<Sample>>,
) -> Result<(), ()> {
    match image_payload_from_path(sample_json.as_str().unwrap(), &img_tfm, encode_images).await {
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
            println!("Failed to load image from path {} {}", sample_json, e);
            Err(())
        }
    }
}

pub async fn consume_oldest_task(
    tasks: &mut std::collections::VecDeque<tokio::task::JoinHandle<Result<(), ()>>>,
) -> Result<(), ()> {
    match tasks.pop_front().unwrap().await {
        Ok(_) => Ok(()),
        Err(e) => {
            println!("worker: sample skipped {}", e);
            Err(())
        }
    }
}

async fn async_pull_samples(
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    limit: usize,
) {
    // We use async-await here, to better use IO stalls
    // We'll issue N async tasks in parallel, and wait for them to finish
    let max_tasks = min(num_cpus::get() * 2, limit);
    let mut tasks = std::collections::VecDeque::new();
    let mut count = 0;
    let shareable_img_tfm = Arc::new(image_transform);

    while let Ok(received) = samples_meta_rx.recv() {
        if received == serde_json::Value::Null {
            println!("file_worker: end of stream received, stopping there");
            let _ = samples_meta_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.push_back(tokio::spawn(pull_sample(
            received,
            shareable_img_tfm.clone(),
            encode_images,
            samples_tx.clone(),
        )));

        // If we have enough tasks, we'll wait for the older one to finish
        if tasks.len() >= max_tasks && consume_oldest_task(&mut tasks).await.is_ok() {
            count += 1;
        }
        if count >= limit {
            break;
        }
    }

    // Make sure to wait for all the remaining tasks
    while !tasks.is_empty() {
        if consume_oldest_task(&mut tasks).await.is_ok() {
            count += 1;
        }
    }
    println!("file_worker: total samples sent: {}\n", count);

    // Signal the end of the stream
    if samples_tx.send(None).is_ok() {};
}

pub fn pull_samples(
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    limit: usize,
) {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            async_pull_samples(
                samples_meta_rx,
                samples_tx,
                image_transform,
                encode_images,
                limit,
            )
            .await;
        });
}
