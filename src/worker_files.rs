use crate::image_processing;
use crate::structs::{ImagePayload, Sample};
use log::{debug, error};
use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;

async fn image_from_path(path: &str) -> Result<image::DynamicImage, image::ImageError> {
    let bytes = std::fs::read(path).map_err(|e| {
        image::ImageError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e))
    })?;

    image::load_from_memory(&bytes)
}

async fn image_payload_from_path(
    path: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
) -> Result<ImagePayload, image::ImageError> {
    match image_from_path(path).await {
        Ok(new_image) => {
            image_processing::image_to_payload(
                new_image,
                img_tfm,
                &"".to_string(),
                encode_images,
                img_to_rgb8,
            )
            .await
        }
        Err(e) => Err(e),
    }
}

async fn pull_sample(
    sample_json: serde_json::Value,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encode_images: bool,
    img_to_rgb8: bool,
    samples_tx: kanal::Sender<Option<Sample>>,
) -> Result<(), ()> {
    match image_payload_from_path(
        sample_json.as_str().unwrap(),
        &img_tfm,
        encode_images,
        img_to_rgb8,
    )
    .await
    {
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
            error!("Failed to load image from path {} {}", sample_json, e);
            Err(())
        }
    }
}

async fn async_pull_samples(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
    limit: usize,
) {
    // We use async-await here, to better use IO stalls
    // We'll issue N async tasks in parallel, and wait for them to finish
    let max_tasks = min(num_cpus::get() * 4, limit);
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
            encode_images,
            img_to_rgb8,
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
    debug!("file_worker: total samples sent: {}\n", count);

    // Signal the end of the stream
    if samples_tx.send(None).is_ok() {};
}

pub fn pull_samples(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
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
                encode_images,
                img_to_rgb8,
                limit,
            )
            .await;
        });
}
