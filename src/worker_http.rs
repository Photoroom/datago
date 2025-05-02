use crate::image_processing;
use crate::structs::{CocaEmbedding, ImagePayload, LatentPayload, Sample, SharedClient, UrlLatent};
use crate::worker_files::consume_oldest_task;
use log::{debug, error, warn};
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;

// ------------------------------------------------------------------
#[derive(Debug, Serialize, Deserialize)]
struct SampleMetadata {
    id: String,
    source: String,
    attributes: HashMap<String, serde_json::Value>,
    duplicate_state: Option<i32>,
    image_direct_url: Option<String>,
    latents: Option<Vec<UrlLatent>>,
    tags: Option<Vec<String>>,
    coca_embedding: Option<CocaEmbedding>,
}

pub async fn bytes_from_url(shared_client: &SharedClient, url: &str) -> Option<Vec<u8>> {
    // Retry on the request a few times
    let timeout = std::time::Duration::from_secs(120);
    let _permit = shared_client.semaphore.acquire();

    // Get a client reference with optimized settings
    let client = &shared_client.client;
    // Send request with specific timeout and connection settings
    if let Ok(response) = client
        .get(url)
        .timeout(timeout)
        .header(reqwest::header::CONNECTION, "keep-alive")
        .send()
        .await
    {
        if let Ok(bytes) = response.bytes().await {
            return Some(bytes.to_vec());
        }
    }

    None
}

async fn image_from_url(
    client: &SharedClient,
    url: &str,
    retries: i32,
) -> Result<image::DynamicImage, image::ImageError> {
    // Retry on the fetch and decode a few times, could happen that we get a broken packet
    for _ in 0..retries {
        if let Some(bytes) = bytes_from_url(client, url).await {
            return image::load_from_memory(&bytes);
        }
    }
    Err(image::ImageError::IoError(std::io::Error::new(
        std::io::ErrorKind::Other,
        "Failed to fetch image bytes",
    )))
}

async fn payload_from_url(
    client: &SharedClient,
    url: &str,
    retries: i32,
) -> Result<Vec<u8>, std::io::Error> {
    // Retry on the fetch and decode a few times, could happen that we get a broken packet
    for _ in 0..retries {
        match bytes_from_url(client, url).await {
            Some(bytes) => {
                return Ok(bytes);
            }
            None => {
                warn!("Failed to get bytes from URL: {}. Retrying", url);
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::Other,
        "Failed to fetch bytes buffer",
    ))
}

async fn image_payload_from_url(
    client: &SharedClient,
    url: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    aspect_ratio: &String,
    encode_images: bool,
    img_to_rgb8: bool,
) -> Result<ImagePayload, image::ImageError> {
    let retries = 5;

    match image_from_url(client, url, retries).await {
        Ok(new_image) => {
            image_processing::image_to_payload(
                new_image,
                img_tfm,
                aspect_ratio,
                encode_images,
                img_to_rgb8,
            )
            .await
        }
        Err(e) => Err(e),
    }
}

async fn pull_sample(
    client: Arc<SharedClient>,
    sample_json: serde_json::Value,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encode_images: bool,
    img_to_rgb8: bool,
    samples_tx: Arc<kanal::Sender<Option<Sample>>>,
) -> Result<(), ()> {
    // Deserialize the sample metadata
    let sample: SampleMetadata = serde_json::from_value(sample_json).unwrap(); // Ok to surface an error here, return type will catch it

    // Pull the image for a start, get an idea of the speed
    let mut image_payload: Option<ImagePayload> = None;
    let mut aspect_ratio = String::new();

    if let Some(image_url) = &sample.image_direct_url {
        image_payload = match image_payload_from_url(
            &client,
            image_url,
            &img_tfm,
            &String::new(),
            encode_images,
            img_to_rgb8,
        )
        .await
        {
            Ok(payload) => {
                aspect_ratio = image_processing::aspect_ratio_to_str((
                    payload.width as i32,
                    payload.height as i32,
                ));
                Some(payload)
            }
            Err(e) => {
                error!("Failed to get image from URL: {}\n {:?}", image_url, e);
                error!("Error: {:?}", e);
                return Err(());
            }
        };
    }

    // Same for the latents, mask and masked images, if they exist
    let mut masks: HashMap<String, ImagePayload> = HashMap::new();
    let mut additional_images: HashMap<String, ImagePayload> = HashMap::new();
    let mut latents: HashMap<String, LatentPayload> = HashMap::new();

    if let Some(exposed_latents) = &sample.latents {
        for latent in exposed_latents {
            if latent.latent_type.contains("image") && !latent.latent_type.contains("latent_") {
                // Image types, registered as latents but they need to be jpg-decoded
                match image_payload_from_url(
                    &client,
                    &latent.file_direct_url,
                    &img_tfm,
                    &aspect_ratio,
                    encode_images,
                    img_to_rgb8,
                )
                .await
                {
                    Ok(additional_image_payload) => {
                        additional_images
                            .insert(latent.latent_type.clone(), additional_image_payload);
                    }

                    Err(e) => {
                        println!(
                            "Failed to get additional image from URL: {} {} {:?}",
                            latent.latent_type, latent.file_direct_url, e
                        );
                        return Err(());
                    }
                }
            } else if latent.is_mask {
                // Mask types, registered as latents but they need to be png-decoded
                match image_payload_from_url(
                    &client,
                    &latent.file_direct_url,
                    &img_tfm,
                    &aspect_ratio,
                    encode_images,
                    false, // Masks are not converted to RGB8
                )
                .await
                {
                    Ok(mask_payload) => {
                        masks.insert(latent.latent_type.clone(), mask_payload);
                    }

                    Err(e) => {
                        println!(
                            "Failed to get mask from URL: {} {} {:?}",
                            latent.latent_type, latent.file_direct_url, e
                        );
                        return Err(());
                    }
                }
            } else {
                // Vanilla latents, pure binary payloads
                match payload_from_url(&client, &latent.file_direct_url, 5).await {
                    Ok(latent_payload) => {
                        latents.insert(
                            latent.latent_type.clone(),
                            LatentPayload {
                                len: latent_payload.len(),
                                data: latent_payload,
                            },
                        );
                    }
                    Err(e) => {
                        error!("Error fetching latent: {} {}", latent.file_direct_url, e);
                        return Err(());
                    }
                }
            }
        }
    }

    // Add the images and latents to the sample
    let pulled_sample = Sample {
        id: sample.id,
        source: sample.source,
        attributes: sample.attributes,
        duplicate_state: sample.duplicate_state.unwrap_or(-1),
        image: image_payload.unwrap_or(ImagePayload {
            data: Vec::new(),
            original_height: 0,
            original_width: 0,
            height: 0,
            width: 0,
            channels: 0,
            bit_depth: 0,
        }),
        masks,
        additional_images,
        latents,
        coca_embedding: sample.coca_embedding.unwrap_or_default().vector,
        tags: sample.tags.unwrap_or_default(),
    };

    if samples_tx.send(Some(pulled_sample)).is_err() {
        // Channel is closed, wrapping up
        return Err(());
    }
    Ok(())
}

async fn async_pull_samples(
    client: &Arc<SharedClient>,
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
    limit: usize,
) {
    // TODO: Join with the other workers' implementation, same logic

    // We use async-await here, to better use IO stalls
    // We'll keep a pool of N async tasks in parallel
    let max_tasks = min(num_cpus::get(), limit);
    debug!("Using {} tasks in the async threadpool", max_tasks);
    let mut tasks = std::collections::VecDeque::new();
    let mut count = 0;
    let shareable_channel_tx: Arc<kanal::Sender<Option<Sample>>> = Arc::new(samples_tx);
    let shareable_img_tfm = Arc::new(image_transform);

    while let Ok(received) = samples_meta_rx.recv() {
        if received == serde_json::Value::Null {
            debug!("http_worker: end of stream received, stopping there");
            let _ = samples_meta_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.push_back(tokio::spawn(pull_sample(
            client.clone(),
            received,
            shareable_img_tfm.clone(),
            encode_images,
            img_to_rgb8,
            shareable_channel_tx.clone(),
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
    debug!("http_worker: total samples sent: {}\n", count);

    // Signal the end of the stream
    if shareable_channel_tx.send(None).is_ok() {} // Channel could have been closed by a .stop() call
}

pub fn pull_samples(
    client: &Arc<SharedClient>,
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
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
                client,
                samples_meta_rx,
                samples_tx,
                image_transform,
                encode_images,
                img_to_rgb8,
                limit,
            )
            .await;
        });
}
