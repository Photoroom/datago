use crate::image_processing;
use crate::structs::{CocaEmbedding, ImagePayload, LatentPayload, Sample, UrlLatent};
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::io::Cursor;
use std::sync::Arc;

// We'll share a single connection pool across all worker threads
#[derive(Clone)]
pub struct SharedClient {
    pub client: reqwest::Client,
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

// ------------------------------------------------------------------
#[derive(Debug, Serialize, Deserialize)]
struct SampleMetadata {
    id: String,
    source: String,
    attributes: std::collections::HashMap<String, serde_json::Value>,
    duplicate_state: Option<i32>,
    image_direct_url: Option<String>,
    latents: Option<Vec<UrlLatent>>,
    tags: Option<Vec<String>>,
    coca_embedding: Option<CocaEmbedding>,
}

async fn bytes_from_url(shared_client: &SharedClient, url: &str) -> Option<Vec<u8>> {
    // Retry on the request a few times
    let retries = 5;
    let timeout = std::time::Duration::from_secs(30);
    for _ in 0..retries {
        let permit = shared_client.semaphore.acquire();

        match shared_client.client.get(url).timeout(timeout).send().await {
            Ok(response) => {
                if let Ok(bytes) = response.bytes().await {
                    return Some(bytes.to_vec());
                }
            }
            Err(e) => {
                println!("Failed to fetch image bytes: {}", e);
            }
        }
        drop(permit);
    }
    None
}

async fn image_from_url(
    client: &SharedClient,
    url: &str,
) -> Result<image::DynamicImage, image::ImageError> {
    // Retry on the fetch and decode a few times, could happen that we get a broken packet
    let retries = 5;
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

async fn image_payload_from_url(
    client: &SharedClient,
    url: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    aspect_ratio: &String,
    encode_images: bool,
) -> Result<ImagePayload, image::ImageError> {
    match image_from_url(client, url).await {
        Ok(mut new_image) => {
            let original_height = new_image.height() as usize;
            let original_width = new_image.width() as usize;
            let mut channels = new_image.color().channel_count() as i8;
            let bit_depth = (new_image.color().bits_per_pixel()
                / new_image.color().channel_count() as u16) as usize;

            // Optionally transform the additional image in the same way the main image was
            if let Some(img_tfm) = img_tfm {
                let aspect_ratio_input = if aspect_ratio.is_empty() {
                    None
                } else {
                    Some(aspect_ratio)
                };

                // TODO: tokio::spawn this
                new_image = img_tfm
                    .crop_and_resize(&new_image, aspect_ratio_input)
                    .await;
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

async fn pull_sample(
    client: SharedClient,
    sample_json: serde_json::Value,
    img_tfm: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    samples_tx: kanal::Sender<Option<Sample>>,
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
                println!("Failed to get image from URL: {}", image_url);
                println!("Error: {:?}", e);
                return Err(());
            }
        };
    }

    // Same for the latents, mask and masked images, if they exist
    let mut masks: std::collections::HashMap<String, ImagePayload> =
        std::collections::HashMap::new();
    let mut additional_images: std::collections::HashMap<String, ImagePayload> =
        std::collections::HashMap::new();
    let mut latents: std::collections::HashMap<String, LatentPayload> =
        std::collections::HashMap::new();

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
                match bytes_from_url(&client, &latent.file_direct_url).await {
                    Some(latent_payload) => {
                        latents.insert(
                            latent.latent_type.clone(),
                            LatentPayload {
                                len: latent_payload.len(),
                                data: latent_payload,
                            },
                        );
                    }
                    None => {
                        println!("Error fetching latent: {}", latent.file_direct_url);
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
    client: SharedClient,
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    limit: usize,
) {
    // We use async-await here, to better use IO stalls
    // We'll keep a pool of N async tasks in parallel
    let max_tasks_per_thread = min(num_cpus::get() * 2, limit);
    let mut tasks = std::collections::VecDeque::new();
    let mut count = 0;
    while let Ok(received) = samples_meta_rx.recv() {
        if received == serde_json::Value::Null {
            println!("http_worker: end of stream received, stopping there");
            samples_meta_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.push_back(tokio::spawn(pull_sample(
            client.clone(),
            received,
            image_transform.clone(),
            encode_images,
            samples_tx.clone(),
        )));

        // If we have enough tasks, we'll wait for the older one to finish
        if tasks.len() >= max_tasks_per_thread {
            match tasks.pop_front().unwrap().await {
                Ok(_) => {
                    count += 1;
                }
                Err(e) => {
                    println!("worker: sample skipped {}", e);
                }
            }
        }
        if count >= limit {
            break;
        }
    }

    // Make sure to wait for all the remaining tasks
    while !tasks.is_empty() {
        match tasks.pop_front().unwrap().await {
            Ok(_) => {
                count += 1;
            }
            Err(e) => {
                println!("worker: sample skipped {}", e);
            }
        }
    }
    println!("http_worker: total samples sent: {}\n", count);

    // Signal the end of the stream
    if samples_tx.send(None).is_ok() {} // Channel could have been closed by a .stop() call
}

pub fn pull_samples(
    client: SharedClient,
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
                client,
                samples_meta_rx,
                samples_tx,
                image_transform,
                encode_images,
                limit,
            )
            .await;
        });
}
