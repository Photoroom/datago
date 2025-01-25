use crate::image_processing;
use pyo3::pyclass;
use pyo3::pymethods;
use serde::Deserialize;
use serde::Serialize;
use std::io::Cursor;
use std::sync::Arc;

// We'll share a single connection pool across all worker threads
#[derive(Clone)]
pub struct SharedClient {
    pub client: reqwest::blocking::Client,
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

// ------------------------------------------------------------------
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentPayload {
    #[pyo3(get, set)]
    data: Vec<u8>,
    #[pyo3(get, set)]
    len: usize,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImagePayload {
    #[pyo3(get, set)]
    pub data: Vec<u8>,
    #[pyo3(get, set)]
    pub original_height: usize, // Good indicator of the image frequency dbResponse at the current resolution
    #[pyo3(get, set)]
    pub original_width: usize,
    #[pyo3(get, set)]
    pub height: usize, // Useful to decode the current payload
    #[pyo3(get, set)]
    pub width: usize,
    #[pyo3(get, set)]
    pub channels: i8,
    #[pyo3(get, set)]
    pub bit_depth: usize,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    #[pyo3(get, set)]
    pub id: String,

    #[pyo3(get, set)]
    pub source: String,

    #[doc(hidden)]
    pub attributes: std::collections::HashMap<String, serde_json::Value>,

    #[pyo3(get, set)]
    pub duplicate_state: i32,

    #[pyo3(get, set)]
    pub image: ImagePayload,

    #[pyo3(get, set)]
    pub masks: std::collections::HashMap<String, ImagePayload>,

    #[pyo3(get, set)]
    pub additional_images: std::collections::HashMap<String, ImagePayload>,

    #[pyo3(get, set)]
    pub latents: std::collections::HashMap<String, LatentPayload>,

    #[pyo3(get, set)]
    pub coca_embedding: Vec<f32>,

    #[pyo3(get, set)]
    pub tags: Vec<String>,
}

#[pymethods]
impl Sample {
    #[getter]
    pub fn attributes(&self) -> String {
        serde_json::to_string(&self.attributes).unwrap()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CocaEmbedding {
    vector: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UrlLatent {
    file_direct_url: String,
    latent_type: String,
    is_mask: bool,
}

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

fn bytes_from_url(shared_client: &SharedClient, url: &str) -> Vec<u8> {
    let retries = 5;
    let timeout = std::time::Duration::from_secs(30);
    for _ in 0..retries {
        let permit = shared_client.semaphore.acquire();
        let response = shared_client.client.get(url).timeout(timeout).send();
        drop(permit);

        match response {
            Ok(response) => {
                let bytes = response.bytes();
                if let Ok(bytes) = bytes {
                    return bytes.to_vec();
                }
                // Try again
            }
            Err(e) => {
                println!("Failed to get response from URL: {}", url);
                println!("Error: {:?}", e);
            }
        }
    }

    vec![]
}

fn image_from_url(
    client: &SharedClient,
    url: &str,
) -> Result<image::DynamicImage, image::ImageError> {
    let bytes = bytes_from_url(client, url);

    // Catch an error down the stack, couldn´t pull the bytes in the first place
    if bytes.is_empty() {
        return Err(image::ImageError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Empty bytes",
        )));
    }

    // Image loading, error handling. Things can happen here too
    image::load_from_memory(&bytes)
}

fn image_payload_from_url(
    client: &SharedClient,
    url: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    aspect_ratio: &String,
    encode_images: bool,
) -> Result<ImagePayload, image::ImageError> {
    match image_from_url(client, url) {
        Ok(mut new_image) => {
            let original_height = new_image.height() as usize;
            let original_width = new_image.width() as usize;
            let mut channels = new_image.color().channel_count() as i8;
            let bit_depth = new_image.color().bits_per_pixel() as usize;

            // Optionally transform the additional image in the same way the main image was
            if let Some(img_tfm) = img_tfm {
                new_image = img_tfm.crop_and_resize(&new_image, aspect_ratio);
            }

            let height = new_image.height() as usize;
            let width = new_image.width() as usize;

            // Encode the image if needed
            let mut image_bytes: Vec<u8> = Vec::new();
            if encode_images {
                // TODO: Handle masks and pngs

                // Force PNG for now
                new_image
                    .write_to(&mut Cursor::new(&mut image_bytes), image::ImageFormat::Png)
                    .unwrap();

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
    client: &SharedClient,
    sample_json: serde_json::Value,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
) -> Option<Sample> {
    // TODO: Make this whole function async

    // Deserialize the sample metadata
    let sample: SampleMetadata = serde_json::from_value(sample_json).unwrap();

    // Pull the image for a start, get an idea of the speed
    let mut image_payload: Option<ImagePayload> = None;
    let mut aspect_ratio = String::new();

    if let Some(image_url) = &sample.image_direct_url {
        image_payload =
            match image_payload_from_url(client, image_url, img_tfm, &String::new(), encode_images)
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
                    return None;
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
                let additional_image_payload = image_payload_from_url(
                    client,
                    &latent.file_direct_url,
                    img_tfm,
                    &aspect_ratio,
                    encode_images,
                );
                if additional_image_payload.is_err() {
                    println!(
                        "Failed to get additional image from URL: {} {}",
                        latent.latent_type, latent.file_direct_url
                    );
                    return None;
                }

                additional_images.insert(
                    latent.latent_type.clone(),
                    additional_image_payload.unwrap(),
                );
            } else if latent.is_mask {
                // Mask types, registered as latents but they need to be png-decoded
                let mask_payload = image_payload_from_url(
                    client,
                    &latent.file_direct_url,
                    img_tfm,
                    &aspect_ratio,
                    encode_images,
                );
                if mask_payload.is_err() {
                    println!("Failed to get mask from URL: {}", latent.file_direct_url);
                    return None;
                }

                masks.insert(latent.latent_type.clone(), mask_payload.unwrap());
            } else {
                // Vanilla latents, pure binary payloads
                let latent_payload = bytes_from_url(client, &latent.file_direct_url);
                if !latent_payload.is_empty() {
                    latents.insert(
                        latent.latent_type.clone(),
                        LatentPayload {
                            data: latent_payload.clone(),
                            len: latent_payload.len(),
                        },
                    );
                } else {
                    println!("Error fetching latent: {}", latent.file_direct_url);
                    return None;
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
        coca_embedding: Vec::<f32>::new(), //sample.coca_embedding.vector, // FIXME
        tags: sample.tags.unwrap_or_default(),
    };
    Some(pulled_sample)
}

pub fn pull_samples(
    client: SharedClient,
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

        if let Some(sample) = pull_sample(&client, received, image_transform, encode_images) {
            if !samples_tx.send(sample).is_ok() {
                println!("http_worker: failed to send a sample");
                break;
            }
        } else {
            break;
        }
    }

    worker_done_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}
