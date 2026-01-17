use crate::image_processing;
use crate::structs::{
    to_python_image_payload, to_python_image_payload_map, CocaEmbedding, ImagePayload,
    LatentPayload, Sample, SharedClient, UrlLatent,
};
use log::{debug, error, warn};
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinError;
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

pub async fn bytes_from_url(
    shared_client: &SharedClient,
    url: &str,
    auth_token: Option<&str>,
) -> Option<Vec<u8>> {
    // Retry on the request a few times
    let timeout = std::time::Duration::from_secs(120);
    let _permit = shared_client.semaphore.acquire().await;

    // Get a client reference with optimized settings
    let client = &shared_client.client;

    // Send request with specific timeout and connection settings
    let mut request = client
        .get(url)
        .timeout(timeout)
        .header(reqwest::header::CONNECTION, "keep-alive");

    if let Some(token) = auth_token {
        request = request.header(reqwest::header::AUTHORIZATION, format!("Bearer {}", token));
    }

    if let Ok(response) = request.send().await {
        if let Ok(bytes) = response.bytes().await {
            return Some(bytes.to_vec());
        }
    }

    None
}

async fn image_from_url(
    client: &SharedClient,
    url: &str,
    retries: u8,
) -> Result<image::DynamicImage, image::ImageError> {
    // Retry on the fetch and decode a few times, could happen that we get a broken packet
    for _ in 0..retries {
        if let Some(bytes) = bytes_from_url(client, url, None).await {
            match image::load_from_memory(&bytes) {
                Ok(image) => return Ok(image),
                Err(e) => {
                    warn!("Failed to decode image from URL: {url}. Retrying");
                    warn!("Error: {e:?}");
                }
            }
        }
    }
    Err(image::ImageError::IoError(std::io::Error::other(
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
        match bytes_from_url(client, url, None).await {
            Some(bytes) => {
                return Ok(bytes);
            }
            None => {
                warn!("Failed to get bytes from URL: {url}. Retrying");
            }
        }
    }
    Err(std::io::Error::other("Failed to fetch bytes buffer"))
}

async fn image_payload_from_url(
    client: &SharedClient,
    url: &str,
    img_tfm: &Option<image_processing::ARAwareTransform>,
    aspect_ratio: &String,
    retries: u8,
    encoding: image_processing::ImageEncoding,
) -> Result<ImagePayload, image::ImageError> {
    match image_from_url(client, url, retries).await {
        Ok(new_image) => {
            image_processing::image_to_payload(new_image, img_tfm, aspect_ratio, encoding).await
        }
        Err(e) => Err(e),
    }
}

async fn pull_sample(
    client: Arc<SharedClient>,
    sample_json: serde_json::Value,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    retries: u8,
    encoding: image_processing::ImageEncoding,
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
            retries,
            encoding,
        )
        .await
        {
            Ok(payload) => {
                aspect_ratio = image_processing::aspect_ratio_to_str((
                    payload.width as u32,
                    payload.height as u32,
                ));
                Some(payload)
            }
            Err(e) => {
                error!("Failed to get image from URL: {image_url}\n {e:?}");
                error!("Error: {e:?}");
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
                    retries,
                    encoding,
                )
                .await
                {
                    Ok(additional_image_payload) => {
                        additional_images
                            .insert(latent.latent_type.clone(), additional_image_payload);
                    }

                    Err(e) => {
                        warn!(
                            "Failed to get additional image from URL: {} {} {:?}",
                            latent.latent_type, latent.file_direct_url, e
                        );
                        return Err(());
                    }
                }
            } else if latent.is_mask {
                // Mask types, registered as latents but they need to be png-decoded
                let mask_encoding = image_processing::ImageEncoding {
                    img_to_rgb8: false, // Masks are not converted to RGB8
                    encode_format: image_processing::EncodeFormat::Png, // Masks are always PNGs, never JPEGs
                    ..encoding
                };
                match image_payload_from_url(
                    &client,
                    &latent.file_direct_url,
                    &img_tfm,
                    &aspect_ratio,
                    retries,
                    mask_encoding,
                )
                .await
                {
                    Ok(mask_payload) => {
                        masks.insert(latent.latent_type.clone(), mask_payload);
                    }

                    Err(e) => {
                        warn!(
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
        image: to_python_image_payload(image_payload.unwrap_or(ImagePayload {
            data: Vec::new(),
            original_height: 0,
            original_width: 0,
            height: 0,
            width: 0,
            channels: 0,
            bit_depth: 0,
            is_encoded: false,
        })),
        masks: to_python_image_payload_map(masks),
        additional_images: to_python_image_payload_map(additional_images),
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
    encoding: image_processing::ImageEncoding,
    limit: usize,
) -> Result<(), String> {
    // We use async-await here, to better use IO stalls
    // We'll keep a pool of N async tasks in parallel
    let default_max_tasks = std::env::var("DATAGO_MAX_TASKS")
        .unwrap_or_else(|_| "0".to_string())
        .parse::<usize>()
        .unwrap_or(num_cpus::get() * 4);

    let max_retries = std::env::var("DATAGO_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(3);

    let max_tasks = min(default_max_tasks, limit);
    debug!("Using {max_tasks} tasks in the async threadpool");
    let mut tasks = tokio::task::JoinSet::new();
    let mut count = 0;
    let shareable_channel_tx: Arc<kanal::Sender<Option<Sample>>> = Arc::new(samples_tx);
    let shareable_img_tfm = Arc::new(image_transform);
    let mut join_error: Option<JoinError> = None;

    while let Ok(received) = samples_meta_rx.recv() {
        if received == serde_json::Value::Null {
            debug!("http_worker: end of stream received, stopping there");
            let _ = samples_meta_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.spawn(pull_sample(
            client.clone(),
            received,
            shareable_img_tfm.clone(),
            max_retries,
            encoding,
            shareable_channel_tx.clone(),
        ));

        // If we have enough tasks, we'll wait for the older one to finish
        if tasks.len() >= max_tasks {
            match tasks.join_next().await {
                Some(Ok(_)) => {
                    count += 1;
                }
                Some(Err(e)) => {
                    // Task failed, log the error
                    error!("file_worker: task failed with error: {e}");
                    join_error = Some(e);
                    break;
                }
                None => {
                    // Task was cancelled or panicked
                    error!("file_worker: task was cancelled or panicked");
                }
            }
        }
        if count >= limit {
            break;
        }
    }

    // Make sure to wait for all the remaining tasks
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(_) => {
                debug!("dispatch_shards: task completed successfully");
                count += 1;
            }
            Err(e) => {
                error!("dispatch_shards: task failed with error: {e}");
                if join_error.is_none() {
                    join_error = Some(e);
                }
            }
        }
    }

    debug!("http_worker: total samples sent: {count}\n");

    // Signal the end of the stream
    let _ = shareable_channel_tx.send(None); // Channel could have been closed by a .stop() call

    if let Some(e) = join_error {
        // If we had an error, return it
        return Err(e.to_string());
    }
    Ok(())
}

pub fn pull_samples(
    client: &Arc<SharedClient>,
    samples_meta_rx: kanal::Receiver<serde_json::Value>,
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
            match async_pull_samples(
                client,
                samples_meta_rx,
                samples_tx,
                image_transform,
                encoding,
                limit,
            )
            .await
            {
                Ok(_) => {
                    debug!("http_worker: all samples pulled successfully");
                }
                Err(e) => {
                    error!("http_worker: error pulling samples: {e}");
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::new_shared_client;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::time::timeout;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_semaphore_connection_limiting() {
        // Create a shared client with only 2 max connections
        let shared_client = Arc::new(new_shared_client(2));

        // Create a mock server that simulates a slow API response
        let server = MockServer::start().await;
        let _mock = Mock::given(method("GET"))
            .and(path("/test"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("test response")
                    .set_delay(Duration::from_millis(100)),
            ) // Simulate 100ms runtime of the request
            .expect(3)
            .mount(&server)
            .await;

        let url = format!("{}/test", server.uri());

        // Track when each request starts and completes
        let mut handles = Vec::new();

        // Spawn 3 tasks to test limiting using the actual bytes_from_url function
        for i in 0..3 {
            let client = shared_client.clone();
            let url = url.clone();
            let handle = tokio::spawn(async move {
                let request_start = Instant::now();

                let result = bytes_from_url(&client, &url, None).await;
                let request_end = Instant::now();

                (i, result.is_some(), request_start, request_end)
            });
            handles.push(handle);
        }

        // Wait for all requests to complete with timeout
        let result = timeout(Duration::from_secs(5), async {
            let mut results = Vec::new();
            for handle in handles {
                if let Ok((i, success, req_start, req_end)) = handle.await {
                    results.push((i, success, req_start, req_end));
                }
            }
            results
        })
        .await;

        assert!(
            result.is_ok(),
            "Test timed out - semaphore may not be working correctly"
        );
        let results = result.unwrap();

        // All requests should succeed
        for (i, success, _, _) in &results {
            assert!(success, "Request {} failed", i);
        }

        // Verify we made the expected number of requests
        assert_eq!(results.len(), 3);

        // Sort results by completion time to analyze concurrency
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| a.3.cmp(&b.3));

        // Calculate the timing to verify semaphore limiting
        let first_request_start = sorted_results[0].2;
        let first_request_end = sorted_results[0].3;
        let first_request_duration = first_request_end.duration_since(first_request_start);
        let second_request_start = sorted_results[1].2;
        let second_request_end = sorted_results[1].3;
        let second_request_duration = second_request_end.duration_since(second_request_start);
        let last_request_start = sorted_results[2].2;
        let last_request_end = sorted_results[2].3;
        let last_request_duration = last_request_end.duration_since(last_request_start);

        // The first 2 requests should complete after ~100ms (server delay)
        // The last request should complete after ~200ms (waits for first two to finish)
        assert!(first_request_duration >= Duration::from_millis(100) && first_request_duration <= Duration::from_millis(200),
            "First request completed too early (duration: {:?}), semaphore may not be limiting connections properly",
            first_request_duration);
        assert!(second_request_duration >= Duration::from_millis(100) && second_request_duration <= Duration::from_millis(200),
            "Second request completed too early (duration: {:?}), semaphore may not be limiting connections properly",
            second_request_duration);
        assert!(last_request_duration >= Duration::from_millis(200) && last_request_duration <= Duration::from_millis(300),
            "Last request completed too early (duration: {:?}), semaphore may not be limiting connections properly",
            last_request_duration);

        // Verify the semaphore is back to full capacity
        assert_eq!(shared_client.semaphore.available_permits(), 2);
    }
}
