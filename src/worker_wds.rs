use crate::image_processing;
use crate::structs::{to_python_image_payload, ImagePayload, Sample, TarballSample};
use log::{debug, error, info};
use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// List all the types we'll support
pub const TEXT_TYPES: [&str; 3] = ["cls", "json", "txt"];
pub const IMG_TYPES: [&str; 3] = ["jpg", "jpeg", "png"];

fn is_supported_type(ext: &str) -> bool {
    let lowered_ext = ext.to_lowercase(); // Garding against case sensitivity
    let ext = lowered_ext.as_str();
    TEXT_TYPES.contains(&ext) || IMG_TYPES.contains(&ext)
}

async fn process_sample(
    sample: TarballSample,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encoding: image_processing::ImageEncoding,
    samples_tx: Arc<kanal::Sender<Option<Sample>>>,
    extension_reference_image: String,
) -> Result<(), ()> {
    if !sample.is_empty() {
        match Path::new(&sample.content[0].filename).file_stem() {
            Some(sample_id) => {
                let mut attributes = HashMap::new();

                // We'll build a single Sample for all the items
                let mut final_sample: Option<Sample> = None;
                let mut sample_aspect_ratio = "".to_string();

                for item in sample.iter() {
                    let ext = Path::new(&item.filename)
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");

                    if !is_supported_type(ext) {
                        debug!("wds_worker: unsupported file type: {}", item.filename);
                    } else if IMG_TYPES.contains(&ext) {
                        // Use spawn_blocking for CPU-bound image decoding and into_bytes()
                        // This allows other async tasks (like network I/O) to make progress
                        // while the CPU-intensive decoding + pixel data extraction happens on a blocking thread pool
                        let buffer = item.buffer.clone();
                        let has_transform = img_tfm.is_some();
                        let needs_encoding = encoding.encode_images || encoding.img_to_rgb8;
                        
                        if !has_transform && !needs_encoding {
                            // Fast path: no transform, no encoding - do decode + into_bytes sync
                            let payload_result = tokio::task::spawn_blocking(move || {
                                let raw_image = image::load_from_memory(&buffer)?;
                                let height = raw_image.height() as usize;
                                let width = raw_image.width() as usize;
                                let channels = raw_image.color().channel_count() as i8;
                                let bit_depth = (raw_image.color().bits_per_pixel() / raw_image.color().channel_count() as u16) as usize;
                                let image_bytes = raw_image.into_bytes();
                                Ok::<_, image::ImageError>(ImagePayload {
                                    data: image_bytes,
                                    original_height: height,
                                    original_width: width,
                                    height,
                                    width,
                                    channels,
                                    bit_depth,
                                    is_encoded: false,
                                })
                            })
                            .await;
                            
                            match payload_result {
                                Ok(Ok(payload)) => {
                                    let image = to_python_image_payload(payload);
                                    
                                    if sample_aspect_ratio.is_empty() {
                                        let pl = image.get_payload();
                                        sample_aspect_ratio = image_processing::aspect_ratio_to_str((
                                            pl.width as u32,
                                            pl.height as u32,
                                        ));
                                    }

                                    if ext == extension_reference_image {
                                        if let Some(ref mut fs) = final_sample {
                                            fs.image = image;
                                        } else {
                                            final_sample = Some(Sample {
                                                id: String::from(sample_id.to_str().unwrap_or("unknown")),
                                                source: sample.name.clone(),
                                                image,
                                                attributes: HashMap::new(),
                                                coca_embedding: vec![],
                                                tags: vec![],
                                                masks: HashMap::new(),
                                                latents: HashMap::new(),
                                                additional_images: HashMap::new(),
                                                duplicate_state: 0,
                                            });
                                        }
                                    } else {
                                        if let Some(ref mut fs) = final_sample {
                                            fs.additional_images.insert(item.filename.clone(), image);
                                        } else {
                                            final_sample = Some(Sample {
                                                id: String::from(sample_id.to_str().unwrap_or("unknown")),
                                                source: sample.name.clone(),
                                                image: to_python_image_payload(ImagePayload {
                                                    data: vec![],
                                                    width: 0,
                                                    height: 0,
                                                    original_height: 0,
                                                    original_width: 0,
                                                    bit_depth: 0,
                                                    channels: 0,
                                                    is_encoded: false,
                                                }),
                                                attributes: HashMap::new(),
                                                coca_embedding: vec![],
                                                tags: vec![],
                                                masks: HashMap::new(),
                                                latents: HashMap::new(),
                                                additional_images: HashMap::new(),
                                                duplicate_state: 0,
                                            });
                                        }
                                    }
                                    debug!("wds_worker: unpacked {}", item.filename);
                                }
                                Ok(Err(e)) => {
                                    debug!("wds_worker: error in fast path: {}", e);
                                    continue;
                                }
                                Err(e) => {
                                    debug!("wds_worker: spawn_blocking failed: {}", e);
                                    continue;
                                }
                            }
                        } else {
                            // Slow path: has transform or needs encoding - use original async code
                            let decoded_result = tokio::task::spawn_blocking(move || {
                                image::load_from_memory(&buffer)
                            })
                            .await;
                            
                            match decoded_result {
                                Ok(Ok(raw_image)) => {
                                    let image = image_processing::image_to_payload(
                                        raw_image,
                                        &img_tfm,
                                        &sample_aspect_ratio,
                                        encoding,
                                    )
                                    .await
                                    .map(to_python_image_payload)
                                    .unwrap_or_else(|_| {
                                        to_python_image_payload(ImagePayload {
                                            data: vec![],
                                            width: 0,
                                            height: 0,
                                            original_height: 0,
                                            original_width: 0,
                                            bit_depth: 0,
                                            channels: 0,
                                            is_encoded: false,
                                        })
                                    });

                                    if sample_aspect_ratio.is_empty() {
                                        let payload = image.get_payload();
                                        sample_aspect_ratio = image_processing::aspect_ratio_to_str((
                                            payload.width as u32,
                                            payload.height as u32,
                                        ));
                                    }

                                    if ext == extension_reference_image {
                                        if let Some(ref mut fs) = final_sample {
                                            fs.image = image;
                                        } else {
                                            final_sample = Some(Sample {
                                                id: String::from(sample_id.to_str().unwrap_or("unknown")),
                                                source: sample.name.clone(),
                                                image,
                                                attributes: HashMap::new(),
                                                coca_embedding: vec![],
                                                tags: vec![],
                                                masks: HashMap::new(),
                                                latents: HashMap::new(),
                                                additional_images: HashMap::new(),
                                                duplicate_state: 0,
                                            });
                                        }
                                    } else {
                                        if let Some(ref mut fs) = final_sample {
                                            fs.additional_images.insert(item.filename.clone(), image);
                                        } else {
                                            final_sample = Some(Sample {
                                                id: String::from(sample_id.to_str().unwrap_or("unknown")),
                                                source: sample.name.clone(),
                                                image: to_python_image_payload(ImagePayload {
                                                    data: vec![],
                                                    width: 0,
                                                    height: 0,
                                                    original_height: 0,
                                                    original_width: 0,
                                                    bit_depth: 0,
                                                    channels: 0,
                                                    is_encoded: false,
                                                }),
                                                attributes: HashMap::new(),
                                                coca_embedding: vec![],
                                                tags: vec![],
                                                masks: HashMap::new(),
                                                latents: HashMap::new(),
                                                additional_images: HashMap::new(),
                                                duplicate_state: 0,
                                            });
                                        }
                                    }
                                    debug!("wds_worker: unpacked {}", item.filename);
                                }
                                Ok(Err(e)) => {
                                    debug!("wds_worker: error loading image: {}", e);
                                    continue;
                                }
                                Err(e) => {
                                    debug!("wds_worker: spawn_blocking failed: {}", e);
                                    continue;
                                }
                            }
                        }
                    } else if TEXT_TYPES.contains(&ext) {
                        // Load the file in to a string
                        let class_file = String::from_utf8_lossy(&item.buffer).to_string();
                        attributes.insert(ext.to_string(), serde_json::json!(class_file));
                        debug!("wds_worker: unpacked {} {}", item.filename, class_file);
                    }
                }

                // Make sure that the sample has the attributes we decoded
                if let Some(ref mut final_sample_ref) = final_sample {
                    final_sample_ref.attributes = attributes;
                    match samples_tx.send(final_sample) {
                        Ok(_) => (),
                        Err(e) => {
                            if !samples_tx.is_closed() {
                                debug!("wds_worker: error dispatching sample: {e}");
                                return Err(());
                            }
                        }
                    }
                    return Ok(());
                }
                return Err(());
            }
            None => {
                debug!("wds_worker: unpacking sample with no ID");
                return Err(());
            }
        }
    }

    Err(())
}

async fn async_deserialize_samples(
    samples_metadata_rx: kanal::Receiver<TarballSample>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
    limit: usize,
    extension_reference_image: String,
) -> Result<(), String> {
    // We use async-await here, to better use IO stalls
    // We'll keep a pool of N async tasks in parallel
    let default_max_tasks = std::env::var("DATAGO_MAX_TASKS")
        .unwrap_or_else(|_| "0".to_string())
        .parse::<usize>()
        .unwrap_or(num_cpus::get());
    let max_tasks = min(num_cpus::get() * 4, default_max_tasks); // Ensure minimum of 8 processing tasks

    info!("WDS: Using {max_tasks} processing tasks in worker threadpool");
    let mut tasks = tokio::task::JoinSet::new();
    let mut count = 0;
    let shareable_channel_tx: Arc<kanal::Sender<Option<Sample>>> = Arc::new(samples_tx);
    let shareable_img_tfm = Arc::new(image_transform);
    let mut join_error: Option<String> = None;

    while let Ok(sample) = samples_metadata_rx.recv() {
        if sample.is_empty() {
            info!("wds_worker: end of stream received, stopping there");
            let _ = samples_metadata_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.spawn(process_sample(
            sample,
            shareable_img_tfm.clone(),
            encoding,
            shareable_channel_tx.clone(),
            extension_reference_image.clone(),
        ));

        // If we have enough tasks, we'll wait for the older one to finish
        if tasks.len() >= max_tasks {
            if let Some(result) = tasks.join_next().await {
                match result {
                    Ok(_) => count += 1,
                    Err(e) => {
                        join_error = Some(format!("Task failed: {e}"));
                        break;
                    }
                }
            }

            if count >= limit {
                break;
            }
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
                    join_error = Some(e.to_string());
                }
            }
        }
    }

    info!("wds_worker: total samples sent: {count}\n");

    // Signal the end of the stream
    let _ = shareable_channel_tx.send(None); // Channel could have been closed by a .stop() call

    if let Some(error) = join_error {
        error!("wds_worker: encountered an error while processing samples: {error}");
        return Err(error);
    }
    Ok(())
}

pub fn deserialize_samples(
    samples_metadata_rx: kanal::Receiver<TarballSample>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
    limit: usize,
    extension_reference_image: String,
) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get()) // Tasks in flight are limited by DATAGO_MAX_TASKS env
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            match async_deserialize_samples(
                samples_metadata_rx,
                samples_tx,
                image_transform,
                encoding,
                limit,
                extension_reference_image,
            )
            .await
            {
                Ok(_) => debug!("wds_worker: all samples processed successfully"),
                Err(e) => error!("wds_worker: error processing samples : {e}"),
            }
        });
}
