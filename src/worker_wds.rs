use crate::image_processing;
use crate::structs::{ImagePayload, Sample, TarballSample};
use log::{debug, error, info, warn};
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
    encode_images: bool,
    img_to_rgb8: bool,
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
                        // Load the image in to a buffer
                        match image::load_from_memory(&item.buffer) {
                            Ok(raw_image) => {
                                let image = image_processing::image_to_payload(
                                    raw_image,
                                    &img_tfm,
                                    &sample_aspect_ratio,
                                    encode_images,
                                    img_to_rgb8,
                                )
                                .await
                                .unwrap_or_else(|_| ImagePayload {
                                    data: vec![],
                                    width: 0,
                                    height: 0,
                                    original_height: 0,
                                    original_width: 0,
                                    bit_depth: 0,
                                    channels: 0,
                                });

                                if sample_aspect_ratio.is_empty() {
                                    // If we don't have an aspect ratio yet, we set it
                                    sample_aspect_ratio = image_processing::aspect_ratio_to_str((
                                        image.width as u32,
                                        image.height as u32,
                                    ));
                                }

                                if ext == extension_reference_image {
                                    // If this is the reference image, we store it in the main image field
                                    final_sample = Some(Sample {
                                        id: String::from(sample_id.to_str().unwrap_or("unknown")),
                                        source: sample.name.clone(),
                                        image: image.clone(),
                                        attributes: attributes.clone(),
                                        coca_embedding: vec![],
                                        tags: vec![],
                                        masks: HashMap::new(),
                                        latents: HashMap::new(),
                                        additional_images: HashMap::new(),
                                        duplicate_state: 0,
                                    });
                                } else {
                                    // Otherwise, we store it in the additional images
                                    match final_sample {
                                        Some(ref mut final_sample_ref) => {
                                            final_sample_ref
                                                .additional_images
                                                .insert(item.filename.clone(), image.clone());
                                        }
                                        None => {
                                            // If final_sample is not initialized, we create it
                                            panic!( "Final sample should be initialized before adding additional images");
                                        }
                                    }
                                }
                                debug!("wds_worker: unpacked {}", item.filename);
                            }
                            Err(e) => {
                                debug!("wds_worker: error loading image: {}", e);
                                continue;
                            }
                        }
                    } else if TEXT_TYPES.contains(&ext) {
                        // Load the file in to a string
                        let class_file = String::from_utf8_lossy(&item.buffer).to_string();
                        attributes.insert(ext.to_string(), serde_json::json!(class_file));
                        debug!("wds_worker: unpacked {}", item.filename);
                    }
                }

                if samples_tx.send(final_sample).is_err() {
                    debug!("wds_worker: stream already closed, wrapping up");
                    return Err(());
                }
                return Ok(());
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
    encode_images: bool,
    img_to_rgb8: bool,
    limit: usize,
    extension_reference_image: String,
) -> Result<(), String> {
    // We use async-await here, to better use IO stalls
    // We'll keep a pool of N async tasks in parallel
    let max_tasks = min(num_cpus::get(), limit);
    info!("Using {} tasks in the async threadpool", max_tasks);
    let mut tasks = tokio::task::JoinSet::new();
    let mut count = 0;
    let shareable_channel_tx: Arc<kanal::Sender<Option<Sample>>> = Arc::new(samples_tx);
    let shareable_img_tfm = Arc::new(image_transform);
    let mut join_error: Option<String> = None;

    while let Ok(sample) = samples_metadata_rx.recv() {
        if sample.is_empty() {
            warn!("wds_worker: end of stream received, stopping there");
            let _ = samples_metadata_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.spawn(process_sample(
            sample,
            shareable_img_tfm.clone(),
            encode_images,
            img_to_rgb8,
            shareable_channel_tx.clone(),
            extension_reference_image.clone(),
        ));

        // If we have enough tasks, we'll wait for the older one to finish
        if tasks.len() >= max_tasks {
            if let Some(result) = tasks.join_next().await {
                match result {
                    Ok(_) => count += 1,
                    Err(e) => {
                        join_error = Some(format!("Task failed: {}", e));
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
                error!("dispatch_shards: task failed with error: {:?}", e);
                if join_error.is_none() {
                    join_error = Some(e.to_string());
                }
            }
        }
    }

    info!("wds_worker: total samples sent: {}\n", count);

    // Signal the end of the stream
    let _ = shareable_channel_tx.send(None); // Channel could have been closed by a .stop() call

    if let Some(error) = join_error {
        error!(
            "wds_worker: encountered an error while processing samples: {}",
            error
        );
        return Err(error);
    }
    Ok(())
}

pub fn deserialize_samples(
    samples_metadata_rx: kanal::Receiver<TarballSample>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
    limit: usize,
    extension_reference_image: String,
) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            match async_deserialize_samples(
                samples_metadata_rx,
                samples_tx,
                image_transform,
                encode_images,
                img_to_rgb8,
                limit,
                extension_reference_image,
            )
            .await
            {
                Ok(_) => debug!("wds_worker: all samples processed successfully"),
                Err(e) => error!("wds_worker: error processing samples : {:?}", e),
            }
        });
}
