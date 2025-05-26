use crate::image_processing;
use crate::structs::{ImagePayload, Sample, TarballContent};
use log::{debug, info, warn};
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

async fn pull_sample(
    sample: TarballContent,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encode_images: bool,
    img_to_rgb8: bool,
    samples_tx: Arc<kanal::Sender<Option<Sample>>>,
) -> Result<(), ()> {
    if !sample.is_empty() {
        match Path::new(&sample[0].filename).file_stem() {
            Some(sample_id) => {
                let mut attributes = HashMap::new();
                let mut image = ImagePayload {
                    data: vec![],
                    width: 0,
                    height: 0,
                    original_height: 0,
                    original_width: 0,
                    bit_depth: 0,
                    channels: 0,
                };

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
                                image = image_processing::image_to_payload(
                                    raw_image,
                                    &img_tfm,
                                    &"".to_string(),
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

                if samples_tx
                    .send(Some(Sample {
                        id: String::from(sample_id.to_str().unwrap_or("unkonwn")),
                        source: "webdataset".to_string(), // FIXME - pass the archive name
                        image,
                        attributes,
                        coca_embedding: vec![],
                        tags: vec![],
                        masks: HashMap::new(),
                        latents: HashMap::new(),
                        additional_images: HashMap::new(),
                        duplicate_state: 0,
                    }))
                    .is_err()
                {
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
    samples_metadata_rx: kanal::Receiver<TarballContent>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
    limit: usize,
) {
    // We use async-await here, to better use IO stalls
    // We'll keep a pool of N async tasks in parallel
    let max_tasks = min(num_cpus::get(), limit);
    info!("Using {} tasks in the async threadpool", max_tasks);
    let mut tasks = tokio::task::JoinSet::new();
    let mut count = 0;
    let shareable_channel_tx: Arc<kanal::Sender<Option<Sample>>> = Arc::new(samples_tx);
    let shareable_img_tfm = Arc::new(image_transform);

    while let Ok(sample) = samples_metadata_rx.recv() {
        if sample.is_empty() {
            warn!("wds_worker: end of stream received, stopping there");
            let _ = samples_metadata_rx.close();
            break;
        }

        // Append a new task to the queue
        tasks.spawn(pull_sample(
            sample,
            shareable_img_tfm.clone(),
            encode_images,
            img_to_rgb8,
            shareable_channel_tx.clone(),
        ));

        // If we have enough tasks, we'll wait for the older one to finish
        if tasks.len() >= max_tasks {
            if tasks.join_next().await.unwrap().is_ok() {
                count += 1;
            } else {
                warn!("wds_worker: task failed, stopping there");
                let _ = samples_metadata_rx.close(); // Stop upstream thread
                break;
            }

            if count >= limit {
                break;
            }
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

    info!("wds_worker: total samples sent: {}\n", count);

    // Signal the end of the stream
    if shareable_channel_tx.send(None).is_ok() {} // Channel could have been closed by a .stop() call
}

pub fn deserialize_samples(
    samples_metadata_rx: kanal::Receiver<TarballContent>,
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
            async_deserialize_samples(
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
