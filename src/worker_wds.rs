use crate::image_processing;
use crate::structs::{to_python_image_payload, ImagePayload, Sample, TarballSample};
use crate::worker_utils::{max_tasks_from_env, run_worker};
use log::{debug, error, info};
use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

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
    extension_reference_image: String,
) -> Result<Sample, ()> {
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
                                    encoding,
                                )
                                .await
                                .map(to_python_image_payload)
                                .unwrap_or_else(|_| {
                                    to_python_image_payload(ImagePayload {
                                        data: std::sync::Arc::from(Vec::<u8>::new()),
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
                                    // If we don't have an aspect ratio yet, we set it
                                    // We need to get the payload from the PythonImagePayload to access width/height
                                    let payload = image.get_payload();
                                    sample_aspect_ratio = image_processing::aspect_ratio_to_str((
                                        payload.width as u32,
                                        payload.height as u32,
                                    ));
                                }

                                if ext == extension_reference_image {
                                    // If this is the reference image, we store it in the main image field
                                    if let Some(mut_final_sample) = &mut final_sample {
                                        mut_final_sample.image = image;
                                    } else {
                                        // Init the sample
                                        final_sample = Some(Sample {
                                            id: String::from(
                                                sample_id.to_str().unwrap_or("unknown"),
                                            ),
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
                                    // Otherwise, we store it in the additional images
                                    if let Some(mut_final_sample) = &mut final_sample {
                                        mut_final_sample
                                            .additional_images
                                            .insert(item.filename.clone(), image);
                                    } else {
                                        // Init the sample
                                        final_sample = Some(Sample {
                                            id: String::from(
                                                sample_id.to_str().unwrap_or("unknown"),
                                            ),
                                            source: sample.name.clone(),
                                            image: to_python_image_payload(ImagePayload {
                                                data: std::sync::Arc::from(Vec::<u8>::new()),
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
                            Err(e) => {
                                debug!("wds_worker: error loading image: {e}");
                                continue;
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
                if let Some(mut final_sample) = final_sample {
                    final_sample.attributes = attributes;
                    return Ok(final_sample);
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
    cancel: CancellationToken,
) -> Result<(), String> {
    let max_tasks = min(num_cpus::get() * 4, max_tasks_from_env(num_cpus::get()));
    let image_transform = Arc::new(image_transform);
    let count = run_worker(
        samples_metadata_rx,
        samples_tx,
        max_tasks,
        limit,
        cancel,
        TarballSample::is_empty,
        move |item| {
            process_sample(
                item,
                image_transform.clone(),
                encoding,
                extension_reference_image.clone(),
            )
        },
    )
    .await?;
    info!("wds_worker: samples delivered: {count}");
    Ok(())
}

pub fn deserialize_samples(
    samples_metadata_rx: kanal::Receiver<TarballSample>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
    limit: usize,
    extension_reference_image: String,
    cancel: CancellationToken,
) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get()) // Tasks in flight are limited by DATAGO_MAX_TASKS env
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        match async_deserialize_samples(
            samples_metadata_rx,
            samples_tx,
            image_transform,
            encoding,
            limit,
            extension_reference_image,
            cancel,
        )
        .await
        {
            Ok(_) => debug!("wds_worker: all samples processed successfully"),
            Err(e) => error!("wds_worker: error processing samples : {e}"),
        }
    });
    runtime.shutdown_timeout(std::time::Duration::from_millis(100));
}
