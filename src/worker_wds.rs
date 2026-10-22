use crate::image_processing;
use crate::structs::{to_python_image_payload, BinaryFile, ImagePayload, Sample, TarballSample};
use crate::worker_utils::{max_tasks_from_env, run_worker};
use log::{debug, error, info, warn};
use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// List all the types we'll support
pub const TEXT_TYPES: [&str; 3] = ["cls", "json", "txt"];
pub const IMG_TYPES: [&str; 3] = ["jpg", "jpeg", "png"];

fn extension(filename: &str) -> &str {
    Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
}

async fn image_payload_from_file(
    file: &BinaryFile,
    image_transform: &Option<image_processing::ARAwareTransform>,
    aspect_ratio: &String,
    encoding: image_processing::ImageEncoding,
) -> Result<ImagePayload, ()> {
    let result = match image::load_from_memory(&file.buffer) {
        Ok(image) => {
            image_processing::image_to_payload(image, image_transform, aspect_ratio, encoding).await
        }
        Err(error) => Err(error),
    };
    result.map_err(|error| warn!("wds_worker: cannot prepare {}: {error}", file.filename))
}

async fn process_sample(
    sample: TarballSample,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encoding: image_processing::ImageEncoding,
    extension_reference_image: String,
) -> Result<Sample, ()> {
    // Prepare the reference first so archive order cannot change the crop used
    // for additional images, or leave the main image as an invalid placeholder.
    let reference = sample
        .iter()
        .find(|file| extension(&file.filename).eq_ignore_ascii_case(&extension_reference_image))
        .ok_or_else(|| warn!("wds_worker: no reference image in {}", sample.name))?;
    let id = Path::new(&reference.filename)
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or(())?
        .to_owned();
    let image = image_payload_from_file(reference, &img_tfm, &String::new(), encoding).await?;
    let aspect_ratio =
        image_processing::aspect_ratio_to_str((image.width as u32, image.height as u32));
    let mut attributes = HashMap::new();
    let mut additional_images = HashMap::new();
    for file in sample.iter() {
        let ext = extension(&file.filename).to_ascii_lowercase();
        if ext.eq_ignore_ascii_case(&extension_reference_image) {
            continue;
        }
        if IMG_TYPES.contains(&ext.as_str()) {
            // Preserve tolerance of unreadable optional images, but never insert
            // empty placeholders which fail later during Python conversion.
            if let Ok(payload) =
                image_payload_from_file(file, &img_tfm, &aspect_ratio, encoding).await
            {
                additional_images.insert(file.filename.clone(), to_python_image_payload(payload));
            }
        } else if TEXT_TYPES.contains(&ext.as_str()) {
            let text = String::from_utf8_lossy(&file.buffer).into_owned();
            attributes.insert(ext, serde_json::Value::String(text));
        } else {
            debug!("wds_worker: unsupported file type: {}", file.filename);
        }
    }
    Ok(Sample {
        id,
        source: sample.name,
        image: to_python_image_payload(image),
        attributes,
        additional_images,
        coca_embedding: vec![],
        tags: vec![],
        masks: HashMap::new(),
        latents: HashMap::new(),
        duplicate_state: 0,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::BinaryFile;
    use std::io::Cursor;

    fn image_file(filename: &str, color: [u8; 3]) -> BinaryFile {
        let image = image::RgbImage::from_pixel(3, 2, image::Rgb(color));
        let mut buffer = Cursor::new(Vec::new());
        image
            .write_to(&mut buffer, image::ImageFormat::Png)
            .unwrap();
        BinaryFile {
            filename: filename.into(),
            buffer: buffer.into_inner(),
        }
    }

    #[tokio::test]
    async fn archive_order_does_not_drop_additional_images() {
        for additional_first in [true, false] {
            let mut files = vec![
                image_file("sample.jpg", [1, 2, 3]),
                image_file("sample.png", [4, 5, 6]),
            ];
            if !additional_first {
                files.reverse();
            }
            let sample = process_sample(
                TarballSample {
                    name: "shard".into(),
                    content: files,
                },
                Arc::new(None),
                image_processing::ImageEncoding::default(),
                "png".into(),
            )
            .await
            .unwrap();
            assert_eq!(sample.id, "sample");
            assert_eq!(sample.image.get_payload().data[0..3], [4, 5, 6]);
            assert_eq!(sample.additional_images.len(), 1);
            assert_eq!(
                sample.additional_images["sample.jpg"].get_payload().data[0..3],
                [1, 2, 3]
            );
        }
    }

    #[tokio::test]
    async fn file_extensions_are_case_insensitive() {
        let sample = process_sample(
            TarballSample {
                name: "shard".into(),
                content: vec![
                    image_file("sample.PNG", [4, 5, 6]),
                    image_file("sample.JPG", [1, 2, 3]),
                    BinaryFile {
                        filename: "sample.TXT".into(),
                        buffer: b"caption".to_vec(),
                    },
                ],
            },
            Arc::new(None),
            image_processing::ImageEncoding::default(),
            "PNG".into(),
        )
        .await
        .unwrap();
        assert_eq!(sample.additional_images.len(), 1);
        assert_eq!(sample.attributes["txt"], "caption");
    }

    #[tokio::test]
    async fn reference_image_controls_the_crop_for_additional_images() {
        let mut buffer = Cursor::new(Vec::new());
        image::RgbImage::new(2, 3)
            .write_to(&mut buffer, image::ImageFormat::Png)
            .unwrap();
        let config: image_processing::ImageTransformConfig =
            serde_json::from_value(serde_json::json!({
                "crop_and_resize": true, "default_image_size": 8,
                "downsampling_ratio": 1, "min_aspect_ratio": 0.5, "max_aspect_ratio": 2.0,
            }))
            .unwrap();
        let sample = process_sample(
            TarballSample {
                name: "shard".into(),
                content: vec![
                    image_file("sample.jpg", [1, 2, 3]), // Landscape, stored first.
                    BinaryFile {
                        filename: "sample.png".into(),
                        buffer: buffer.into_inner(),
                    },
                ],
            },
            Arc::new(Some(config.get_ar_aware_transform())),
            image_processing::ImageEncoding::default(),
            "png".into(),
        )
        .await
        .unwrap();
        let reference = sample.image.get_payload();
        let additional = sample.additional_images["sample.jpg"].get_payload();
        assert!(reference.height > reference.width);
        assert_eq!(
            (additional.width, additional.height),
            (reference.width, reference.height)
        );
    }

    #[tokio::test]
    async fn reference_encoding_failure_is_rejected() {
        let mut buffer = Cursor::new(Vec::new());
        image::RgbaImage::new(3, 2)
            .write_to(&mut buffer, image::ImageFormat::Png)
            .unwrap();
        let result = process_sample(
            TarballSample {
                name: "shard".into(),
                content: vec![BinaryFile {
                    filename: "sample.png".into(),
                    buffer: buffer.into_inner(),
                }],
            },
            Arc::new(None),
            image_processing::ImageEncoding {
                encode_images: true,
                encode_format: image_processing::EncodeFormat::Jpeg,
                ..Default::default()
            },
            "png".into(),
        )
        .await;
        assert!(result.is_err()); // JPEG cannot encode RGBA without conversion.
    }

    #[tokio::test]
    async fn missing_or_unreadable_reference_is_rejected() {
        for file in [
            image_file("sample.jpg", [1, 2, 3]),
            BinaryFile {
                filename: "sample.png".into(),
                buffer: b"corrupt".to_vec(),
            },
        ] {
            let result = process_sample(
                TarballSample {
                    name: "shard".into(),
                    content: vec![file],
                },
                Arc::new(None),
                image_processing::ImageEncoding::default(),
                "png".into(),
            )
            .await;
            assert!(result.is_err());
        }
    }
}
