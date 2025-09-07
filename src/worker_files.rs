use crate::image_processing;
use crate::structs::{ImagePayload, Sample};
use io_uring::{opcode, types, IoUring};
use log::{debug, error};
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::Arc;

// Maintain the objects which are tied to an ongoing io_uring request
// makes it possible to make sure that their lifetime is correctly handled,
// and to keep all the relevant information in sync
#[allow(dead_code)]
struct IoUringRequest {
    file: File, // This will keep the file descriptor open for the lifetime of the request
    buffer: Vec<u8>,
    path: String,
}

// We'll label all the requests issued over time
type RequestId = u64;

struct IoTracker {
    in_flight: HashMap<RequestId, IoUringRequest>,
    next_id: RequestId, // Auto-incrementing ID generator
}

impl IoTracker {
    fn new() -> Self {
        Self {
            in_flight: HashMap::with_capacity(128),
            next_id: 1,
        }
    }

    fn insert(&mut self, request: IoUringRequest) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        self.in_flight.insert(id, request);
        id
    }

    fn remove(&mut self, id: RequestId) -> Option<IoUringRequest> {
        self.in_flight.remove(&id)
    }

    fn len(&self) -> usize {
        self.in_flight.len()
    }
}

struct IoUringResult {
    buffer: Vec<u8>,
    path: String,
}

async fn io_uring_submit_read(
    path: &str,
    ring: &mut IoUring,
    uid: u64,
) -> Result<IoUringRequest, std::io::Error> {
    // Open the file
    let file = File::open(path)?;
    let file_fd = file.as_raw_fd();
    let file_size = file.metadata()?.len() as u32;

    // Prepare the read operation
    let mut buffer = vec![0u8; file_size as usize];
    let sqe = opcode::Read::new(types::Fd(file_fd), buffer.as_mut_ptr(), file_size)
        .build()
        .user_data(uid);

    // Submit the read operation
    unsafe {
        ring.submission()
            .push(&sqe)
            .expect("Failed to push to submission queue");
    }

    Ok(IoUringRequest {
        file, // We need to keep the file descriptor open for lifetime reasons
        buffer,
        path: path.to_string(),
    })
}

async fn io_uring_drain_into_tasks(
    ring: &mut IoUring,
    io_tracker: &mut IoTracker,
    samples_tx: &kanal::Sender<Option<Sample>>,
    task_queue: &mut tokio::task::JoinSet<Result<(), std::io::Error>>,
    image_transform: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
) {
    // We get a handle to the ring + the files which were kept open

    // Not required while polling is used, could be an env variable
    ring.completion().sync();

    while let Some(cqe) = ring.completion().next() {
        // Grab the raw results from the completion queue
        if cqe.result() < 0 {
            error!("io_uring: Read failed: {}", cqe.result());
            let error_closure = async move || Err(std::io::Error::last_os_error());
            task_queue.spawn(error_closure());
        }

        let request = &io_tracker.in_flight[&cqe.user_data()];

        let io_uring_result = IoUringResult {
            buffer: request.buffer[..cqe.result() as usize].to_vec(),
            path: request.path.clone(), // FIXME: ideally we would `move` here instead
        };

        // Remove this request from the tracker
        io_tracker.remove(cqe.user_data());

        // Spawn a task to create a full sample from the results
        task_queue.spawn(sample_from_io_uring_read(
            io_uring_result,
            samples_tx.clone(), // FIXME: these clones are a bit ugly, probably possible to do better
            image_transform.clone(),
            encode_images,
            img_to_rgb8,
        ));
    }
}

#[allow(dead_code)] // we use this for unit tests
async fn io_uring_read_file(path: &str) -> Result<Vec<u8>, std::io::Error> {
    // Submit / read cycle with a single file, really not a good idea for perf but good for unit testing
    let mut ring = IoUring::new(1).unwrap();
    let mut io_tracker = IoTracker::new();

    // Submit a single read request
    match io_uring_submit_read(path, &mut ring, io_tracker.next_id).await {
        Ok(request) => {
            io_tracker.insert(request);
        }
        Err(e) => {
            return Err(e);
        }
    }

    // Get the result from the queue
    match io_uring_retire_available_reads(&mut ring, &mut io_tracker).await {
        Ok(mut results) if results.len() == 1 => Ok(results.remove(0).buffer), // Takes ownership
        Ok(_) => Err(std::io::Error::other("Failed to read file")),
        Err(e) => Err(e),
    }
}

async fn sample_from_io_uring_read(
    result: IoUringResult,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
) -> Result<(), std::io::Error> {
    match image::load_from_memory(&result.buffer) {
        Ok(raw_image) => {
            let image = image_processing::image_to_payload(
                raw_image,
                &image_transform,
                &"".to_string(),
                encode_images,
                img_to_rgb8,
            )
            .await
            .unwrap();

            let sample = Sample {
                id: result.path,
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

            // Channel is closed, wrapping up
            if samples_tx.send(Some(sample)).is_err() {
                return Err(std::io::Error::other(
                    "file_worker: failed to send sample to channel",
                ));
            }

            Ok(())
        }
        Err(e) => {
            error!("file_worker: failed opening file {e}");
            Err(std::io::Error::other("Failed to read file"))
        }
    }
}

async fn io_uring_retire_available_reads(
    ring: &mut IoUring,
    io_tracker: &mut IoTracker,
) -> Result<Vec<IoUringResult>, std::io::Error> {
    // We get a handle to the ring + the files which were kept open. Handle a batch of completions, if possible
    // NOTE: This is not very efficient, because we retire a bunch of payloads prior to
    // processing them. This should be correct and a good baseline though, used for some tests.

    let mut results = Vec::<IoUringResult>::with_capacity(io_tracker.len());

    ring.completion().sync(); // Ensure kernel -> userspace sync

    while let Some(cqe) = ring.completion().next() {
        if cqe.result() < 0 {
            error!("io_uring: Read failed: {}", cqe.result());
            return Err(std::io::Error::last_os_error());
        }

        let request = &io_tracker.in_flight[&cqe.user_data()];

        results.push(IoUringResult {
            buffer: request.buffer[..cqe.result() as usize].to_vec(),
            path: request.path.clone(), // FIXME: ideally we would `move` here instead
        });

        // Remove this request from the tracker
        io_tracker.remove(cqe.user_data());
    }
    Ok(results)
}

async fn retire_done_tasks_from_set(
    tasks: &mut tokio::task::JoinSet<Result<(), std::io::Error>>,
) -> usize {
    let mut count = 0;
    while let Some(task_result) = tasks.try_join_next() {
        match task_result {
            Ok(_) => {
                count += 1;
            }
            Err(e) => {
                error!("file_worker: failed to process sample: {e}");
            }
        }
    }
    count
}

async fn io_uring_pipeline(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Arc<Option<image_processing::ARAwareTransform>>,
    encode_images: bool,
    img_to_rgb8: bool,
    limit: usize,
) {
    let mut count: usize = 0;

    // Limit concurrent IO, but make it possible to batch IO, and to overlap it with compute
    // Each pipeline will host a io_uring queue.
    let io_batch_size: usize = 4; // submit request in batch
    let io_depth: usize = 64; // how many requests to keep in flight in the ring

    let mut ring = IoUring::builder()
        // .setup_sqpoll(100) // Enable kernel-polling mode // needs to be used with register_files(), opengl vertex buffer style
        // .setup_iopoll() // Optional: Combine with IOPOLL for buffered I/O
        .build(io_depth as u32)
        .expect("Failed to create io_uring"); // Queue depth

    let mut io_tracker = IoTracker::new();

    // Limit concurrent processing tasks, but make it possible to overlap IO and compute
    // This is only about the tasks in flight in this io_uring submission queue
    let mut main_tasks_queue = tokio::task::JoinSet::new();
    let max_number_tasks = num_cpus::get();

    while let Ok(received) = samples_metadata_rx.recv() {
        if received == serde_json::Value::Null {
            debug!("file_worker: end of stream received, stopping there");
            let _ = samples_metadata_rx.close();
            break;
        }

        // Submit a read request, log it being in flight.
        // Note that we need to index the request, as it may be processed in a different order
        let path = received.as_str().unwrap();
        match io_uring_submit_read(path, &mut ring, io_tracker.next_id).await {
            Ok(request) => {
                io_tracker.insert(request);
                debug!("file_worker: submitted read request for {path}");
            }
            Err(e) => {
                error!("file_worker: failed to submit read request for {path}: {e}");
                continue;
            }
        }

        // If enough submissions in the queue, submit a batch
        if ring.submission().len() >= io_batch_size {
            ring.submit().expect("Failed to submit batch");
        }

        // Retire the done tasks from the main queue
        count += retire_done_tasks_from_set(&mut main_tasks_queue).await;

        // If enough submissions in the queue, opportunistically retire what is already available
        while io_tracker.len() >= io_batch_size {
            io_uring_drain_into_tasks(
                &mut ring,
                &mut io_tracker,
                &samples_tx,
                &mut main_tasks_queue,
                &image_transform,
                encode_images,
                img_to_rgb8,
            )
            .await;
        }

        // If too many in flight tasks, busy loop until some are done
        while main_tasks_queue.len() > max_number_tasks {
            count += retire_done_tasks_from_set(&mut main_tasks_queue).await;
            tokio::task::yield_now().await; // Give some time back to the scheduler
        }

        if count > limit {
            break;
        }
    }

    // Consume all the remaining IO, close the files properly
    while io_tracker.len() > 0 {
        match io_uring_retire_available_reads(&mut ring, &mut io_tracker).await {
            Ok(_) => {}
            Err(_) => {
                break;
            }
        }
    }
    debug!("file_worker: total samples sent: {count}\n");

    // Signal the end of the stream
    if samples_tx.send(None).is_ok() {};
}

async fn async_pull_samples(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
    limit: usize,
) {
    // Each pipeline will host a io_uring queue.
    // Max number of concurrent pipelines would depend on the underlying hardware
    // TODO: @lefaudeux - env variable, VAST or SSD may benefit from different settings
    let mut pipelines = tokio::task::JoinSet::new();
    let max_pipelines = num_cpus::get() / 2; // CPUs outside of typical hyperthreading
    debug!("Using {max_pipelines} io_uring pipelines in parallel");

    let shareable_img_tfm = Arc::new(image_transform);

    while pipelines.len() < max_pipelines {
        let rx_channel = samples_metadata_rx.clone();
        let tx_channel = samples_tx.clone();
        let img_tfm = shareable_img_tfm.clone();
        pipelines.spawn(io_uring_pipeline(
            rx_channel,
            tx_channel,
            img_tfm,
            encode_images,
            img_to_rgb8,
            limit,
        ));
    }

    // If we have enough tasks, we'll wait for the older one to finish
    let _ = pipelines.join_all().await;

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
    tokio::runtime::Builder::new_multi_thread() // one thread per core by default
        .thread_name("datago")
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

#[allow(dead_code)]
async fn image_from_path(path: &str) -> Result<image::DynamicImage, image::ImageError> {
    let bytes = io_uring_read_file(path).await?;
    let img = image::load_from_memory(&bytes)?;
    Ok(img)
}

#[allow(dead_code)]
async fn image_payload_from_path(
    path: &str,
    image_transform: &Option<image_processing::ARAwareTransform>,
    encode_images: bool,
    img_to_rgb8: bool,
) -> Result<ImagePayload, image::ImageError> {
    let img = image_from_path(path).await?;
    let payload = image_processing::image_to_payload(
        img,
        image_transform,
        &"".to_string(),
        encode_images,
        img_to_rgb8,
    )
    .await?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_processing::ImageTransformConfig;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_image(path: &std::path::Path) {
        // Create a simple 24x32 PNG image
        let img = image::DynamicImage::new_rgb8(24, 32);
        img.save(path).unwrap();
    }

    #[tokio::test]
    async fn test_image_from_path_success() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result = image_from_path(image_path.to_str().unwrap()).await;
        assert!(result.is_ok());

        let img = result.unwrap();
        assert_eq!(img.width(), 24);
        assert_eq!(img.height(), 32);
    }

    #[tokio::test]
    async fn test_io_uring_file_load() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result = io_uring_read_file(image_path.to_str().unwrap()).await;
        assert!(result.is_ok());

        let bytes = result.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!(img.width(), 24);
        assert_eq!(img.height(), 32);
    }

    #[tokio::test]
    async fn test_image_from_path_invalid_file() {
        let result = image_from_path("/nonexistent/path.png").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_image_from_path_invalid_image_data() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("not_an_image.txt");
        fs::write(&file_path, "This is not image data").unwrap();

        let result = image_from_path(file_path.to_str().unwrap()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_image_payload_from_path_basic() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result =
            image_payload_from_path(image_path.to_str().unwrap(), &None, false, false).await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.width, 24);
        assert_eq!(payload.height, 32);
        assert_eq!(payload.original_width, 24);
        assert_eq!(payload.original_height, 32);
        assert!(!payload.data.is_empty());
    }

    #[tokio::test]
    async fn test_image_payload_from_path_with_transform() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");

        // Create a larger test image
        let img = image::DynamicImage::new_rgb8(100, 100);
        img.save(&image_path).unwrap();

        let transform_config = ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 64,
            downsampling_ratio: 16,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: false,
            image_to_rgb8: false,
        };

        let transform = Some(transform_config.get_ar_aware_transform());

        let result =
            image_payload_from_path(image_path.to_str().unwrap(), &transform, false, false).await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.original_width, 100);
        assert_eq!(payload.original_height, 100);
        // Transformed size should be different
        assert_ne!(payload.width, 100);
        assert_ne!(payload.height, 100);
    }

    #[tokio::test]
    async fn test_image_payload_from_path_with_encoding() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &None,
            true, // encode_images = true
            false,
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.channels, -1); // Encoded images have channels = -1
        assert!(!payload.data.is_empty());

        // Should be able to decode the image
        let decoded = image::load_from_memory(&payload.data);
        assert!(decoded.is_ok());
    }

    #[tokio::test]
    async fn test_image_payload_from_path_rgb8_conversion() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");

        // Create an RGBA image
        let img = image::DynamicImage::new_rgba8(1, 1);
        img.save(&image_path).unwrap();

        let result = image_payload_from_path(
            image_path.to_str().unwrap(),
            &None,
            false,
            true, // image_to_rgb8 = true
        )
        .await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.channels, 3); // Should be converted to RGB (3 channels)
        assert_eq!(payload.bit_depth, 8);
    }

    // #[tokio::test]
    // async fn test_pull_sample_success() {
    //     let temp_dir = TempDir::new().unwrap();
    //     let image_path = temp_dir.path().join("test.png");
    //     create_test_image(&image_path);

    //     let (tx, rx) = kanal::bounded(10);
    //     let sample_json = serde_json::Value::String(image_path.to_str().unwrap().to_string());

    //     let result = pull_sample(sample_json, Arc::new(None), false, false, tx).await;

    //     assert!(result.is_ok());

    //     // Check that a sample was sent
    //     let received = rx.recv().unwrap();
    //     assert!(received.is_some());

    //     let sample = received.unwrap();
    //     assert_eq!(sample.source, "filesystem");
    //     assert!(!sample.id.is_empty());
    //     assert_eq!(sample.image.width, 1);
    //     assert_eq!(sample.image.height, 1);
    //     assert!(sample.attributes.is_empty());
    //     assert!(sample.coca_embedding.is_empty());
    //     assert!(sample.tags.is_empty());
    //     assert!(sample.masks.is_empty());
    //     assert!(sample.latents.is_empty());
    //     assert!(sample.additional_images.is_empty());
    //     assert_eq!(sample.duplicate_state, 0);
    // }

    // #[tokio::test]
    // async fn test_pull_sample_invalid_path() {
    //     let (tx, rx) = kanal::bounded(10);
    //     let sample_json = serde_json::Value::String("/nonexistent/path.png".to_string());

    //     let result = pull_sample(sample_json, Arc::new(None), false, false, tx).await;

    //     assert!(result.is_err());

    //     // No sample should be sent on error
    //     assert!(rx.try_recv().is_err());
    // }

    // #[tokio::test]
    // async fn test_async_pull_samples_basic() {
    //     let temp_dir = TempDir::new().unwrap();

    //     // Create multiple test images
    //     let mut image_paths = Vec::new();
    //     for i in 0..3 {
    //         let image_path = temp_dir.path().join(format!("test_{i}.png"));
    //         create_test_image(&image_path);
    //         image_paths.push(image_path.to_str().unwrap().to_string());
    //     }

    //     let (metadata_tx, metadata_rx) = kanal::bounded(10);
    //     let (samples_tx, samples_rx) = kanal::bounded(10);

    //     // Send image paths
    //     for path in &image_paths {
    //         metadata_tx
    //             .send(serde_json::Value::String(path.clone()))
    //             .unwrap();
    //     }
    //     metadata_tx.send(serde_json::Value::Null).unwrap(); // End marker

    //     async_pull_samples(metadata_rx, samples_tx, None, false, false, 10).await;

    //     // Check received samples
    //     let mut received_samples = Vec::new();
    //     while let Ok(sample_opt) = samples_rx.recv() {
    //         if let Some(sample) = sample_opt {
    //             received_samples.push(sample);
    //         } else {
    //             break; // End marker
    //         }
    //     }

    //     assert_eq!(received_samples.len(), image_paths.len());
    //     for sample in &received_samples {
    //         assert_eq!(sample.source, "filesystem");
    //         assert_eq!(sample.image.width, 1);
    //         assert_eq!(sample.image.height, 1);
    //     }
    // }

    #[tokio::test]
    async fn test_async_pull_samples_with_limit() {
        let temp_dir = TempDir::new().unwrap();

        // Create more images than the limit
        for i in 0..10 {
            let image_path = temp_dir.path().join(format!("test_{i}.png"));
            create_test_image(&image_path);
        }

        let (metadata_tx, metadata_rx) = kanal::bounded(20);
        let (samples_tx, samples_rx) = kanal::bounded(20);

        // Send more paths than the limit
        for i in 0..10 {
            let path = temp_dir.path().join(format!("test_{i}.png"));
            metadata_tx
                .send(serde_json::Value::String(
                    path.to_str().unwrap().to_string(),
                ))
                .unwrap();
        }
        metadata_tx.send(serde_json::Value::Null).unwrap();

        let limit = 3;
        async_pull_samples(metadata_rx, samples_tx, None, false, false, limit).await;

        // Count received samples
        let mut count = 0;
        while let Ok(sample_opt) = samples_rx.recv() {
            if sample_opt.is_some() {
                count += 1;
            } else {
                break;
            }
        }

        // Should respect the limit (might be slightly more due to async processing)
        assert!(count <= limit + 2); // Allow some buffer for async processing
    }

    fn create_test_webp_image(path: &std::path::Path) {
        // Create a simple 2x2 WebP image
        let img = image::DynamicImage::new_rgb8(2, 2);
        img.save_with_format(path, image::ImageFormat::WebP)
            .unwrap();
    }

    #[tokio::test]
    async fn test_image_from_path_webp() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.webp");
        create_test_webp_image(&image_path);

        let result = image_from_path(image_path.to_str().unwrap()).await;
        assert!(result.is_ok());

        let img = result.unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
    }

    #[tokio::test]
    async fn test_image_payload_from_path_webp() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.webp");
        create_test_webp_image(&image_path);

        let result =
            image_payload_from_path(image_path.to_str().unwrap(), &None, false, false).await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.width, 2);
        assert_eq!(payload.height, 2);
        assert_eq!(payload.original_width, 2);
        assert_eq!(payload.original_height, 2);
        assert!(!payload.data.is_empty());
    }

    // #[tokio::test]
    // async fn test_pull_sample_webp() {
    //     let temp_dir = TempDir::new().unwrap();
    //     let image_path = temp_dir.path().join("test.webp");
    //     create_test_webp_image(&image_path);

    //     let (tx, rx) = kanal::bounded(10);
    //     let sample_json = serde_json::Value::String(image_path.to_str().unwrap().to_string());

    //     let result = pull_sample(sample_json, Arc::new(None), false, false, tx).await;

    //     assert!(result.is_ok());

    //     // Check that a sample was sent
    //     let received = rx.recv().unwrap();
    //     assert!(received.is_some());

    //     let sample = received.unwrap();
    //     assert_eq!(sample.source, "filesystem");
    //     assert!(!sample.id.is_empty());
    //     assert_eq!(sample.image.width, 2);
    //     assert_eq!(sample.image.height, 2);
    // }

    // #[test]
    // fn test_pull_samples_sync_wrapper() {
    //     let temp_dir = TempDir::new().unwrap();
    //     let image_path = temp_dir.path().join("test.png");
    //     create_test_image(&image_path);

    //     let (metadata_tx, metadata_rx) = kanal::bounded(10);
    //     let (samples_tx, samples_rx) = kanal::bounded(10);

    //     // Send one image path
    //     metadata_tx
    //         .send(serde_json::Value::String(
    //             image_path.to_str().unwrap().to_string(),
    //         ))
    //         .unwrap();
    //     metadata_tx.send(serde_json::Value::Null).unwrap();

    //     // Test the sync wrapper
    //     pull_samples(metadata_rx, samples_tx, None, false, false, 1);

    //     // Check that a sample was received
    //     let sample_opt = samples_rx.recv().unwrap();
    //     assert!(sample_opt.is_some());

    //     let sample = sample_opt.unwrap();
    //     assert_eq!(sample.source, "filesystem");

    //     // Should receive end marker
    //     let end_marker = samples_rx.recv().unwrap();
    //     assert!(end_marker.is_none());
    // }
}
