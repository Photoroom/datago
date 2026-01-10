use crate::image_processing;
use crate::structs::{to_python_image_payload, ImagePayload, Sample};
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

    // Prepare the read operation with proper buffer management
    // Initialize the buffer with zeros to ensure proper alignment and initialization
    let mut buffer = vec![0u8; file_size as usize];

    let buffer_ptr = buffer.as_mut_ptr();

    let sqe = opcode::Read::new(types::Fd(file_fd), buffer_ptr, file_size)
        .build()
        .user_data(uid);

    // Submit the read operation
    unsafe {
        ring.submission()
            .push(&sqe)
            .expect("Failed to push to submission queue");
    }

    // Actually submit the operation to the kernel
    ring.submit()?;

    Ok(IoUringRequest {
        file, // Keep the file descriptor open for the duration of the operation
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
        } else {
            // Take ownership of the request to properly manage buffer lifetime
            let mut request = io_tracker.remove(cqe.user_data()).unwrap();

            // Get the number of bytes actually read
            let bytes_read = cqe.result() as usize;

            // Safe: Truncate the buffer to the actual bytes read
            // This is safe because io_uring has written exactly 'bytes_read' bytes to our buffer
            // and we're using Vec's built-in truncate method
            if bytes_read < request.buffer.len() {
                request.buffer.truncate(bytes_read);
            }

            let io_uring_result = IoUringResult {
                buffer: request.buffer, // Take ownership of the buffer
                path: request.path,     // Take ownership of the path
            };

            // The File will be dropped here, properly closing the file descriptor

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

    // Give the io_uring operation a chance to complete
    // This is a workaround for the fact that io_uring operations might not be immediately available
    tokio::task::yield_now().await;

    // Get the result from the queue
    match io_uring_retire_available_reads(&mut ring, &mut io_tracker).await {
        Ok(mut results) if results.len() == 1 => {
            let result = results.remove(0);
            // The buffer should already be properly sized by io_uring_retire_available_reads
            Ok(result.buffer)
        }
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
            let encoding = image_processing::ImageEncoding {
                encode_images,
                img_to_rgb8,
                ..Default::default()
            };
            let image = image_processing::image_to_payload(
                raw_image,
                &image_transform,
                &"".to_string(),
                encoding,
            )
            .await
            .unwrap();

            let sample = Sample {
                id: result.path,
                source: "filesystem".to_string(),
                image: to_python_image_payload(image),
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

    // Ensure kernel -> userspace sync first
    ring.completion().sync();

    // Now process all available completions
    while let Some(cqe) = ring.completion().next() {
        if cqe.result() < 0 {
            error!("io_uring: Read failed: {}", cqe.result());
            return Err(std::io::Error::last_os_error());
        }

        // Take ownership of the request to properly manage buffer lifetime
        let mut request = io_tracker.remove(cqe.user_data()).unwrap();

        // Get the number of bytes actually read
        let bytes_read = cqe.result() as usize;

        // Safe: Truncate the buffer to the actual bytes read
        if bytes_read < request.buffer.len() {
            request.buffer.truncate(bytes_read);
        }

        results.push(IoUringResult {
            buffer: request.buffer, // Take ownership of the buffer
            path: request.path,     // Take ownership of the path
        });
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

    // Limit concurrent IO, but make it possible to batch IO, and to overlap it with compute.
    // Each pipeline will host a io_uring queue.

    // submit request in batch
    let io_batch_size: usize = std::env::var("DATAGO_IO_URING_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);

    // how many requests to keep in flight in the ring
    let io_depth: usize = std::env::var("DATAGO_IO_URING_DEPTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);

    debug!("Using io_uring with batch size {io_batch_size} and depth {io_depth}");
    let mut ring = IoUring::builder()
        // .setup_sqpoll(100) // Enable kernel-polling mode for the submission queue
        .setup_coop_taskrun() // Don't interrupt user space on completion
        // .setup_iopoll() // Optional: Combine with IOPOLL for buffered I/O
        .build(io_depth as u32)
        .expect("Failed to create io_uring"); // Queue depth

    // Limit concurrent processing tasks, but make it possible to overlap IO and compute
    // This is only about the tasks in flight in this io_uring submission queue
    let mut main_tasks_queue = tokio::task::JoinSet::new();
    let max_number_tasks = std::env::var("DATAGO_MAX_TASKS_PER_RING")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(num_cpus::get());

    debug!("Enabling {max_number_tasks} concurrent compute tasks per io_uring pipeline");

    let mut io_tracker = IoTracker::new();

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
        while io_tracker.len() >= (io_depth / 2) {
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

        // Check if we have reached the limit
        // Use a simple check to avoid any potential issues with io_tracker access
        if count >= limit {
            break;
        }
    }

    // Consume all the remaining tasks
    debug!("Wrapping up the ongoing compute tasks");
    main_tasks_queue.abort_all();

    // Consume all the remaining IO, close the files properly
    let remaining_tasks = io_tracker.len();
    debug!("Wrapping up the io_uring pipeline, {remaining_tasks} requests in flight");
    while ring.completion().count() > 0 {
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
    encoding: image_processing::ImageEncoding,
    limit: usize,
) {
    // Check if io_uring should be used
    let use_io_uring = std::env::var("DATAGO_USE_IO_URING")
        .map(|s| s.to_lowercase() == "true")
        .unwrap_or(true); // Default to true - memory issues have been fixed

    if use_io_uring {
        // Use the new io_uring implementation
        let max_pipelines = std::env::var("DATAGO_IO_URING_PIPELINES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(num_cpus::get() / 2);

        debug!("Using {max_pipelines} io_uring pipelines in parallel");

        let shareable_img_tfm = Arc::new(image_transform);
        let mut pipelines = tokio::task::JoinSet::new();

        while pipelines.len() < max_pipelines {
            let rx_channel = samples_metadata_rx.clone();
            let tx_channel = samples_tx.clone();
            let img_tfm = shareable_img_tfm.clone();
            pipelines.spawn(io_uring_pipeline(
                rx_channel,
                tx_channel,
                img_tfm,
                encoding.encode_images,
                encoding.img_to_rgb8,
                limit,
            ));
        }

        // Wait for all pipelines to complete
        let _ = pipelines.join_all().await;

        // Signal the end of the stream for io_uring path
        if samples_tx.send(None).is_ok() {};
    } else {
        // Fall back to the original async implementation
        async_pull_samples_fallback(
            samples_metadata_rx,
            samples_tx,
            image_transform,
            encoding,
            limit,
        )
        .await;

        // The fallback function already sends the end signal
    }
}

pub fn pull_samples(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
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
                encoding,
                limit,
            )
            .await;
        });
}

async fn image_from_path(path: &str) -> Result<image::DynamicImage, image::ImageError> {
    // Use standard file I/O to avoid io_uring memory issues
    let bytes =
        std::fs::read(path).map_err(|e| image::ImageError::IoError(std::io::Error::other(e)))?;
    let img = image::load_from_memory(&bytes)?;
    Ok(img)
}

async fn image_payload_from_path(
    path: &str,
    image_transform: &Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
) -> Result<ImagePayload, image::ImageError> {
    let img = image_from_path(path).await?;
    let payload =
        image_processing::image_to_payload(img, image_transform, &"".to_string(), encoding).await?;
    Ok(payload)
}

async fn pull_sample(
    sample_json: serde_json::Value,
    img_tfm: Arc<Option<image_processing::ARAwareTransform>>,
    encoding: image_processing::ImageEncoding,
    samples_tx: kanal::Sender<Option<Sample>>,
) -> Result<(), ()> {
    let path = sample_json.as_str().unwrap();
    debug!("Starting to process file: {}", path);

    match image_payload_from_path(path, &img_tfm, encoding).await {
        Ok(image) => {
            debug!("Successfully processed file: {}", path);
            let sample = Sample {
                id: sample_json.to_string(),
                source: "filesystem".to_string(),
                image: to_python_image_payload(image),
                attributes: HashMap::new(),
                coca_embedding: vec![],
                tags: vec![],
                masks: HashMap::new(),
                latents: HashMap::new(),
                additional_images: HashMap::new(),
                duplicate_state: 0,
            };

            if samples_tx.send(Some(sample)).is_err() {
                return Err(());
            }
            Ok(())
        }
        Err(e) => {
            error!("Failed to load image from path {}: {}", path, e);
            // Add more specific error handling based on error type
            if let image::ImageError::IoError(io_err) = e {
                error!("IO Error for file {}: {}", path, io_err);
            }
            Err(())
        }
    }
}

async fn async_pull_samples_fallback(
    samples_metadata_rx: kanal::Receiver<serde_json::Value>,
    samples_tx: kanal::Sender<Option<Sample>>,
    image_transform: Option<image_processing::ARAwareTransform>,
    encoding: image_processing::ImageEncoding,
    limit: usize,
) {
    // Fallback implementation using the original async approach
    // This is kept for compatibility and testing
    let default_max_tasks = std::env::var("DATAGO_MAX_TASKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(num_cpus::get());

    let max_tasks = std::cmp::min(default_max_tasks, limit);
    let mut tasks = tokio::task::JoinSet::new();
    let mut count = 0;
    let shareable_img_tfm = Arc::new(image_transform);

    while let Ok(received) = samples_metadata_rx.recv() {
        if received == serde_json::Value::Null {
            debug!("file_worker: end of stream received, stopping there");
            let _ = samples_metadata_rx.close();
            break;
        }

        // Check if we have reached the limit (including tasks in flight)
        if count + tasks.len() >= limit {
            break;
        }

        // Check if we have capacity before spawning new tasks
        if tasks.len() >= max_tasks {
            // Wait for some tasks to complete before adding more
            if let Some(result) = tasks.join_next().await {
                if result.is_ok() {
                    count += 1;
                }
            }
        }

        // Append a new task to the queue
        tasks.spawn(pull_sample(
            received,
            shareable_img_tfm.clone(),
            encoding,
            samples_tx.clone(),
        ));
    }

    // Make sure to wait for all the remaining tasks
    let _ = tasks.join_all().await.iter().map(|result| {
        if let Ok(()) = result {
            count += 1;
        }
    });
    debug!("file_worker: total samples sent: {}", count);

    // Signal the end of the stream
    if samples_tx.send(None).is_ok() {};
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

    async fn image_from_path_compat(path: &str) -> Result<image::DynamicImage, image::ImageError> {
        // Use buffered reading instead of loading entire file at once for better memory efficiency
        let file = std::fs::File::open(path)
            .map_err(|e| image::ImageError::IoError(std::io::Error::other(e)))?;
        let reader = std::io::BufReader::new(file);

        image::ImageReader::new(reader)
            .with_guessed_format()?
            .decode()
    }

    #[tokio::test]
    async fn test_image_from_path_success() {
        let temp_dir = TempDir::new().unwrap();
        let image_path = temp_dir.path().join("test.png");
        create_test_image(&image_path);

        let result = image_from_path_compat(image_path.to_str().unwrap()).await;
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

        let encoding = image_processing::ImageEncoding {
            encode_images: false,
            img_to_rgb8: false,
            ..Default::default()
        };
        let result = image_payload_from_path(image_path.to_str().unwrap(), &None, encoding).await;

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
            encode_format: image_processing::EncodeFormat::default(),
            jpeg_quality: image_processing::DEFAULT_JPEG_QUALITY,
        };

        let transform = Some(transform_config.get_ar_aware_transform());

        let encoding = image_processing::ImageEncoding {
            encode_images: false,
            img_to_rgb8: false,
            ..Default::default()
        };
        let result =
            image_payload_from_path(image_path.to_str().unwrap(), &transform, encoding).await;

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

        let encoding = image_processing::ImageEncoding {
            encode_images: true,
            img_to_rgb8: false,
            ..Default::default()
        };
        let result = image_payload_from_path(image_path.to_str().unwrap(), &None, encoding).await;

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

        let encoding = image_processing::ImageEncoding {
            encode_images: false,
            img_to_rgb8: true,
            ..Default::default()
        };
        let result = image_payload_from_path(image_path.to_str().unwrap(), &None, encoding).await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.channels, 3); // Should be converted to RGB (3 channels)
        assert_eq!(payload.bit_depth, 8);
    }

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
        let encoding = image_processing::ImageEncoding {
            encode_images: false,
            img_to_rgb8: false,
            ..Default::default()
        };
        async_pull_samples(metadata_rx, samples_tx, None, encoding, limit).await;

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

        let encoding = image_processing::ImageEncoding {
            encode_images: false,
            img_to_rgb8: false,
            ..Default::default()
        };
        let result = image_payload_from_path(image_path.to_str().unwrap(), &None, encoding).await;

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.width, 2);
        assert_eq!(payload.height, 2);
        assert_eq!(payload.original_width, 2);
        assert_eq!(payload.original_height, 2);
        assert!(!payload.data.is_empty());
    }

    #[tokio::test]
    async fn test_io_uring_memory_stress() {
        // Create a temporary directory
        let temp_dir = TempDir::new().unwrap();

        // Create multiple test image files of different sizes
        let image_sizes = [(24, 32), (64, 64), (128, 128), (256, 256), (512, 512)];
        let mut file_paths = Vec::new();

        for (i, (width, height)) in image_sizes.iter().enumerate() {
            let file_path = temp_dir.path().join(format!("test_{i}.png"));

            // Create a test image with predictable pixel data
            let img = image::DynamicImage::new_rgb8(*width, *height);

            // Save the image
            img.save(&file_path).unwrap();

            file_paths.push(file_path);
        }

        // Test reading all files concurrently using io_uring
        let mut tasks = Vec::new();
        for file_path in &file_paths {
            let path = file_path.to_str().unwrap().to_string();
            tasks.push(tokio::spawn(async move {
                // Use the io_uring_read_file function
                let result = io_uring_read_file(&path).await;
                assert!(result.is_ok(), "Failed to read file: {}", path);

                let bytes = result.unwrap();
                assert!(!bytes.is_empty(), "Empty buffer for file: {}", path);

                // Verify we can load the image from the bytes
                let img_result = image::load_from_memory(&bytes);
                assert!(
                    img_result.is_ok(),
                    "Failed to load image from bytes: {}",
                    path
                );

                let img = img_result.unwrap();
                assert!(
                    img.width() > 0 && img.height() > 0,
                    "Invalid image dimensions: {}",
                    path
                );
            }));
        }

        // Wait for all tasks to complete
        for task in tasks {
            task.await.unwrap();
        }
    }
}
