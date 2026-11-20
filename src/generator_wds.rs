use crate::image_processing::{ARAwareTransform, ImageEncoding};
use crate::source::{Source, SourceContext};
use crate::structs::{new_shared_client, BinaryFile, Sample, SharedClient, TarballSample};
use crate::worker_utils::{max_retries_from_env, max_tasks_from_env};
use crate::worker_wds;

use async_tar::Archive;
use log::{debug, error, info, warn};
use rand::seq::SliceRandom;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use bracoxide::explode;
use futures::AsyncReadExt;
use futures::StreamExt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use tokio::io::BufReader;
use tokio_util::compat::TokioAsyncReadCompatExt;
use tokio_util::io::StreamReader;
use tokio_util::sync::CancellationToken; // For grouping, if more complex grouping is needed

fn default_reference_image_type() -> String {
    "jpg".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceWebDatasetConfig {
    pub url: String,

    #[serde(default = "default_reference_image_type")]
    pub reference_image_type: String,

    #[serde(default)]
    pub random_sampling: bool,

    #[serde(default)]
    pub concurrent_downloads: usize,

    #[serde(default)]
    pub auth_token: String,

    #[serde(default)]
    pub rank: usize,

    #[serde(default)]
    pub world_size: usize,
}

fn hash_fn(key: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

// Match WebDataset's base_plus_ext: retain directories and split at the first
// dot in the final path component. Dots in directory names are part of the key.
pub(crate) fn key_and_field(path: &str) -> Option<(&str, &str)> {
    let filename = path.rsplit('/').next()?;
    let (stem, field) = filename.split_once('.')?;
    if stem.is_empty() {
        return None;
    }
    Some((&path[..path.len() - field.len() - 1], field))
}

async fn pull_tarballs(
    shared_client: Arc<SharedClient>,
    url: serde_json::Value,
    samples_metadata_tx: kanal::Sender<TarballSample>,
    config: SourceWebDatasetConfig,
) -> Result<(), String> {
    // Given the url to a given tarball, we'll download it and submit the samples on the fly to the worker pool
    debug!(
        "dispatch_shards: downloading a new TarballSample {:?}",
        url.as_str()
    );

    let auth_token = if config.auth_token.is_empty() {
        None
    } else {
        Some(config.auth_token.as_str())
    };
    let url = url.as_str().unwrap();

    // We use a shared client to make it possible to limit the number of outstanding connections
    let _permit = shared_client.semaphore.acquire().await;
    let mut request_builder = shared_client.client.get(url);
    if let Some(token) = auth_token {
        request_builder = request_builder.bearer_auth(token);
    }

    // Grab an async byte stream from the request, we'll try to untar the results on the fly
    let response = request_builder.send().await;
    let response = match response {
        Ok(resp) => resp,
        Err(e) => {
            error!("Failed to send request for {}: {}", url, e);
            return Err(format!("Failed to send request: {}", e));
        }
    };

    if !response.status().is_success() {
        error!(
            "Failed to download TarballSample {}: HTTP {}",
            url,
            response.status()
        );
        return Err(format!(
            "Failed to download TarballSample {}: HTTP {}",
            url,
            response.status()
        ));
    }

    // Convert the byte stream to an AsyncRead
    let byte_stream = response.bytes_stream();
    let stream_reader =
        StreamReader::new(byte_stream.map(|res_bytes| res_bytes.map_err(std::io::Error::other)));

    // Wrap in BufReader for the async Tar reader
    let buf_reader = BufReader::new(stream_reader);

    // Create a tar archive reader from the decompressed stream
    let archive = Archive::new(buf_reader.compat());

    let mut entries = archive
        .entries()
        .map_err(|e| format!("Failed to fetch TarballSample: {e}"))?; // This returns a stream

    let mut current_sample_key: Option<String> = None;
    let mut current_files_for_sample = TarballSample::new(url.to_string());

    while let Some(entry_result) = entries.next().await {
        let mut entry =
            entry_result.map_err(|e| format!("Failed to read TarballSample entry: {e}"))?;

        let header_path = entry
            .path()
            .map_err(|e| format!("Error considering TarballSample content {e}"))?
            .into_owned();
        let filename = header_path.to_string_lossy().into_owned();

        let Some((entry_key, _)) = key_and_field(&filename) else {
            continue;
        };
        let entry_key = entry_key.to_owned();

        // If we have a > 1 world size, we need to dispatch the samples to the correct rank
        // We'll do this by hashing the key and checking if it matches our rank, we skip otherwise
        if config.world_size > 1 {
            let hash = hash_fn(&entry_key);
            let target_rank = (hash % config.world_size as u64) as usize;
            if target_rank != config.rank {
                continue;
            }
        }

        // If the key changes, the previous sample is complete
        if current_sample_key.is_none() {
            current_sample_key = Some(entry_key.clone());
        }
        if current_sample_key.as_ref() != Some(&entry_key) && !current_files_for_sample.is_empty() {
            // Sort the files in the sample so that the reference image is first
            // This will make sure that it is always processed first, and will serve as a reference
            // for the final aspect ratio bucket
            current_files_for_sample.content.sort_by_key(|file| {
                !key_and_field(&file.filename).is_some_and(|(_, field)| {
                    field.eq_ignore_ascii_case(&config.reference_image_type)
                })
            });

            if samples_metadata_tx.send(current_files_for_sample).is_err() {
                debug!("dispatch_shards (streaming): samples_metadata_tx channel closed.");
                let _ = samples_metadata_tx.close(); // Make sure that we close on both ends
                return Ok(());
            }

            // Start a new sample
            current_files_for_sample = TarballSample::new(url.to_string());
            current_sample_key = Some(entry_key.clone());
        }

        let mut buffer = Vec::new();
        entry
            .read_to_end(&mut buffer)
            .await
            .map_err(|e| format!("Failed to read TarballSample {e}"))?; // Read the content of the current file

        current_files_for_sample.add(BinaryFile { filename, buffer });
        debug!(
            "dispatch_shards (streaming): processed entry {:?}, key: {:?}",
            Path::new(&current_files_for_sample.content.last().unwrap().filename)
                .file_name()
                .unwrap_or_default(),
            entry_key
        );
    }
    // Send the last collected sample if any
    if !current_files_for_sample.content.is_empty()
        && samples_metadata_tx.send(current_files_for_sample).is_err()
        && !samples_metadata_tx.is_disconnected()
    {
        return Err("Failed to send last sample".into());
    }

    debug!("dispatch_shards (streaming): finished processing TarballSample {url}");
    Ok(())
}

async fn pull_tarballs_task(
    shared_client: Arc<SharedClient>,
    url: serde_json::Value,
    retries: u8,
    samples_metadata_tx: kanal::Sender<TarballSample>,
    config: SourceWebDatasetConfig,
) -> Result<(), String> {
    let mut attempt = 0;
    let mut last_error = String::new();

    while attempt < retries {
        match pull_tarballs(
            shared_client.clone(),
            url.clone(),
            samples_metadata_tx.clone(),
            config.clone(),
        )
        .await
        {
            Ok(_) => {
                return Ok(()); // Successfully pulled the tarball
            }
            Err(e) => {
                attempt += 1;
                debug!("Error pulling TarballSample: {e}. Attempt {attempt}/{retries}");
                last_error = e;
                if samples_metadata_tx.is_disconnected() {
                    debug!(
                        "dispatch_shards: samples_metadata_tx channel closed, stopping retries."
                    );
                    return Err("Channel closed".into());
                }
            }
        }
    }
    Err(format!(
        "Failed to pull TarballSample {url} after {retries} attempts: {last_error}"
    ))
}

async fn get_url_list(
    shared_client: &Arc<SharedClient>,
    config: &SourceWebDatasetConfig,
) -> Result<Vec<serde_json::Value>, String> {
    let _permit = shared_client.semaphore.acquire().await;

    // Either ping the url to get the pages, or use the {...} syntax
    if config.url.contains("{") {
        // URL should look like this:
        // https://storage.googleapis.com/webdataset/testdata/publaynet-train-{000000..000009}.tar
        // We need to parse the URL and generate all the possible URLs
        // for instance https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar

        // Extract the pattern within curly braces
        let urls =
            explode(&config.url).map_err(|error| format!("invalid shard pattern: {error}"))?;

        Ok(urls
            .iter()
            .map(|url| serde_json::Value::String(url.clone()))
            .collect())
    } else {
        // Given the url, list all the available webdataset files
        let request = reqwest::Request::new(
            reqwest::Method::GET,
            Url::parse(&config.url).map_err(|e| format!("Failed parsing url: {e}"))?,
        );

        let response = shared_client
            .client
            .execute(request)
            .await
            .map_err(|e| format!("Failed parsing reply: {e}"))?
            .error_for_status()
            .map_err(|e| e.to_string())?;

        let response_text = response
            .text()
            .await
            .map_err(|e| format!("Failed parsing reply: {e}"))?;
        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|error| format!("invalid shard listing: {error}"))?;

        // Parse all the "items" in the response
        if let Some(items) = response_json.get("items") {
            Ok(items
                .as_array()
                .ok_or("shard listing items must be an array")?
                .iter()
                .filter_map(|item| item.get("mediaLink").cloned())
                .collect())
        } else if response_json
            .get("kind")
            .and_then(serde_json::Value::as_str)
            == Some("storage#objects")
        {
            Ok(Vec::new()) // GCS omits items for an empty listing.
        } else {
            Err("shard listing must contain an items array".into())
        }
    }
}

async fn tasks_from_shards(
    shared_client: Arc<SharedClient>,
    samples_metadata_tx: kanal::Sender<TarballSample>,
    config: &SourceWebDatasetConfig,
    retries: u8,
) -> Result<serde_json::Value, String> {
    match get_url_list(&shared_client, config).await {
        Ok(mut task_list) => {
            // Shuffle the items if needed
            if config.random_sampling {
                task_list.shuffle(&mut rand::rng());
            }

            // Now submit all the tasks, making sure that not too many of them are in flight
            let mut tasks = tokio::task::JoinSet::new();
            let response_json = serde_json::Value::Null;
            let mut count = 0;

            info!("WDS: Using {} download tasks", config.concurrent_downloads);

            for url in task_list {
                // Escape out if the channel is closed
                if samples_metadata_tx.is_disconnected() {
                    debug!(
                        "dispatch_shards: channel is closed, enough samples probably. Bailing out"
                    );
                    break;
                }

                // All good, submit a new async task
                tasks.spawn(pull_tarballs_task(
                    shared_client.clone(),
                    url,
                    retries,
                    samples_metadata_tx.clone(),
                    config.clone(),
                ));

                // Some bookkeeping, to limit the number of tasks in flight
                // we'll wait for the first one to finish before adding a new one
                if tasks.len() >= config.concurrent_downloads {
                    match tasks.join_next().await {
                        Some(res) => {
                            match res.map_err(|error| format!("shard task failed: {error}"))? {
                                Ok(_) => {
                                    debug!("dispatch_shards: task completed successfully");
                                    count += 1;
                                }
                                Err(e) => {
                                    // Logging as debug, could be that channels are closed
                                    debug!("dispatch_shards: task returned error: {e:?}");
                                    return Err(e);
                                }
                            }
                        }
                        None => {
                            // Task was cancelled or panicked
                            error!("dispatch_shards: task was cancelled or panicked");
                        }
                    }
                }

                count += 1;
            }

            // Consume all the remaining tasks
            while let Some(result) = tasks.join_next().await {
                match result.map_err(|error| format!("shard task failed: {error}"))? {
                    Ok(_) => {
                        debug!("dispatch_shards: task completed successfully");
                        count += 1;
                    }
                    Err(e) => {
                        debug!("dispatch_shards: task returned error: {e}");
                        return Err(e);
                    }
                }
            }

            // Bookkeeping and report
            if count == 0 {
                warn!("No items found in the response");
            }
            debug!("Served {count} items from the bucket");

            Ok(response_json)
        }
        Err(e) => {
            warn!("Failed to get URL list: {e}");
            Err(e) // Return a JoinError with the error message
        }
    }
}

fn query_shards_and_dispatch(
    shared_client: Arc<SharedClient>,
    samples_metadata_tx: kanal::Sender<TarballSample>,
    source_config: SourceWebDatasetConfig,
    cancel: CancellationToken,
) -> Result<(), String> {
    // List all the shards from the bucket
    // for each of them, start an async task to download the TarballSample and unpack it in memory
    // Then, for each sample in the tarball (multiple payloads typically), send it to the processing channel

    let max_retries = max_retries_from_env();

    // Use more threads for the download runtime to handle increased concurrency
    let download_threads = std::cmp::max(4, source_config.concurrent_downloads);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(download_threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => Ok(()),
                result = tasks_from_shards(
                    shared_client.clone(),
                    samples_metadata_tx,
                    &source_config,
                    max_retries,
                ) => {
                    result.map(|_| ())
                }
            }
        })
}

const DEFAULT_CONCURRENT_DOWNLOADS: usize = 8;

/// Samples stored as groups of files in tar shards, streamed over HTTP.
pub(crate) struct WebDatasetSource {
    config: SourceWebDatasetConfig,
    http: Arc<SharedClient>,
    image_transform: Arc<Option<ARAwareTransform>>,
    encoding: ImageEncoding,
}

impl Source for WebDatasetSource {
    const NAME: &'static str = "WebDataset";
    type Config = SourceWebDatasetConfig;
    /// The files of one sample, already downloaded and untarred.
    type Item = TarballSample;

    fn new(mut config: Self::Config, context: &SourceContext) -> Self {
        if config.concurrent_downloads == 0 {
            info!("WDS: Defaulting to {DEFAULT_CONCURRENT_DOWNLOADS} concurrent_downloads");
            config.concurrent_downloads = DEFAULT_CONCURRENT_DOWNLOADS;
        }
        Self {
            config,
            http: Arc::new(new_shared_client(context.max_connections)),
            image_transform: context.image_transform.clone(),
            encoding: context.encoding,
        }
    }

    fn enumerate(
        &self,
        items: kanal::Sender<Self::Item>,
        cancel: &CancellationToken,
    ) -> Result<(), String> {
        query_shards_and_dispatch(
            self.http.clone(),
            items,
            self.config.clone(),
            cancel.clone(),
        )
    }

    async fn load(&self, item: Self::Item) -> Result<Sample, ()> {
        worker_wds::process_sample(
            item,
            self.image_transform.clone(),
            self.encoding,
            self.config.reference_image_type.clone(),
        )
        .await
    }

    fn max_tasks(&self) -> usize {
        let cpus = num_cpus::get();
        max_tasks_from_env(cpus).min(cpus * 4)
    }

    fn queue_capacity(samples_buffer: usize) -> usize {
        // Shards arrive in bursts: keep room for them even with a small buffer.
        (samples_buffer * 2).max(128)
    }
}

// -------- Unit tests --------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::DatagoClient;
    use crate::error::DatagoError;
    use serde_json::json;
    use std::collections::HashSet;
    use std::io::Cursor;
    use std::time::Duration;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const SAMPLE_COUNT: usize = 16;
    const READ_TIMEOUT: Duration = Duration::from_secs(5);

    fn shard(start: usize, end: usize) -> Vec<u8> {
        let mut jpeg = Cursor::new(Vec::new());
        image::RgbImage::from_pixel(4, 3, image::Rgb([11, 22, 33]))
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        let mut archive = tar::Builder::new(Vec::new());
        for id in start..end {
            for (extension, bytes) in [("jpg", jpeg.get_ref().as_slice()), ("cls", b"1".as_slice())]
            {
                let mut header = tar::Header::new_gnu();
                header.set_size(bytes.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                archive
                    .append_data(&mut header, format!("{id:04}.{extension}"), bytes)
                    .unwrap();
            }
        }
        archive.into_inner().unwrap()
    }

    async fn fixture(fail_second_shard: bool) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/listing"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "items": [
                    {"mediaLink": format!("{}/first.tar", server.uri())},
                    {"mediaLink": format!("{}/second.tar", server.uri())},
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/first.tar"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shard(0, SAMPLE_COUNT / 2)))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/second.tar"))
            .respond_with(if fail_second_shard {
                ResponseTemplate::new(404)
            } else {
                ResponseTemplate::new(200).set_body_bytes(shard(SAMPLE_COUNT / 2, SAMPLE_COUNT))
            })
            .mount(&server)
            .await;
        server
    }

    fn client(url: String, random_sampling: bool, rank: usize, world_size: usize) -> DatagoClient {
        DatagoClient::new(
            json!({
                "source_type": "webdataset",
                "source_config": {
                    "url": url, "random_sampling": random_sampling,
                    "concurrent_downloads": 2, "rank": rank, "world_size": world_size,
                },
                // Consume the finite fixture through EOS, not an early-limit exit.
                "limit": SAMPLE_COUNT * 2, "samples_buffer_size": 1,
            })
            .to_string(),
        )
    }

    fn collect(client: &DatagoClient) -> Vec<Sample> {
        let mut samples = Vec::new();
        while let Some(sample) = client
            .try_get_sample_timeout(READ_TIMEOUT)
            .expect("WebDataset fixture pipeline failed")
        {
            let payload = sample.image.get_payload();
            assert_eq!(
                (
                    payload.width,
                    payload.height,
                    payload.channels,
                    payload.bit_depth
                ),
                (4, 3, 3, 8)
            );
            assert!(!payload.data.is_empty());
            samples.push(sample);
            assert!(
                samples.len() <= SAMPLE_COUNT,
                "fixture delivered duplicate samples"
            );
        }
        samples
    }

    fn ids(samples: &[Sample]) -> HashSet<String> {
        let ids: HashSet<_> = samples.iter().map(|sample| sample.id.clone()).collect();
        assert_eq!(ids.len(), samples.len(), "duplicate sample IDs");
        ids
    }

    // WebDataset keys run up to the first "." of the file name, so "0000.seg.png"
    // belongs to sample "0000", next to "0000.jpg".
    fn compound_extension_shard() -> Vec<u8> {
        let encode = |format| {
            let mut bytes = Cursor::new(Vec::new());
            image::RgbImage::from_pixel(4, 3, image::Rgb([11, 22, 33]))
                .write_to(&mut bytes, format)
                .unwrap();
            bytes.into_inner()
        };
        let (jpg, png) = (
            encode(image::ImageFormat::Jpeg),
            encode(image::ImageFormat::Png),
        );
        let mut archive = tar::Builder::new(Vec::new());
        for (filename, bytes) in [
            ("0000.jpg", jpg.as_slice()),
            ("0000.seg.png", png.as_slice()),
            ("0001.jpg", jpg.as_slice()),
            ("0001.cls", b"1".as_slice()),
            ("0001.depth.png", png.as_slice()),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, filename, bytes).unwrap();
        }
        archive.into_inner().unwrap()
    }

    async fn compound_extension_fixture() -> MockServer {
        shard_fixture(compound_extension_shard()).await
    }

    async fn shard_fixture(bytes: Vec<u8>) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/listing"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "items": [{"mediaLink": format!("{}/compound.tar", server.uri())}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/compound.tar"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
            .mount(&server)
            .await;
        server
    }

    #[test]
    fn webdataset_keys_preserve_directories() {
        assert_eq!(key_and_field("0000.jpg"), Some(("0000", "jpg")));
        assert_eq!(
            key_and_field("dir.v1/nested/0000.seg.png"),
            Some(("dir.v1/nested/0000", "seg.png"))
        );
        assert_eq!(key_and_field("dir.v1/no_extension"), None);
        assert_eq!(key_and_field("dir/.hidden.jpg"), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn directory_samples_keep_distinct_ids_fields_and_ranks() {
        let mut png = Cursor::new(Vec::new());
        image::RgbImage::from_pixel(4, 3, image::Rgb([11, 22, 33]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let keys = ["a.v1/0000", "b/0000"];
        let mut archive = tar::Builder::new(Vec::new());
        for key in keys {
            // Put the auxiliary image first to verify exact reference selection.
            for (field, bytes) in [
                ("seg.png", png.get_ref().as_slice()),
                ("png", png.get_ref().as_slice()),
                ("json", b"base".as_slice()),
                ("caption.json", b"caption".as_slice()),
                ("metadata.json", b"metadata".as_slice()),
            ] {
                let mut header = tar::Header::new_gnu();
                header.set_size(bytes.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                archive
                    .append_data(&mut header, format!("{key}.{field}"), bytes)
                    .unwrap();
            }
        }
        let server = shard_fixture(archive.into_inner().unwrap()).await;
        for world_size in [1, 3] {
            let mut delivered = HashSet::new();
            for rank in 0..world_size {
                let client = DatagoClient::new(
                    json!({
                        "source_type": "webdataset",
                        "source_config": {
                            "url": format!("{}/listing", server.uri()),
                            "reference_image_type": "png",
                            "concurrent_downloads": 1, "rank": rank, "world_size": world_size,
                        },
                        "limit": 10, "samples_buffer_size": 1,
                    })
                    .to_string(),
                );
                let samples = collect(&client);
                let expected: HashSet<_> = keys
                    .iter()
                    .filter(|key| hash_fn(key) % world_size as u64 == rank as u64)
                    .map(|key| key.to_string())
                    .collect();
                assert_eq!(ids(&samples), expected);
                for sample in samples {
                    assert!(delivered.insert(sample.id.clone()));
                    assert_eq!(sample.additional_images.len(), 1);
                    assert!(sample
                        .additional_images
                        .contains_key(&format!("{}.seg.png", sample.id)));
                    assert_eq!(sample.attributes.len(), 3);
                    assert_eq!(sample.attributes["json"], "base");
                    assert_eq!(sample.attributes["caption.json"], "caption");
                    assert_eq!(sample.attributes["metadata.json"], "metadata");
                }
            }
            assert_eq!(delivered, keys.map(String::from).into_iter().collect());
        }
    }

    #[test]
    fn test_urls_from_pattern() {
        let urls = explode("https://example.com/train-{000000..000009}.tar").unwrap();
        assert_eq!(urls.len(), 10);
        assert_eq!(urls[0], "https://example.com/train-000000.tar");
        assert_eq!(urls[9], "https://example.com/train-000009.tar");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webdataset_dispatch() {
        let server = fixture(false).await;
        for random_sampling in [false, true] {
            let config = SourceWebDatasetConfig {
                url: format!("{}/listing", server.uri()),
                reference_image_type: "jpg".into(),
                random_sampling,
                concurrent_downloads: 2,
                rank: 0,
                world_size: 1,
                auth_token: String::new(),
            };
            let http = Arc::new(new_shared_client(2));
            let (sender, receiver) = kanal::bounded(2);
            let cancel = CancellationToken::new();
            let feeder_cancel = cancel.clone();
            let feeder = std::thread::spawn(move || {
                query_shards_and_dispatch(http, sender, config, feeder_cancel)
            });
            let received = (0..SAMPLE_COUNT)
                .map(|_| receiver.recv_timeout(READ_TIMEOUT))
                .collect::<Result<Vec<_>, _>>();
            let end = received
                .as_ref()
                .ok()
                .map(|_| receiver.recv_timeout(READ_TIMEOUT));
            // Always shut down and join before asserting, including on failure.
            cancel.cancel();
            receiver.close().unwrap();
            feeder.join().unwrap().expect("fixture feeder failed");
            // The feeder returning drops the last sender, which closes the
            // queue: there is no marker item, and no wait for the timeout.
            use kanal::ReceiveErrorTimeout::{Closed, SendClosed};
            assert!(matches!(
                end.expect("fixture metadata read failed"),
                Err(Closed | SendClosed)
            ));
            for sample in received.expect("fixture feeder did not deliver metadata") {
                assert_eq!(sample.content.len(), 2);
                assert!(sample.iter().all(|file| !file.buffer.is_empty()));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webdataset_orchestrate() {
        let server = fixture(false).await;
        let expected: HashSet<_> = (0..SAMPLE_COUNT).map(|id| format!("{id:04}")).collect();
        for random_sampling in [false, true] {
            for _ in 0..2 {
                let client = client(format!("{}/listing", server.uri()), random_sampling, 0, 1);
                let samples = collect(&client);
                assert_eq!(ids(&samples), expected);
            }
        }
        // Random sampling and concurrent decoding need not produce different
        // orders on every run. The contract is complete, unique delivery.
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webdataset_ranks() {
        let server = fixture(false).await;
        let mut all_ids = HashSet::new();
        for rank in 0..2 {
            let client = client(format!("{}/listing", server.uri()), false, rank, 2);
            let samples = collect(&client);
            let actual = ids(&samples);
            let expected: HashSet<_> = (0..SAMPLE_COUNT)
                .map(|id| format!("{id:04}"))
                .filter(|id| hash_fn(id) % 2 == rank as u64)
                .collect();
            assert_eq!(actual, expected);
            assert!(all_ids.is_disjoint(&actual));
            all_ids.extend(actual);
        }
        assert_eq!(all_ids.len(), SAMPLE_COUNT);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_shard_is_reported_as_source_failure_not_exhaustion() {
        let server = fixture(true).await;
        let client = client(format!("{}/listing", server.uri()), false, 0, 1);
        loop {
            match client.try_get_sample_timeout(READ_TIMEOUT) {
                Ok(Some(_)) => {} // A successful shard may publish first.
                Err(DatagoError::Source(message)) => {
                    assert!(
                        message.contains("404"),
                        "missing HTTP failure context: {message}"
                    );
                    break;
                }
                other => panic!("expected a shard failure, got {other:?}"),
            }
        }
        assert!(client.try_get_sample().unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn compound_extensions_stay_in_their_sample() {
        let server = compound_extension_fixture().await;
        let config = SourceWebDatasetConfig {
            url: format!("{}/listing", server.uri()),
            reference_image_type: "jpg".into(),
            random_sampling: false,
            concurrent_downloads: 1,
            rank: 0,
            world_size: 1,
            auth_token: String::new(),
        };
        let (sender, receiver) = kanal::bounded(8);
        pull_tarballs(
            Arc::new(new_shared_client(1)),
            json!(format!("{}/compound.tar", server.uri())),
            sender.clone(),
            config,
        )
        .await
        .unwrap();

        let mut groups = Vec::new();
        while let Some(sample) = receiver.try_recv().unwrap() {
            let mut filenames: Vec<_> = sample.iter().map(|file| file.filename.clone()).collect();
            filenames.sort();
            groups.push(filenames);
        }
        assert_eq!(
            groups,
            [
                vec!["0000.jpg", "0000.seg.png"],
                vec!["0001.cls", "0001.depth.png", "0001.jpg"],
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn compound_extension_images_reach_additional_images() {
        let server = compound_extension_fixture().await;
        let client = client(format!("{}/listing", server.uri()), false, 0, 1);
        let mut additional: Vec<_> = collect(&client)
            .iter()
            .map(|sample| {
                let mut filenames: Vec<_> = sample.additional_images.keys().cloned().collect();
                filenames.sort();
                (sample.id.clone(), filenames)
            })
            .collect();
        additional.sort();
        assert_eq!(
            additional,
            [
                ("0000".to_string(), vec!["0000.seg.png".to_string()]),
                ("0001".to_string(), vec!["0001.depth.png".to_string()]),
            ]
        );
    }

    #[test]
    #[ignore = "opt-in smoke test against the external GCS dataset"]
    fn public_bucket_smoke() {
        for random_sampling in [false, true] {
            let client = client(
                "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/"
                    .into(),
                random_sampling,
                0,
                1,
            );
            for _ in 0..2 {
                assert!(client
                    .try_get_sample_timeout(Duration::from_secs(30))
                    .expect("public GCS pipeline failed")
                    .is_some());
            }
            client.stop();
        }
    }
}
