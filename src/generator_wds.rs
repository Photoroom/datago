use crate::client::DatagoClient;
use crate::structs::{new_shared_client, DatagoEngine, SharedClient, TarballContent, WDSContent};
use crate::worker_wds;

use async_tar::Archive;
use kanal::bounded;
use log::{debug, info, warn};
use rand::seq::SliceRandom;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;

use futures::AsyncReadExt;
use futures::StreamExt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use tokio::io::BufReader;
use tokio_util::compat::TokioAsyncReadCompatExt;
use tokio_util::io::StreamReader; // For grouping, if more complex grouping is needed

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceWebDatasetConfig {
    pub url: String,

    #[serde(default)]
    pub random_sampling: bool,

    #[serde(default)]
    pub max_concurrency: usize,

    #[serde(default)]
    pub auth_token: String,

    #[serde(default)]
    pub rank: usize,

    #[serde(default)]
    pub world_size: usize,
}

fn urls_from_pattern(url: &str) -> Vec<String> {
    // Extract the pattern within curly braces
    let pattern = url.split('{').nth(1).unwrap_or("");
    let pattern = pattern.split('}').next().unwrap_or("");

    // Split the pattern into parts
    let parts: Vec<&str> = pattern.split("..").collect();
    let start = parts[0].parse::<i32>().unwrap_or(0);
    let end = parts[1].parse::<i32>().unwrap_or(0);
    assert!(end >= start, "End must be greater than start");
    assert!(
        parts[0].len() == parts[1].len(),
        "Couldn't make sense of the URL pattern provided"
    );

    // Generate all the URLs, note that not all patterns will have the same number of digits
    let digit_count = parts[0].len();
    (start..=end)
        .map(|i| {
            url.replace(
                &format!("{{{}}}", pattern),
                &format!("{:0width$}", i, width = digit_count),
            )
        })
        .collect()
}

fn hash_fn(key: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

async fn pull_tarball(
    shared_client: Arc<SharedClient>,
    url: serde_json::Value,
    samples_metadata_tx: kanal::Sender<TarballContent>,
    config: SourceWebDatasetConfig,
) -> Result<(), String> {
    // Given the url to a given tarball, we'll download it and submit the samples on the fly to the worker pool
    debug!(
        "dispatch_shards: downloading a new tarball {:?}",
        url.as_str()
    );

    let auth_token = if config.auth_token.is_empty() {
        None
    } else {
        Some(config.auth_token.as_str())
    };
    let url = url.as_str().unwrap();

    // We use a shared client to make it possible to limit the number of outstanding connections
    let _permit = shared_client.semaphore.acquire();
    let mut request_builder = shared_client.client.get(url);
    if let Some(token) = auth_token {
        request_builder = request_builder.bearer_auth(token);
    }

    // Grab an async byte stream from the request, we'll try to untar the results on the fly
    let response = request_builder.send().await;
    if response.is_err() {
        return Err("Failed to send request".into());
    }
    let response = response.unwrap();
    if !response.status().is_success() {
        return Err(format!("Failed to download tarball: {}", response.status()));
    }

    let byte_stream = response.bytes_stream();

    // Convert the byte stream to an AsyncRead
    let stream_reader =
        StreamReader::new(byte_stream.map(|res_bytes| {
            res_bytes.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }));

    // Wrap in BufReader for the async Tar reader
    let buf_reader = BufReader::new(stream_reader);

    // Create a tar archive reader from the decompressed stream
    let archive = Archive::new(buf_reader.compat());

    let mut entries = archive
        .entries()
        .map_err(|e| format!("Failed to fetch tarball: {}", e))?; // This returns a stream

    let mut current_sample_key: Option<String> = None;
    let mut current_files_for_sample: TarballContent = Vec::new();

    while let Some(entry_result) = entries.next().await {
        let mut entry = entry_result.map_err(|e| format!("Failed to read tarball entry: {}", e))?;

        let header_path = entry
            .path()
            .map_err(|e| format!("Error considering tarball content {}", e))?
            .into_owned();
        let filename = header_path.to_string_lossy().into_owned();

        // Simple key extraction: basename without extension
        let entry_key = Path::new(&filename)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();

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
            if samples_metadata_tx.send(current_files_for_sample).is_err() {
                debug!("dispatch_shards (streaming): samples_metadata_tx channel closed.");
                return Err("Channel closed".into());
            }
            current_files_for_sample = Vec::new();
            current_sample_key = Some(entry_key.clone());
        }

        let mut buffer = Vec::new();
        entry
            .read_to_end(&mut buffer)
            .await
            .map_err(|e| format!("Failed to read tarball {}", e))?; // Read the content of the current file

        current_files_for_sample.push(WDSContent { filename, buffer });
        debug!(
            "dispatch_shards (streaming): processed entry {:?}, key: {:?}",
            Path::new(&current_files_for_sample.last().unwrap().filename)
                .file_name()
                .unwrap_or_default(),
            entry_key
        );
    }
    // Send the last collected sample if any
    if !current_files_for_sample.is_empty()
        && samples_metadata_tx.send(current_files_for_sample).is_err()
    {
        debug!("dispatch_shards (streaming): samples_metadata_tx channel closed for last sample.");
        return Err("Channel closed".into());
    }

    debug!(
        "dispatch_shards (streaming): finished processing tarball {}",
        url
    );
    Ok(())
}

async fn pull_tarball_task(
    shared_client: Arc<SharedClient>,
    url: serde_json::Value,
    samples_metadata_tx: kanal::Sender<TarballContent>,
    config: SourceWebDatasetConfig,
) -> Result<(), String> {
    let retries = 3;
    let mut attempt = 0;
    let mut success = false;

    while attempt < retries && !success {
        match pull_tarball(
            shared_client.clone(),
            url.clone(),
            samples_metadata_tx.clone(),
            config.clone(),
        )
        .await
        {
            Ok(_) => {
                success = true;
            }
            Err(e) => {
                attempt += 1;
                debug!(
                    "Error pulling tarball: {}. Attempt {}/{}",
                    e, attempt, retries
                );
            }
        }
    }
    if !success {
        return Err(format!("Failed to pull tarball after {} attempts", retries));
    }
    debug!("dispatch_shards: finished pulling tarball");
    Ok(())
}

async fn get_url_list(
    shared_client: &Arc<SharedClient>,
    config: &SourceWebDatasetConfig,
) -> Result<Vec<serde_json::Value>, String> {
    let _permit = shared_client.semaphore.acquire();

    // Either ping the url to get the pages, or use the {...} syntax
    if config.url.contains("{") {
        // URL should look like this:
        // https://storage.googleapis.com/webdataset/testdata/publaynet-train-{000000..000009}.tar
        // We need to parse the URL and generate all the possible URLs
        // for instance https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar

        // Extract the pattern within curly braces
        let urls = urls_from_pattern(&config.url);

        Ok(urls
            .iter()
            .map(|url| serde_json::Value::String(url.clone()))
            .collect())
    } else {
        assert!(config.url.contains("https://storage.googleapis.com/"));

        // Given the url, list all the available webdataset files
        let request = reqwest::Request::new(
            reqwest::Method::GET,
            Url::parse(&config.url).map_err(|e| format!("Failed parsing url: {}", e))?,
        );

        let response = shared_client
            .client
            .execute(request)
            .await
            .map_err(|e| format!("Failed parsing reply: {}", e))?;

        let response_text = response
            .text()
            .await
            .map_err(|e| format!("Failed parsing reply: {}", e))?;
        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).unwrap_or(serde_json::Value::Null);

        // Parse all the "items" in the response
        if let Some(items) = response_json.get("items") {
            Ok(items
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| item.get("mediaLink").cloned())
                .collect())
        } else {
            // If the response is empty, return an empty vector
            warn!("dispatch_shards: no items found in the response");
            Err("dispatch_shards: no items found in the response".into())
        }
    }
}

async fn tasks_from_shards(
    shared_client: Arc<SharedClient>,
    samples_metadata_tx: kanal::Sender<TarballContent>,
    config: &SourceWebDatasetConfig,
) -> Result<serde_json::Value, ()> {
    match get_url_list(&shared_client, config).await {
        Ok(mut task_list) => {
            // Shuffle the items if needed
            if config.random_sampling {
                task_list.shuffle(&mut rand::rng());
            }

            // Now submit all the tasks, making sure that too many of them are not in flight
            let mut tasks = VecDeque::new();
            let response_json = serde_json::Value::Null;
            let mut count = 0;

            for url in task_list {
                tasks.push_back(tokio::spawn(pull_tarball_task(
                    shared_client.clone(),
                    url,
                    samples_metadata_tx.clone(),
                    config.clone(),
                )));

                // Some bookkeeping, to limit the number of tasks in flight
                // we'll wait for the first one to finish before adding a new one
                if tasks.len() >= config.max_concurrency {
                    // Catch an early stop in this case, can be that the stream is closed
                    // and we don't want to keep going forever
                    let res = tasks.pop_front().unwrap().await;
                    debug!(
                        "dispatch_shards: waiting for a task to finish. Got {:?}",
                        res
                    );
                    // Note that the double check is needed, as the task can fail the spawn or return an error
                    if res.is_err() || res.unwrap().is_err() {
                        debug!("dispatch_shards: task failed, stopping there");
                        break;
                    }
                }

                count += 1;
            }

            // Consume all the remaining tasks
            while !tasks.is_empty() {
                let _ = tasks.pop_front().unwrap().await;
            }

            // Bookkeeping and report
            if count == 0 {
                warn!("No items found in the response");
            }
            debug!("Served {} items from the bucket", count);

            // Send an empty value to signal the end of the stream
            if samples_metadata_tx.send(vec![]).is_err() {
                debug!("list_shards: stream already closed, wrapping up");
            }

            Ok(response_json)
        }
        Err(e) => {
            warn!("Failed to get URL list: {}", e);
            Err(())
        }
    }
}

fn query_shards_and_dispatch(
    shared_client: Arc<SharedClient>,
    samples_metadata_tx: kanal::Sender<TarballContent>,
    source_config: SourceWebDatasetConfig,
) {
    // List all the shards from the bucket
    // for each of them, start an async task to download the tarball
    // and unpack it in memory
    // Then, for each sample in the tarball, send it to the channel
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(source_config.max_concurrency)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let _ =
                tasks_from_shards(shared_client.clone(), samples_metadata_tx, &source_config).await;
            // FIXME: handle errors
        });
}

// ---- Global orchestration ---------
pub fn orchestrate(client: &DatagoClient) -> DatagoEngine {
    // Allocate all the message passing pipes
    let (samples_metadata_tx, samples_metadata_rx) = bounded::<TarballContent>(32);
    let (samples_tx, samples_rx) = bounded(client.samples_buffer);

    info!("Using webdataset as source");

    // Convert the source_config to a SourceWebDatasetConfig
    let mut source_config: SourceWebDatasetConfig =
        serde_json::from_value(client.source_config.clone()).unwrap();

    if source_config.max_concurrency == 0 {
        info!("WDS: Defaulting to 8 max_concurrency");
        source_config.max_concurrency = 8;
    }

    // List the contents of the bucket and feed the workers
    let http_client = Arc::new(new_shared_client(client.max_connections));

    let feeder = Some(thread::spawn(move || {
        query_shards_and_dispatch(http_client, samples_metadata_tx, source_config);
    }));

    // Kick the workers which deserialize all the payloads
    let image_transform = client.image_transform.clone();
    let encode_images = client.encode_images;
    let img_to_rgb8 = client.image_to_rgb8;
    let limit = client.limit;
    let samples_tx_worker = samples_tx.clone();
    let worker = Some(thread::spawn(move || {
        worker_wds::deserialize_samples(
            samples_metadata_rx,
            samples_tx_worker,
            image_transform,
            encode_images,
            img_to_rgb8,
            limit,
        );
    }));

    DatagoEngine {
        samples_rx,
        feeder,
        worker,
    }
}

// -------- Unit tests --------

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_urls_from_pattern() {
        let url = "https://storage.googleapis.com/webdataset/testdata/publaynet-train-{000000..000009}.tar";
        let urls = urls_from_pattern(url);
        print!("URLs: {:?}", urls.clone());
        assert_eq!(urls.len(), 10);
        assert_eq!(
            urls[0],
            "https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar"
        );
        assert_eq!(
            urls[9],
            "https://storage.googleapis.com/webdataset/testdata/publaynet-train-000009.tar"
        );
    }

    #[test]
    fn test_webdataset_dispatch() {
        let random_sampling = [false, true];

        for s in random_sampling {
            let config = SourceWebDatasetConfig {
                url:
                    "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/"
                        .into(),
                auth_token: "".into(),
                random_sampling: s,
                max_concurrency: 2,
                rank: 0,
                world_size: 1,
            };

            // Query the bucket, pull the workload
            let http_client = Arc::new(new_shared_client(2));
            let (samples_meta_tx, samples_meta_rx) = bounded::<TarballContent>(2);

            let mut count = 0;
            let limit = 2;
            let feeder = thread::spawn(move || {
                query_shards_and_dispatch(http_client, samples_meta_tx, config);
            });

            while count < limit {
                if let Ok(sample) = samples_meta_rx.recv() {
                    count += 1;

                    // Check that we got something
                    assert!(!sample.is_empty(), "Sample is empty");
                    for s in sample.iter() {
                        assert!(!s.filename.is_empty(), "Filename is empty");
                        assert!(!s.buffer.is_empty(), "Buffer is empty");
                    }
                } else {
                    debug!("No more items to receive");
                    break;
                }
            }

            debug!("Received {} items", count);
            let _ = samples_meta_rx.close();
            feeder.join().expect("Feeder thread panicked");

            assert!(count >= limit, "Not enough items found in the bucket");
        }
    }

    #[cfg(test)]
    use crate::structs::Sample;

    #[cfg(test)]
    use serde_json::json;
    #[test]
    fn test_webdataset_orchestrate() {
        fn get_samples(do_random_sampling: bool, n_samples: usize) -> Vec<Sample> {
            let client_config = json!({
                "source_config": {
                    "url": "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/",
                    "random_sampling": do_random_sampling,
                    "max_concurrency": 2
                },
                "limit": n_samples,
                "num_threads": 1,
                "max_connections": 1,
                "samples_buffer_size": 1
            });

            let mut client = DatagoClient::new(client_config.to_string());
            let engine = orchestrate(&client);
            let mut count = 0;
            let limit: i32 = 2;

            let mut samples = Vec::new();

            while let Ok(sample) = engine.samples_rx.recv() {
                assert!(sample.is_some(), "Sample is None");
                let sample = sample.unwrap();
                assert!(!sample.image.data.is_empty(), "Image data is empty");
                assert!(sample.image.original_height > 0, "Original height is 0");
                assert!(sample.image.original_width > 0, "Original width is 0");
                assert!(sample.image.height > 0, "Height is 0");
                assert!(sample.image.width > 0, "Width is 0");
                assert!(sample.image.channels > 0, "Channels is 0");
                assert!(sample.image.bit_depth > 0, "Bit depth is 0");
                count += 1;
                samples.push(sample);
                if count >= limit {
                    break;
                }
            }
            info!("Received {} items", count);
            assert!(count >= limit, "Not enough items found in the bucket");
            client.stop();

            samples
        }

        let random_sampling = [false, true];
        for s in random_sampling {
            let samples_1 = get_samples(s, 10);
            let samples_2 = get_samples(s, 10);
            assert_eq!(samples_1.len(), samples_2.len(), "Samples length mismatch");

            if s {
                let sample_ids_1 = samples_1.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
                let sample_ids_2 = samples_2.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
                assert_ne!(
                    sample_ids_1, sample_ids_2,
                    "Samples should not be in the same order"
                );
            }
        }
    }

    #[test]
    fn test_webdataset_ranks() {
        fn get_samples(rank: usize, world_size: usize, n_samples: usize) -> Vec<Sample> {
            let client_config = json!({
                "source_config": {
                    "url": "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/",
                    "random_sampling": false,
                    "max_concurrency": 2,
                    "rank": rank,
                    "world_size": world_size,
                },
                "limit": n_samples,

                "num_threads": 1,
                "max_connections": 1,
                "samples_buffer_size": 1
            });

            let mut client = DatagoClient::new(client_config.to_string());
            let engine = orchestrate(&client);
            let mut count = 0;

            let mut samples = Vec::new();

            while let Ok(sample) = engine.samples_rx.recv() {
                assert!(sample.is_some(), "Sample is None");

                count += 1;
                samples.push(sample.unwrap());
                if count >= n_samples {
                    break;
                }
            }
            info!("Received {} items", count);
            client.stop();

            samples
        }

        let samples_1 = get_samples(0, 2, 20);
        let samples_2 = get_samples(1, 2, 20);
        assert_eq!(samples_1.len(), samples_2.len(), "Samples length mismatch");

        // Check that the samples are all different, as per a set
        let sample_ids_1 = samples_1.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
        let sample_ids_2 = samples_2.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
        let mut all_samples = sample_ids_1.clone();
        all_samples.extend(sample_ids_2.clone());
        all_samples.sort();
        all_samples.dedup();
        assert_eq!(
            all_samples.len(),
            sample_ids_1.len() + sample_ids_2.len(),
            "Samples should all be different"
        );
    }
}
