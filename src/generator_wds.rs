use crate::client::DatagoClient;
use crate::structs::{new_shared_client, DatagoEngine, SharedClient, TarballContent, WDSContent};
use crate::worker_http::bytes_from_url;
use crate::worker_wds;

use kanal::bounded;
use log::{debug, warn};
use rand::seq::SliceRandom;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::error::Error;
use std::io::Cursor;
use std::io::Read;
use std::sync::Arc;
use std::thread;
use tar::Archive;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceWebDatasetConfig {
    pub url: String,

    #[serde(default)]
    pub shuffle: bool,
    pub max_tasks_in_flight: usize,
}

fn urls_from_pattern(url: &str) -> Vec<String> {
    // Extract the pattern within curly braces
    let pattern = url.split('{').nth(1).unwrap_or("");
    let pattern = pattern.split('}').next().unwrap_or("");

    // Split the pattern into parts
    let parts: Vec<&str> = pattern.split("..").collect();
    let start = parts[0].parse::<i32>().unwrap_or(0);
    let end = parts[1].parse::<i32>().unwrap_or(0);

    // Generate all the possible URLs
    (start..=end)
        .map(|i| url.replace(&format!("{{{}}}", pattern), &format!("{:06}", i)))
        .collect()
}

async fn untar_bytes_in_memory(tar_bytes: &Vec<u8>) -> Result<TarballContent, Box<dyn Error>> {
    let cursor = Cursor::new(tar_bytes);
    let mut archive = Archive::new(cursor);
    let mut files = TarballContent::new();

    for entry_result in archive.entries()? {
        let mut entry = entry_result?;
        let path = entry.path()?;
        let filename = path.display().to_string();

        let mut buffer = Vec::new();
        entry.read_to_end(&mut buffer)?;

        files.push(WDSContent { filename, buffer });
    }
    Ok(files)
}

fn group_samples(samples: TarballContent) -> Result<Vec<TarballContent>, Box<dyn Error>> {
    let mut grouped_samples: HashMap<i32, TarballContent> = HashMap::new();

    // Go through the files and group them by sample. The last number group in the filename is the sample number
    // Note that for performance reasons we'll consume "samples" and move its content into the result
    for content in samples.into_iter() {
        let filename = &content.filename;

        if let Some(sample_number) = filename
            .split(&['-', '_', '.'][..]) // The separator is either '-' or '_'
            .nth(
                // Get the item before the last
                filename
                    .split(&['-', '_', '.'][..])
                    .count()
                    .checked_sub(2)
                    .unwrap(),
            )
            .and_then(|s| s.parse::<i32>().ok())
        {
            grouped_samples
                .entry(sample_number)
                .or_default()
                .push(content);
        }
    }
    Ok(grouped_samples.into_values().collect())
}

async fn pull_tarball(
    shared_client: Arc<SharedClient>,
    url: serde_json::Value,
    samples_metadata_tx: kanal::Sender<TarballContent>,
    shuffle: bool,
) -> Result<(), ()> {
    debug!(
        "dispatch_shards: downloading a new tarball {:?}",
        url.as_str()
    );
    let mut retries = 5;
    let _permit = shared_client.semaphore.acquire();

    while retries > 0 {
        match bytes_from_url(&shared_client, url.as_str().unwrap()).await {
            Some(tarball) => {
                debug!("dispatch_shards: tarball downloaded");
                if let Ok(contents) = untar_bytes_in_memory(&tarball).await {
                    let mut samples = group_samples(contents).unwrap();

                    // Shuffle the samples if needed. This is only within the tarball
                    // but we shuffle across tarballs on top
                    if shuffle {
                        samples.shuffle(&mut rand::rng());
                    }

                    for sample in samples.into_iter() {
                        if samples_metadata_tx.send(sample).is_err() {
                            debug!("dispatch_shards: stream already closed, all good");
                            return Err(());
                        }
                    }
                    return Ok(());
                } else {
                    warn!("dispatch_shards: failed to unpack tarball");
                    retries -= 1;
                    if retries == 0 {
                        println!("dispatch_shards: failed to unpack tarball after 5 attempts");
                        return Err(());
                    }
                }
            }
            None => {
                warn!("dispatch_shards: failed to download tarball {:?}", url);
                retries -= 1;
                if retries == 0 {
                    warn!("dispatch_shards: failed to download tarball after 5 attempts");
                    return Err(());
                }
            }
        }
    }

    Err(())
}

async fn get_url_list(
    shared_client: &Arc<SharedClient>,
    config: &SourceWebDatasetConfig,
) -> Vec<serde_json::Value> {
    let _permit = shared_client.semaphore.acquire();

    // Either ping the url to get the pages, or use the {...} syntax
    if config.url.contains("{") {
        // URL should look like this:
        // https://storage.googleapis.com/webdataset/testdata/publaynet-train-{000000..000009}.tar
        // We need to parse the URL and generate all the possible URLs
        // for instance https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar

        // Extract the pattern within curly braces
        let urls = urls_from_pattern(&config.url);

        urls.iter()
            .map(|url| serde_json::Value::String(url.clone()))
            .collect()
    } else {
        assert!(config.url.contains("https://storage.googleapis.com/"));

        // Given the url, list all the available webdataset files
        let request = reqwest::Request::new(reqwest::Method::GET, Url::parse(&config.url).unwrap());
        let response = shared_client.client.execute(request).await.unwrap();
        let response_text = response.text().await.unwrap();
        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).unwrap_or(serde_json::Value::Null);

        // Parse all the "items" in the response
        if let Some(items) = response_json.get("items") {
            items
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| item.get("mediaLink").cloned())
                .collect()
        } else {
            // If the response is empty, return an empty vector
            warn!("dispatch_shards: no items found in the response");
            vec![]
        }
    }
}

async fn list_shards(
    shared_client: Arc<SharedClient>,
    samples_metadata_tx: kanal::Sender<TarballContent>,
    config: &SourceWebDatasetConfig,
) -> Result<serde_json::Value, tokio::task::JoinError> {
    let mut task_list = get_url_list(&shared_client, config).await;

    // Shuffle the items if needed
    if config.shuffle {
        task_list.shuffle(&mut rand::rng());
    }

    // Now submit all the tasks, making sure that too many of them are not in flight
    let mut tasks = VecDeque::new();
    let response_json = serde_json::Value::Null;
    let mut count = 0;

    for url in task_list {
        tasks.push_back(
            tokio::spawn(pull_tarball(
                shared_client.clone(),
                url,
                samples_metadata_tx.clone(),
                config.shuffle,
            ))
            .await,
        );

        // Some bookkeeping, to limit the number of tasks in flight
        // we'll wait for the first one to finish before adding a new one
        if tasks.len() >= config.max_tasks_in_flight {
            // Catch an early stop in this case, can be that the stream is closed
            // and we don't want to keep going forever
            let res = tasks.pop_front().unwrap();
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
        let _ = tasks.pop_front().unwrap();
    }

    if count == 0 {
        warn!("No items found in the response");
    }
    debug!("Served {} items from the bucket", count);

    if samples_metadata_tx.send(vec![]).is_err() {
        debug!("list_shards: stream already closed, wrapping up");
    }

    Ok(response_json)
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
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let _ = list_shards(shared_client.clone(), samples_metadata_tx, &source_config).await;
        });
}

// ---- Global orchestration ---------
pub fn orchestrate(client: &DatagoClient) -> DatagoEngine {
    // Allocate all the message passing pipes
    let (samples_metadata_tx, samples_metadata_rx) = bounded::<TarballContent>(32);
    let (samples_tx, samples_rx) = bounded(client.samples_buffer);

    // Convert the source_config to a SourceWebDatasetConfig
    let source_config: SourceWebDatasetConfig =
        serde_json::from_value(client.source_config.clone()).unwrap();

    println!("Using webdataset as source");

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
        let shuffle = [false, true];

        for s in shuffle {
            let config = SourceWebDatasetConfig {
                url:
                    "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/"
                        .into(),
                shuffle: s,
                max_tasks_in_flight: 2,
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
                    println!("No more items to receive");
                    break;
                }
            }

            println!("Received {} items", count);
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
        fn get_samples(do_shuffle: bool, n_samples: usize) -> Vec<Sample> {
            let client_config = json!({
                "source_config": {
                    "url": "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/",
                    "shuffle": do_shuffle,
                    "max_tasks_in_flight": 2
                },
                "limit": n_samples,
                "rank": 0,
                "world_size": 1,
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
            println!("Received {} items", count);
            assert!(count >= limit, "Not enough items found in the bucket");
            client.stop();

            samples
        }

        let shuffle = [false, true];
        for s in shuffle {
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
}
