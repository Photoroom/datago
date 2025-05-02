use crate::client::DatagoClient;
use crate::structs::{new_shared_client, DatagoEngine, SharedClient, TarballContent, WDSContent};
use crate::worker_http::bytes_from_url;
use crate::worker_wds;

use kanal::bounded;
use log::{debug, info, warn};
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

async fn list_shards(
    shared_client: Arc<SharedClient>,
    pages_tx: kanal::Sender<serde_json::Value>,
    config: &SourceWebDatasetConfig,
) -> Result<serde_json::Value, reqwest_middleware::Error> {
    let _permit = shared_client.semaphore.acquire();

    println!("listing shards in bucket: {}", config.url);

    // Either ping the url to get the pages, or use the {...} syntax
    if config.url.contains("{") {
        // URL should look like this:
        // https://storage.googleapis.com/webdataset/testdata/publaynet-train-{000000..000009}.tar
        // We need to parse the URL and generate all the possible URLs
        // for instance https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar

        // Extract the pattern within curly braces
        let mut urls = urls_from_pattern(&config.url);

        if config.shuffle {
            // Shuffle the URLs first, we'll shuffle the result buffer on top
            urls.shuffle(&mut rand::rng()); // Seed is random already
        }

        for url in urls {
            // Send the URL to the channel
            if pages_tx.send(serde_json::Value::String(url)).is_err() {
                debug!("Failed to send item");
                break;
            }
        }
        // to be continued
        Ok(serde_json::Value::Null)
    } else {
        assert!(config.url.contains("https://storage.googleapis.com/"));

        // Given the url, list all the available webdataset files
        let request = reqwest::Request::new(reqwest::Method::GET, Url::parse(&config.url).unwrap());
        let response = shared_client.client.execute(request).await?;
        let response_text = response.text().await?;
        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).unwrap_or(serde_json::Value::Null);

        // Parse all the "items" in the response
        let mut count = 0;
        if let Some(items) = response_json.get("items") {
            // Extract all media links from items
            let mut media_links: Vec<serde_json::Value> = items
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| item.get("mediaLink").cloned())
                .collect();

            // Shuffle the items if needed
            if config.shuffle {
                media_links.shuffle(&mut rand::rng());
            }

            for link in media_links {
                if pages_tx.send(link.clone()).is_err() {
                    debug!("Failed to send item");
                    break;
                }
                debug!("Pushed new item: {}", link);
                count += 1;
            }
        }

        if count == 0 {
            warn!("No items found in the response");
        }

        debug!("Found {} items in the bucket", count);
        if pages_tx.send(serde_json::Value::Null).is_err() {
            debug!("ping_pages: stream already closed, wrapping up");
        };
        Ok(response_json)
    }
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
    response_json: serde_json::Value,
    samples_meta_tx: kanal::Sender<TarballContent>,
    shuffle: bool,
) {
    debug!(
        "dispatch_shards: downloading a new tarball {:?}",
        response_json.as_str().unwrap()
    );
    let mut retries = 5;

    while retries > 0 {
        match bytes_from_url(&shared_client, response_json.as_str().unwrap()).await {
            Some(tarball) => {
                debug!("dispatch_shards: tarball downloaded");
                if let Ok(contents) = untar_bytes_in_memory(&tarball).await {
                    let mut samples = group_samples(contents).unwrap();

                    // Shuffle the samples if needed
                    if shuffle {
                        samples.shuffle(&mut rand::rng());
                    }

                    for sample in samples.into_iter() {
                        if samples_meta_tx.send(sample).is_err() {
                            debug!("dispatch_shards: stream already closed, all good");
                            return;
                        }
                    }
                } else {
                    warn!("dispatch_shards: failed to unpack tarball");
                    retries -= 1;
                    if retries == 0 {
                        println!("dispatch_shards: failed to unpack tarball after 5 attempts");
                        return;
                    }
                }
            }
            None => {
                warn!(
                    "dispatch_shards: failed to download tarball {:?}",
                    response_json.as_str().unwrap()
                );
                retries -= 1;
                if retries == 0 {
                    warn!("dispatch_shards: failed to download tarball after 5 attempts");
                    return;
                }
            }
        }
    }
}

async fn async_pull_and_dispatch_tarballs(
    shared_client: Arc<SharedClient>,
    pages_rx: kanal::Receiver<serde_json::Value>,
    samples_meta_tx: kanal::Sender<TarballContent>,
    shuffle: bool,
) {
    // While we have something, send the samples to the channel
    let max_tasks = 10; // NOTE: could be exposed as a parameter
    let mut tasks = VecDeque::new();

    loop {
        match pages_rx.recv() {
            Ok(serde_json::Value::Null) => {
                warn!("dispatch_pages: end of stream received, stopping there");
                break;
            }
            Ok(response_json) => {
                // Dispatch the tarball download to a new task
                debug!("dispatch_shards: dispatching a new task");
                tasks.push_back(tokio::spawn(pull_tarball(
                    shared_client.clone(),
                    response_json,
                    samples_meta_tx.clone(),
                    shuffle,
                )));

                // Some bookkeeping, to limit the number of tasks in flight
                if tasks.len() >= max_tasks {
                    // Wait for the oldest task to finish // FIXME: we should wait for _any_ task to be finished
                    if let Some(task) = tasks.pop_front() {
                        if let Err(e) = task.await {
                            warn!("dispatch_shards: task failed - {}", e);
                        }
                    }
                }
            }
            Err(_) => {
                break; // already in the outer loop
            }
        }
    }

    // Consume alle the remaining tasks
    while !tasks.is_empty() {
        if let Some(task) = tasks.pop_front() {
            if let Err(e) = task.await {
                warn!("dispatch_shards: task failed - {}", e);
            }
        }
    }

    // Either we don't have any more samples or we have reached the limit
    info!("dispatch_shards closing");

    // Send an empty value to signal the end of the stream
    if samples_meta_tx.send(vec![]).is_err() {
        debug!("dispatch_shards: stream already closed, all good");
    }
}

fn dispatch_shards(
    shared_client: Arc<SharedClient>,
    pages_rx: kanal::Receiver<serde_json::Value>,
    samples_meta_tx: kanal::Sender<TarballContent>,
    shuffle: bool,
) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(5)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            _ = async_pull_and_dispatch_tarballs(
                shared_client.clone(),
                pages_rx,
                samples_meta_tx,
                shuffle,
            )
            .await;
        });
}

fn query_shards(
    shared_client: Arc<SharedClient>,
    pages_tx: kanal::Sender<serde_json::Value>,
    source_config: SourceWebDatasetConfig,
) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let _ = list_shards(shared_client.clone(), pages_tx, &source_config).await;
        });
}

// ---- Global orchestration ---------
pub fn orchestrate(client: &DatagoClient) -> DatagoEngine {
    let http_client = Arc::new(new_shared_client(client.max_connections));

    // Allocate all the message passing pipes
    let (pages_tx, pages_rx) = bounded(5);
    let (samples_meta_tx, samples_meta_rx) = bounded::<TarballContent>(32);
    let (samples_tx, samples_rx) = bounded(client.samples_buffer);

    // convert the source_config to a SourceWebDatasetConfig
    let source_config: SourceWebDatasetConfig =
        serde_json::from_value(client.source_config.clone()).unwrap();

    println!("Using webdataset as source");

    // List the contents of the bucket
    let shuffle = source_config.shuffle;
    let shared_client: Arc<SharedClient> = http_client.clone();
    let pages_tx_pinger = pages_tx.clone();
    let pinger = Some(thread::spawn(move || {
        query_shards(shared_client, pages_tx_pinger, source_config);
    }));

    // Spawn a thread which will dispatch the pages to the workers
    let shared_client: Arc<SharedClient> = http_client.clone();
    let pages_rx_pinger = pages_rx.clone();
    let feeder = Some(thread::spawn(move || {
        dispatch_shards(shared_client, pages_rx_pinger, samples_meta_tx, shuffle);
    }));

    // Kick the workers which deserialize all the payloads
    let image_transform = client.image_transform.clone();
    let encode_images = client.encode_images;
    let img_to_rgb8 = client.image_to_rgb8;
    let limit = client.limit;
    let samples_tx_worker = samples_tx.clone();
    let worker = Some(thread::spawn(move || {
        worker_wds::deserialize_samples(
            samples_meta_rx,
            samples_tx_worker,
            image_transform,
            encode_images,
            img_to_rgb8,
            limit,
        );
    }));

    DatagoEngine {
        pages_rx,
        samples_tx,
        samples_rx,

        pinger,
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
    fn test_webdataset_query_google() {
        let shuffle = [false, true];

        for s in shuffle {
            fn test(require_shuffle: bool) -> Vec<serde_json::Value> {
                let config = SourceWebDatasetConfig {
                    url:
                        "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/"
                            .into(),
                    shuffle: require_shuffle,
                };

                // Test the bucket query
                let http_client = Arc::new(new_shared_client(2));
                let (pages_tx, pages_rx) = bounded::<serde_json::Value>(2);
                let pinger = thread::spawn(move || query_shards(http_client, pages_tx, config));

                let mut count = 0;
                let max_count: i32 = 5;

                let mut pages = Vec::new();
                while let Ok(page) = pages_rx.recv() {
                    if page.is_null() {
                        break;
                    }
                    pages.push(page);
                    count += 1;

                    if count >= max_count {
                        break;
                    }
                }

                assert!(count == max_count, "No items found in the bucket");
                let _ = pages_rx.close();
                pinger.join().unwrap();
                return pages;
            }

            // First request, check that everything runs fine
            let pages = test(s);
            assert_eq!(pages.len(), 5);
            assert!(
                pages.iter().all(|page| !page.is_null()),
                "All pages should be non-null"
            );

            // Second test, depending on shuffle or not check that the order is respected
            if s {
                // Check that the pages are shuffled
                let shuffled = test(true);
                assert_ne!(pages, shuffled, "Pages should be shuffled");
            } else {
                // Check that the pages are in order
                let repeat = test(false);
                assert_eq!(pages, repeat, "Pages should not be shuffled");
            }
            println!("Pages: {:?}", pages);
        }
    }

    #[test]
    fn test_webdataset_query_expand() {
        let shuffle = [false, true];

        for s in shuffle {
            fn test(require_shuffle: bool) -> Vec<serde_json::Value> {
                let config = SourceWebDatasetConfig {
                    url: "https://storage.googleapis.com/webdataset/testdata/publaynet-train-{000000..000009}.tar/"
                        .into(),
                    shuffle: require_shuffle,
                };

                // Test the bucket query
                let http_client = Arc::new(new_shared_client(2));
                let (pages_tx, pages_rx) = bounded::<serde_json::Value>(2);
                let pinger = thread::spawn(move || query_shards(http_client, pages_tx, config));

                let mut count = 0;
                let max_count: i32 = 5;

                let mut pages = Vec::new();
                while let Ok(page) = pages_rx.recv() {
                    if page.is_null() {
                        break;
                    }
                    pages.push(page);
                    count += 1;

                    if count >= max_count {
                        break;
                    }
                }

                assert!(count == max_count, "No items found in the bucket");
                let _ = pages_rx.close();
                pinger.join().unwrap();
                return pages;
            }

            // First request, check that everything runs fine
            let pages = test(s);
            assert_eq!(pages.len(), 5);
            assert!(
                pages.iter().all(|page| !page.is_null()),
                "All pages should be non-null"
            );

            // Second test, depending on shuffle or not check that the order is respected
            if s {
                // Check that the urls are different when shuffle is enabled
                // Note: There's a small probability this test could fail by chance
                let pages2 = test(s);
                assert_ne!(
                    pages, pages2,
                    "URLs should be in different order when shuffle is enabled"
                );
            } else {
                // Check that the urls are in the same order (for pattern expansion)
                let pages2 = test(s);
                assert_eq!(
                    pages, pages2,
                    "URLs should be in the same order when shuffle is disabled"
                );
            }
            println!("Pages: {:?}", pages);
        }
    }

    #[test]
    fn test_webdataset_dispatch() {
        // TODO: Test the shuffling
        let shuffle = false;
        let config = SourceWebDatasetConfig {
            url: "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/"
                .into(),
            shuffle: shuffle,
        };

        // Query the bucket, pull the workload
        let http_client = Arc::new(new_shared_client(2));
        let (pages_tx, pages_rx) = bounded::<serde_json::Value>(2);
        let (samples_meta_tx, samples_meta_rx) = bounded::<TarballContent>(2);

        let mut count = 0;
        let limit = 2;
        let pinger_client = http_client.clone();
        let pinger = thread::spawn(move || query_shards(pinger_client, pages_tx, config));
        let samples_meta_tx_client = samples_meta_tx.clone();
        let pages_rx_client = pages_rx.clone();
        let dispatcher = thread::spawn(move || {
            dispatch_shards(
                http_client,
                pages_rx_client,
                samples_meta_tx_client,
                shuffle,
            );
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
        let _ = pages_rx.close();
        let _ = samples_meta_rx.close();
        pinger.join().unwrap();
        dispatcher.join().unwrap();

        assert!(count >= limit, "Not enough items found in the bucket");
    }

    #[cfg(test)]
    use serde_json::json;

    #[test]
    fn test_webdataset_orchestrate() {
        let client_config = json!({
            "source_config": {
                "url": "https://storage.googleapis.com/storage/v1/b/webdataset/o?prefix=fake-imagenet/",
                "shuffle": false,
            },
            "limit": 2,
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
            if count >= limit {
                break;
            }
        }
        println!("Received {} items", count);
        assert!(count >= limit, "Not enough items found in the bucket");
        client.stop();
    }
}
