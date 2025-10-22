use crate::client::DatagoClient;
use crate::worker_http;
use log::{debug, error, info, warn};
use reqwest::header::HeaderMap;
use reqwest::header::HeaderValue;
use reqwest::header::AUTHORIZATION;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::structs::{new_shared_client, DatagoEngine, SharedClient};
use kanal::bounded;
use kanal::Sender;
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDBConfig {
    pub sources: String,
    pub page_size: usize,

    #[serde(default)]
    pub sources_ne: String,

    #[serde(default)]
    pub require_images: bool,

    #[serde(default)]
    pub require_embeddings: bool,

    #[serde(default)]
    pub tags: String,

    #[serde(default)]
    pub tags_all: String,

    #[serde(default)]
    pub tags_ne: String,

    #[serde(default)]
    pub tags_ne_all: String,

    #[serde(default)]
    pub tags_empty: String,

    #[serde(default)]
    pub has_attributes: String,

    #[serde(default)]
    pub lacks_attributes: String,

    #[serde(default)]
    pub has_masks: String,

    #[serde(default)]
    pub lacks_masks: String,

    #[serde(default)]
    pub has_latents: String,

    #[serde(default)]
    pub lacks_latents: String,

    #[serde(default)]
    pub min_short_edge: i32,

    #[serde(default)]
    pub max_short_edge: i32,

    #[serde(default)]
    pub min_pixel_count: i32,

    #[serde(default)]
    pub max_pixel_count: i32,

    #[serde(default)]
    pub duplicate_state: i32,

    #[serde(default)]
    pub attributes: String,

    #[serde(default)]
    pub random_sampling: bool,

    #[serde(default)]
    pub rank: usize,

    #[serde(default)]
    pub world_size: usize,
}

// TODO: Derive from the above
#[derive(Debug, Serialize, Deserialize)]
struct DbRequest {
    pub fields: String,
    pub sources: String,
    pub sources_ne: String,
    pub page_size: String,

    pub tags: String,
    pub tags_all: String,
    pub tags_ne: String,
    pub tags_ne_all: String,
    pub tags_empty: String,

    pub has_attributes: String,
    pub lacks_attributes: String,

    pub has_masks: String,
    pub lacks_masks: String,

    pub has_latents: String,
    pub lacks_latents: String,
    pub return_latents: String,

    pub min_short_edge: String,
    pub max_short_edge: String,

    pub min_pixel_count: String,
    pub max_pixel_count: String,

    pub duplicate_state: String,
    pub attributes: String,
    pub random_sampling: bool,

    pub partitions_count: String,
    pub partition: String,
}

// implement a helper to get the http request which corresponds to the db request structure above
impl DbRequest {
    async fn get_http_request(&self, api_url: &str, api_key: &str) -> reqwest::Request {
        let mut url = if self.random_sampling {
            Url::parse(&format!("{api_url}images/random/"))
        } else {
            Url::parse(&format!("{api_url}images/"))
        }
        .unwrap(); // Cannot survive without the URL, that's a panic

        // Edit the URL with the query parameters
        {
            let mut query_pairs = url.query_pairs_mut();
            let query_pairs = &mut query_pairs;

            let mut maybe_add_field = move |field: &str, value: &str| {
                if !value.is_empty() {
                    query_pairs.append_pair(field, value);
                }
            };

            let return_latents = if !self.has_latents.is_empty() {
                format!("{},{}", self.has_latents, self.has_masks)
            } else {
                self.has_masks.clone()
            };

            maybe_add_field("fields", &self.fields);
            maybe_add_field("sources", &self.sources);
            maybe_add_field("sources__ne", &self.sources_ne);
            maybe_add_field("page_size", &self.page_size);

            maybe_add_field("tags", &self.tags);
            maybe_add_field("tags__all", &self.tags_all);
            maybe_add_field("tags__ne", &self.tags_ne);
            maybe_add_field("tags__ne_all", &self.tags_ne_all);
            maybe_add_field("tags__empty", &self.tags_empty);
            maybe_add_field("has_attributes", &self.has_attributes);
            maybe_add_field("lacks_attributes", &self.lacks_attributes);
            maybe_add_field("has_masks", &self.has_masks);
            maybe_add_field("lacks_masks", &self.lacks_masks);
            maybe_add_field("has_latents", &self.has_latents);
            maybe_add_field("lacks_latents", &self.lacks_latents);
            maybe_add_field("return_latents", &return_latents);
            maybe_add_field("short_edge__gte", &self.min_short_edge);
            maybe_add_field("short_edge__lte", &self.max_short_edge);
            maybe_add_field("pixel_count__gte", &self.min_pixel_count);
            maybe_add_field("pixel_count__lte", &self.max_pixel_count);
            maybe_add_field("duplicate_state", &self.duplicate_state);
            maybe_add_field("attributes", &self.attributes);
            maybe_add_field("partitions_count", &self.partitions_count);
            maybe_add_field("partition", &self.partition);
        }

        let mut req = reqwest::Request::new(reqwest::Method::GET, url);
        req.headers_mut().append(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Token {api_key}"))
                .expect("Couldn't parse the provided API key"),
        );

        debug!("Request URL: {:?}\n", req.url().as_str());

        req
    }
}

fn build_request(source_config: SourceDBConfig) -> DbRequest {
    // Build the request to the DB, given the source configuration
    // There are a lot of straight copies, but also some internal logic
    let mut fields = "id,source,attributes,height,width,tags".to_string();

    if source_config.require_images {
        fields.push_str(",image_direct_url");
    }

    if !source_config.has_latents.is_empty() || !source_config.has_masks.is_empty() {
        fields.push_str(",latents");
        debug!(
            "Including some latents: {} {}",
            source_config.has_latents, source_config.has_masks
        );
    }

    if !source_config.tags.is_empty() {
        fields.push_str(",tags");
        debug!(
            "Including some tags, must have any of: {}",
            source_config.tags
        );
    }

    if !source_config.tags_all.is_empty() {
        fields.push_str(",tags");
        debug!(
            "Including tags, must have all of: {}",
            source_config.tags_all
        );
    }

    if !source_config.tags_ne.is_empty() {
        fields.push_str(",tags");
        debug!(
            "Including tags, must not have any of: {}",
            source_config.tags_ne
        );
    }

    if !source_config.tags_empty.is_empty() {
        fields.push_str(",tags");
        debug!(
            "Using filter: Tags must{} be empty",
            if source_config.tags_empty == "true" {
                " not"
            } else {
                ""
            }
        );
        if !source_config.tags_all.is_empty()
            || !source_config.tags.is_empty()
            || !source_config.tags_ne.is_empty()
            || !source_config.tags_ne_all.is_empty()
        {
            warn!("You've set `tags_empty` in addition to `tags`, `tags_all`, `tags_ne` or `tags_ne_all`. The combination might be incompatible or redundant.");
        }
    }

    if !source_config.tags_ne_all.is_empty() {
        fields.push_str(",tags");
        debug!(
            "Including tags, must not have all of: {}",
            source_config.tags_ne_all
        );
    }

    if source_config.require_embeddings {
        fields.push_str(",coca_embedding");
        debug!("Including embeddings");
    }

    if source_config.duplicate_state >= 0 {
        fields.push_str(",duplicate_state");
    }

    assert!(
        (source_config.rank == 0 && source_config.world_size == 0)
            || (source_config.rank < source_config.world_size),
        "Rank cannot be greater than or equal to world size"
    );

    debug!("Fields: {fields}");
    debug!(
        "Rank: {}, World size: {}",
        source_config.rank, source_config.world_size
    );
    let mut return_latents = source_config.has_latents.clone();
    if !source_config.has_masks.is_empty() {
        return_latents.push_str(&format!(",{}", source_config.has_masks));
    }

    let maybe_add_int = |value: i32| {
        if value > 0 {
            value.to_string()
        } else {
            "".to_string()
        }
    };

    DbRequest {
        fields,
        sources: source_config.sources,
        sources_ne: source_config.sources_ne,
        page_size: source_config.page_size.to_string(),
        tags: source_config.tags,
        tags_all: source_config.tags_all,
        tags_ne: source_config.tags_ne,
        tags_ne_all: source_config.tags_ne_all,
        tags_empty: source_config.tags_empty,
        has_attributes: source_config.has_attributes,
        lacks_attributes: source_config.lacks_attributes,
        has_masks: source_config.has_masks,
        lacks_masks: source_config.lacks_masks,
        has_latents: source_config.has_latents,
        lacks_latents: source_config.lacks_latents,
        return_latents,
        min_short_edge: maybe_add_int(source_config.min_short_edge),
        max_short_edge: maybe_add_int(source_config.max_short_edge),
        min_pixel_count: maybe_add_int(source_config.min_pixel_count),
        max_pixel_count: maybe_add_int(source_config.max_pixel_count),
        duplicate_state: maybe_add_int(source_config.duplicate_state),
        attributes: source_config.attributes,
        random_sampling: source_config.random_sampling,
        partition: if source_config.world_size > 1 {
            format!("{}", source_config.rank)
        } else {
            "".to_string()
        },
        partitions_count: if source_config.world_size > 1 {
            format!("{}", source_config.world_size)
        } else {
            "".to_string()
        },
    }
}

async fn get_response(
    shared_client: Arc<SharedClient>,
    request: &reqwest::Request,
) -> Result<serde_json::Value, reqwest_middleware::Error> {
    let _permit = shared_client.semaphore.acquire();

    match shared_client
        .client
        .execute(request.try_clone().unwrap())
        .await
    {
        Ok(response) => match response.text().await {
            Ok(response_text) => {
                Ok(serde_json::from_str(&response_text).unwrap_or(serde_json::Value::Null))
            }
            Err(e) => Err(reqwest_middleware::Error::from(e)),
        },
        Err(e) => Err(e),
    }
}

async fn async_pull_and_dispatch_pages(
    shared_client: &Arc<SharedClient>,
    samples_metadata_tx: kanal::Sender<serde_json::Value>,
    source_config: SourceDBConfig,
    limit: usize,
) {
    let api_url = std::env::var("DATAROOM_API_URL").expect("DATAROOM_API_URL not set");
    let api_key = std::env::var("DATAROOM_API_KEY").expect("DATAROOM_API_KEY not set");
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Token  {api_key}")).unwrap(),
    );

    let db_request = build_request(source_config.clone());

    // Send the request and dispatch the response to the channel
    let mut response_json: serde_json::Value;
    let mut next_url = &serde_json::Value::Null;

    let initial_request = db_request.get_http_request(&api_url, &api_key).await;

    match get_response(shared_client.clone(), &initial_request).await {
        Ok(tentative_json) => {
            response_json = tentative_json.clone();
            if let Some(next) = response_json.get("next") {
                next_url = next;
            } else {
                debug!("No next URL in the response {response_json:?}");
            }
        }
        Err(e) => {
            error!(
                "Failed fetching the first page: {}\nURL: {}",
                e,
                initial_request.url()
            );
            return;
        }
    }

    // Walk the pages and send them to the channel
    let mut count = 0;
    let max_submitted_samples = (1.1 * (limit as f64)).ceil() as usize;
    'outer: while count < max_submitted_samples {
        match response_json.get("results") {
            Some(results) => {
                // Go over the samples from the current page
                for sample in results.as_array().unwrap() {
                    let sample_json = serde_json::from_value(sample.clone()).unwrap();

                    // Push the sample to the channel
                    if samples_metadata_tx.send(sample_json).is_err() {
                        break 'outer;
                    }

                    count += 1;

                    if count >= max_submitted_samples {
                        // NOTE: This doesn´t count the samples which have actually been processed
                        debug!(
                            "dispatch_pages: reached the limit of samples requested. Shutting down"
                        );
                        break 'outer;
                    }
                }
            }
            None => {
                debug!("No results in the response: {response_json:?}");
            }
        }

        // Ask for the next page
        if next_url == &serde_json::Value::Null {
            debug!("No more pages, exiting");
            break;
        }
        let mut new_request = reqwest::Request::new(
            reqwest::Method::GET,
            reqwest::Url::parse(next_url.as_str().unwrap()).unwrap(),
        );
        new_request.headers_mut().extend(headers.clone());

        next_url = &serde_json::Value::Null;
        match get_response(shared_client.clone(), &new_request).await {
            Ok(tentative_json) => {
                response_json = tentative_json.clone();
                next_url = response_json
                    .get("next")
                    .unwrap_or(&serde_json::Value::Null);
            }
            Err(e) => {
                error!(
                    "Failed fetching a new page: {}\nURL: {}",
                    e,
                    new_request.url()
                );
            }
        }
    }

    // Either we don't have any more samples or we have reached the limit
    debug!(
        "pull_and_dispatch_pages: total samples requested: {limit}. page samples served {count}"
    );

    // Send an empty value to signal the end of the stream
    match samples_metadata_tx.send(serde_json::Value::Null) {
        Ok(_) => {}
        Err(_) => {
            debug!("pull_and_dispatch_pages: stream already closed, all good");
        }
    }
}

pub fn pull_and_dispatch_pages(
    shared_client: &Arc<SharedClient>,
    samples_metadata_tx: Sender<serde_json::Value>,
    source_config: SourceDBConfig,
    limit: usize,
) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            async_pull_and_dispatch_pages(
                shared_client,
                samples_metadata_tx.clone(),
                source_config,
                limit,
            )
            .await;
        });

    // Send an empty value to signal the end of the stream
    if samples_metadata_tx.send(serde_json::Value::Null).is_err() {
        debug!("pull_and_dispatch_pages: stream already closed, all good");
    };
}

pub fn orchestrate(client: &DatagoClient) -> DatagoEngine {
    // Start pulling the samples, which spread across two steps. The samples will land in the last kanal,
    // all the threads pausing when the required buffer depth is reached.

    // A first thread will ping the DB and get pages back, meaning documents with a lot of per-sample meta data.
    // This meta data is then dispatched to a worker pool, which will download the payloads, deserialize them,
    // do the required pre-processing then commit to the ready queue.

    let http_client = Arc::new(new_shared_client(client.max_connections));

    // Allocate all the message passing pipes
    let (samples_metadata_tx, samples_metadata_rx) = bounded(client.samples_buffer * 2);
    let (samples_tx, samples_rx) = bounded(client.samples_buffer);

    // Convert the source_config to a SourceDBConfig
    let source_db_config: SourceDBConfig =
        serde_json::from_value(client.source_config.clone()).unwrap();

    info!("Using DB as source");
    let limit = client.limit;
    let source_config = source_db_config.clone();
    let shared_client = http_client.clone();

    // Spawn a thread which will ping the DB
    let feeder = Some(thread::spawn(move || {
        pull_and_dispatch_pages(&shared_client, samples_metadata_tx, source_config, limit);
    }));

    // Spawn a thread which will handle the async workers
    let image_transform = client.image_transform.clone();
    let encoding = crate::image_processing::ImageEncoding {
        encode_images: client.encode_images,
        img_to_rgb8: client.img_to_rgb8,
        encode_format: client.encode_format,
        jpeg_quality: client.jpeg_quality,
    };
    let limit = client.limit;
    let samples_tx_worker = samples_tx.clone();
    let samples_metadata_rx_worker = samples_metadata_rx.clone();

    let worker = Some(thread::spawn(move || {
        worker_http::pull_samples(
            &http_client,
            samples_metadata_rx_worker,
            samples_tx_worker,
            image_transform,
            encoding,
            limit,
        );
    }));

    DatagoEngine {
        samples_rx,
        feeder,
        worker,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_request() {
        let config = SourceDBConfig {
            sources: "source1,source2".to_string(),
            page_size: 10,
            sources_ne: "source3".to_string(),
            require_images: true,
            require_embeddings: true,
            tags: "tag1,tag2".to_string(),
            tags_all: "tag3,tag4".to_string(),
            tags_ne: "tag5".to_string(),
            tags_ne_all: "tag6".to_string(),
            tags_empty: "false".to_string(),
            has_attributes: "attr1".to_string(),
            lacks_attributes: "attr2".to_string(),
            has_masks: "mask1".to_string(),
            lacks_masks: "mask2".to_string(),
            has_latents: "latent1".to_string(),
            lacks_latents: "latent2".to_string(),
            min_short_edge: 100,
            max_short_edge: 1000,
            min_pixel_count: 10000,
            max_pixel_count: 1000000,
            duplicate_state: 1,
            attributes: "attr=val".to_string(),
            random_sampling: false,
            rank: 1,
            world_size: 2,
        };

        let request = build_request(config);

        assert!(request.fields.contains("id,source,attributes"));
        assert!(request.fields.contains("image_direct_url"));
        assert!(request.fields.contains("coca_embedding"));
        assert!(request.fields.contains("latents"));
        assert!(request.fields.contains("tags"));
        assert_eq!(request.sources, "source1,source2");
        assert_eq!(request.sources_ne, "source3");
        assert_eq!(request.page_size, "10");
        assert_eq!(request.tags, "tag1,tag2");
        assert_eq!(request.tags_all, "tag3,tag4");
        assert_eq!(request.tags_ne, "tag5");
        assert_eq!(request.tags_ne_all, "tag6");
        assert_eq!(request.tags_empty, "false");
        assert_eq!(request.has_attributes, "attr1");
        assert_eq!(request.lacks_attributes, "attr2");
        assert_eq!(request.has_masks, "mask1");
        assert_eq!(request.lacks_masks, "mask2");
        assert_eq!(request.has_latents, "latent1");
        assert_eq!(request.lacks_latents, "latent2");
        assert_eq!(request.min_short_edge, "100");
        assert_eq!(request.max_short_edge, "1000");
        assert_eq!(request.min_pixel_count, "10000");
        assert_eq!(request.max_pixel_count, "1000000");
        assert_eq!(request.duplicate_state, "1");
        assert_eq!(request.attributes, "attr=val");
        assert!(!request.random_sampling);
        assert_eq!(request.partition, "1");
        assert_eq!(request.partitions_count, "2");
    }

    #[test]
    fn test_build_request_minimal() {
        let config = SourceDBConfig {
            sources: "source1".to_string(),
            page_size: 20,
            sources_ne: "".to_string(),
            require_images: false,
            require_embeddings: false,
            tags: "".to_string(),
            tags_all: "".to_string(),
            tags_ne: "".to_string(),
            tags_ne_all: "".to_string(),
            tags_empty: "".to_string(),
            has_attributes: "".to_string(),
            lacks_attributes: "".to_string(),
            has_masks: "".to_string(),
            lacks_masks: "".to_string(),
            has_latents: "".to_string(),
            lacks_latents: "".to_string(),
            min_short_edge: 0,
            max_short_edge: 0,
            min_pixel_count: 0,
            max_pixel_count: 0,
            duplicate_state: -1,
            attributes: "".to_string(),
            random_sampling: true,
            rank: 0,
            world_size: 1,
        };

        let request = build_request(config);

        assert_eq!(request.fields, "id,source,attributes,height,width,tags");
        assert_eq!(request.sources, "source1");
        assert_eq!(request.sources_ne, "");
        assert_eq!(request.page_size, "20");
        assert_eq!(request.min_short_edge, "");
        assert_eq!(request.max_short_edge, "");
        assert_eq!(request.duplicate_state, "");
        assert!(request.random_sampling);
        assert_eq!(request.partition, "");
        assert_eq!(request.partitions_count, "");
    }

    #[tokio::test]
    async fn test_get_http_request() {
        let db_request = DbRequest {
            fields: "id,source".to_string(),
            sources: "source1".to_string(),
            sources_ne: "".to_string(),
            page_size: "10".to_string(),
            tags: "tag1".to_string(),
            tags_all: "".to_string(),
            tags_ne: "".to_string(),
            tags_ne_all: "".to_string(),
            tags_empty: "".to_string(),
            has_attributes: "".to_string(),
            lacks_attributes: "".to_string(),
            has_masks: "".to_string(),
            lacks_masks: "".to_string(),
            has_latents: "".to_string(),
            lacks_latents: "".to_string(),
            return_latents: "".to_string(),
            min_short_edge: "".to_string(),
            max_short_edge: "".to_string(),
            min_pixel_count: "".to_string(),
            max_pixel_count: "".to_string(),
            duplicate_state: "".to_string(),
            attributes: "".to_string(),
            random_sampling: false,
            partitions_count: "".to_string(),
            partition: "".to_string(),
        };

        let request = db_request
            .get_http_request("https://api.example.com/", "test_key")
            .await;
        let url = request.url().to_string();

        assert!(url.starts_with("https://api.example.com/images/"));
        assert!(url.contains("fields=id%2Csource"));
        assert!(url.contains("sources=source1"));
        assert!(url.contains("tags=tag1"));
        assert!(url.contains("page_size=10"));

        let auth_header = request.headers().get(AUTHORIZATION).unwrap();
        assert_eq!(auth_header, "Token test_key");
    }

    #[tokio::test]
    async fn test_get_http_request_random_sampling() {
        let db_request = DbRequest {
            fields: "id,source".to_string(),
            sources: "source1".to_string(),
            sources_ne: "".to_string(),
            page_size: "10".to_string(),
            tags: "".to_string(),
            tags_all: "".to_string(),
            tags_ne: "".to_string(),
            tags_ne_all: "".to_string(),
            tags_empty: "".to_string(),
            has_attributes: "".to_string(),
            lacks_attributes: "".to_string(),
            has_masks: "".to_string(),
            lacks_masks: "".to_string(),
            has_latents: "".to_string(),
            lacks_latents: "".to_string(),
            return_latents: "".to_string(),
            min_short_edge: "".to_string(),
            max_short_edge: "".to_string(),
            min_pixel_count: "".to_string(),
            max_pixel_count: "".to_string(),
            duplicate_state: "".to_string(),
            attributes: "".to_string(),
            random_sampling: true,
            partitions_count: "".to_string(),
            partition: "".to_string(),
        };

        let request = db_request
            .get_http_request("https://api.example.com/", "test_key")
            .await;
        let url = request.url().to_string();

        assert!(url.starts_with("https://api.example.com/images/random/"));
    }

    #[test]
    #[should_panic(expected = "Rank cannot be greater than or equal to world size")]
    fn test_broken_ranks() {
        let config = SourceDBConfig {
            sources: "source1".to_string(),
            page_size: 20,
            sources_ne: "".to_string(),
            require_images: false,
            require_embeddings: false,
            tags: "".to_string(),
            tags_all: "".to_string(),
            tags_ne: "".to_string(),
            tags_ne_all: "".to_string(),
            tags_empty: "".to_string(),
            has_attributes: "".to_string(),
            lacks_attributes: "".to_string(),
            has_masks: "".to_string(),
            lacks_masks: "".to_string(),
            has_latents: "".to_string(),
            lacks_latents: "".to_string(),
            min_short_edge: 0,
            max_short_edge: 0,
            min_pixel_count: 0,
            max_pixel_count: 0,
            duplicate_state: -1,
            attributes: "".to_string(),
            random_sampling: true,
            rank: 2, // This is the problematic part
            world_size: 1,
        };

        let _request = build_request(config);
    }
}
