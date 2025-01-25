use reqwest::header::HeaderMap;
use reqwest::header::HeaderValue;
use reqwest::header::AUTHORIZATION;
use reqwest::Url;
use serde::Deserialize;
use serde::Serialize;


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceDBConfig {
    pub sources: String,
    pub sources_ne: String,
    pub require_images: bool,
    pub require_embeddings: bool,
    pub tags: String,
    pub tags_ne: String,
    pub has_attributes: String,
    pub lacks_attributes: String,
    pub has_masks: String,
    pub lacks_masks: String,
    pub has_latents: String,
    pub lacks_latents: String,
    pub min_short_edge: i32,
    pub max_short_edge: i32,
    pub min_pixel_count: i32,
    pub max_pixel_count: i32,
    pub duplicate_state: i32,
    pub random_sampling: bool,
}


// TODO: Derive from the above
#[derive(Debug, Serialize, Deserialize)]
pub struct DbRequest {
    pub fields: String,
    pub sources: String,
    pub sources_ne: String,
    pub page_size: String,

    pub tags: String,
    pub tags_ne: String,

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
    pub random_sampling: bool,

    pub partitions_count: String,
    pub partition: String,
}

// implement a helper to get the http request which correspons to the db request structure above
impl DbRequest {
    pub fn get_http_request(&self, api_url: &str, api_key: &str) -> reqwest::blocking::Request {
        let mut url = if self.random_sampling {
            Url::parse(&format!("{}images/random/", api_url)).unwrap()
        } else {
            Url::parse(&format!("{}images/", api_url)).unwrap()
        };

        let mut query_pairs = url.query_pairs_mut();

        let maybe_add_field =
            |query_pairs: &mut url::form_urlencoded::Serializer<url::UrlQuery>,
             field: &str,
             value: &str| {
                if !value.is_empty() {
                    query_pairs.append_pair(field, value);
                }
            };

        let return_latents = if !self.has_latents.is_empty() {
            format!("{},{}", self.has_latents, self.has_masks)
        } else {
            self.has_masks.clone()
        };

        maybe_add_field(&mut query_pairs, "fields", &self.fields);
        maybe_add_field(&mut query_pairs, "sources", &self.sources);
        maybe_add_field(&mut query_pairs, "sources__ne", &self.sources_ne);
        maybe_add_field(&mut query_pairs, "page_size", &self.page_size);

        maybe_add_field(&mut query_pairs, "tags", &self.tags);
        maybe_add_field(&mut query_pairs, "tags__ne", &self.tags_ne);

        maybe_add_field(&mut query_pairs, "has_attributes", &self.has_attributes);
        maybe_add_field(&mut query_pairs, "lacks_attributes", &self.lacks_attributes);

        maybe_add_field(&mut query_pairs, "has_masks", &self.has_masks);
        maybe_add_field(&mut query_pairs, "lacks_masks", &self.lacks_masks);

        maybe_add_field(&mut query_pairs, "has_latents", &self.has_latents);
        maybe_add_field(&mut query_pairs, "lacks_latents", &self.lacks_latents);
        maybe_add_field(&mut query_pairs, "return_latents", &return_latents);

        maybe_add_field(&mut query_pairs, "short_edge__gte", &self.min_short_edge);
        maybe_add_field(&mut query_pairs, "short_edge__lte", &self.max_short_edge);
        maybe_add_field(&mut query_pairs, "pixel_count__gte", &self.min_pixel_count);
        maybe_add_field(&mut query_pairs, "pixel_count__lte", &self.max_pixel_count);

        maybe_add_field(&mut query_pairs, "duplicate_state", &self.duplicate_state);

        maybe_add_field(&mut query_pairs, "partitions_count", &self.partitions_count);
        maybe_add_field(&mut query_pairs, "partition", &self.partition);

        drop(query_pairs);

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Token  {}", api_key)).unwrap(),
        );

        let mut req = reqwest::blocking::Request::new(reqwest::Method::GET, url);
        req.headers_mut().extend(headers);

        println!("{:?}\n", req);

        req
    }
}

pub fn ping_pages(
    pages_tx: kanal::Sender<serde_json::Value>,
    source_config: SourceDBConfig,
    rank: usize,
    world_size: usize,
    page_size: usize,
    num_samples: usize,
) {
    let retries = 5;

    // Build the request to the DB, given the source configuration
    // There are a lot of straight copies, but also some internal logic
    let mut fields = "id,source,attributes,source".to_string();

    if source_config.require_images {
        fields.push_str(",image_direct_url");
    }

    if source_config.has_latents.len() > 0 || source_config.has_masks.len() > 0 {
		fields.push_str(",latents");
		println!("Including some latents: {} {}", source_config.has_latents, source_config.has_masks);
	}

	if source_config.tags.len() > 0 {
		fields.push_str(",tags");
		println!("Including some tags: {}", source_config.tags);
	}

    if source_config.require_embeddings {
        fields.push_str(",coca_embedding");
		println!("Including embeddings");
    }

    if source_config.duplicate_state >= 0 {
        fields.push_str(",duplicate_state");
    }

    println!("Fields: {}", fields);
    println!("Rank: {}, World size: {}", rank, world_size);
    let mut return_latents = source_config.has_latents.clone();
    if source_config.has_masks.len() > 0 {
        return_latents.push_str(&format!(",{}", source_config.has_masks));
    }

    let maybe_add_int = |value: i32| {
        if value > 0 {
            value.to_string()
        } else {
            "".to_string()
        }
    };

    let db_request = DbRequest {
        fields: fields,
        sources: source_config.sources,
        sources_ne:source_config.sources_ne,
        page_size: page_size.to_string(),
        tags: source_config.tags,
        tags_ne: source_config.tags_ne,
        has_attributes: source_config.has_attributes,
        lacks_attributes: source_config.lacks_attributes,
        has_masks: source_config.has_masks,
        lacks_masks: source_config.lacks_masks,
        has_latents: source_config.has_latents,
        lacks_latents: source_config.lacks_latents,
        return_latents: return_latents,
        min_short_edge: maybe_add_int(source_config.min_short_edge),
        max_short_edge: maybe_add_int(source_config.max_short_edge),
        min_pixel_count: maybe_add_int(source_config.min_pixel_count),
        max_pixel_count: maybe_add_int(source_config.max_pixel_count),
        duplicate_state: maybe_add_int(source_config.duplicate_state),
        random_sampling: source_config.random_sampling,
        partition: if world_size >1 { format!(",{}", rank)} else {"".to_string()},
        partitions_count: if world_size >1 { format!(",{}", world_size) } else {"".to_string()},
    };

    let api_url = std::env::var("DATAROOM_API_URL").expect("DATAROOM_API_URL not set");
    let api_key = std::env::var("DATAROOM_API_KEY").expect("DATAROOM_API_KEY not set");
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Token  {}", api_key)).unwrap(),
    );

    let client = reqwest::blocking::Client::new();

    // Send the request and dispatch the response to the channel
    let mut response_json = serde_json::Value::Null;
    let mut next_url = &serde_json::Value::Null;

    for _ in 0..retries {
        let initial_request = db_request.get_http_request(&api_url, &api_key);
        if let Ok(response) = client.execute(initial_request.try_clone().unwrap()) {
            // We can now deserialize the response, extract the "result" and "next" fields.
            if let Ok(response_text) = response.text() {
                response_json = serde_json::from_str(&response_text).unwrap();
                if let Some(next) = response_json.get("next") {
                    next_url = next;
                } else {
                    println!("No next URL in the response");
                    println!("{:?}", response_text);
                }
                break;
            }
        }
    }

    if response_json == serde_json::Value::Null {
        println!("Failed to get the initial response from the DB");
        return;
    }

    // While we have something in the Send the samples to the channel
    let mut count = 0;
    while count < num_samples && *next_url != serde_json::Value::Null {
        // Push the page to the channel
        match pages_tx.send(response_json.clone()) {
            Ok(_) => {
                println!("ping_pages: sent new page");
            }
            Err(_) => {
                println!("ping_pages: stream, wrapping up");
                break;
            }
        }
        count += page_size;

        // Ask for the next page
        let mut new_request = reqwest::blocking::Request::new(
            reqwest::Method::GET,
            reqwest::Url::parse(next_url.as_str().unwrap()).unwrap(),
        );
        new_request.headers_mut().extend(headers.clone());

        next_url = &serde_json::Value::Null;
        for _ in 0..retries {
            if let Ok(new_response) = client.execute(new_request.try_clone().unwrap()) {
                if !new_response.status().is_success() {
                    println!("Error: {:?}", new_response);
                    continue;
                }

                // TODO: below can fail in its own right, not caught properly for a retry
                response_json = serde_json::from_str(&new_response.text().unwrap()).unwrap();
                next_url = response_json.get("next").unwrap();
                break;
            }
        }
    }

    // Either we don't have any more samples or we have reached the limit
    println!(
        "Total samples requested: {}. Page samples served {}",
        num_samples, count
    );

    // Send an empty value to signal the end of the stream
    match pages_tx.send(serde_json::Value::Null) {
        Ok(_) => {}
        Err(_) => {
            println!("ping_pages: stream already closed, all good");
        }
    };
    println!("ping_pages: down");
}

pub fn pull_pages(
    pages_rx: kanal::Receiver<serde_json::Value>,
    samples_meta_tx: kanal::Sender<serde_json::Value>,
    num_samples: usize,
) {
    // While we have something in the Send the samples to the channel
    let mut count = 0;
    while count < num_samples {
        let response_json = pages_rx.recv().unwrap();
        if response_json == serde_json::Value::Null {
            println!("pull_pages: end of stream received, stopping there");
            break;
        }

        // Go over the samples from the current page
        for sample in response_json.get("results").unwrap().as_array().unwrap() {
            let sample_json = serde_json::from_value(sample.clone()).unwrap();
            samples_meta_tx.send(sample_json).unwrap();
            count += 1;
            if count >= num_samples {
                // NOTE: This doesn´t count the samples which have actually been processed
                // meaning that we'll drop the buffer if we reach the limit
                println!("pull_pages: reached the limit of samples requested. Shutting down");
                pages_rx.close();
                break;
            }
        }
    }

    // Either we don't have any more samples or we have reached the limit
    println!(
        "Total samples requested: {}. Samples served {}",
        num_samples, count
    );

    // Send an empty value to signal the end of the stream
    samples_meta_tx.send(serde_json::Value::Null).unwrap();
    println!("pull_pages: down");
}
