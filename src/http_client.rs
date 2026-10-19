use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use std::sync::Arc;

// We'll share a single connection pool across all worker threads
#[derive(Clone)]
pub struct SharedClient {
    pub client: ClientWithMiddleware,
    pub semaphore: Arc<tokio::sync::Semaphore>,
    pub max_connections: usize,
}

pub fn new_shared_client(max_connections: usize) -> SharedClient {
    let retry_policy = ExponentialBackoff::builder()
        .retry_bounds(
            std::time::Duration::from_millis(100), // min_retry_interval
            std::time::Duration::from_secs(3),
        )
        .build_with_max_retries(3);

    let client = ClientBuilder::new(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap(),
    )
    .with(RetryTransientMiddleware::new_with_policy(retry_policy))
    .build();

    SharedClient {
        client,
        semaphore: Arc::new(tokio::sync::Semaphore::new(max_connections)),
        max_connections,
    }
}
