//! The seam between front ends and the back end.
//!
//! A front end is a [`Source`]: it knows where samples come from (`enumerate`)
//! and how to prepare one of them (`load`). The back end is [`launch`]: it owns
//! the queues, the threads and runtime, bounded concurrency, the sample limit,
//! cancellation and failure reporting, identically for every source.
//!
//! ```text
//!  feeder thread (runtime)      worker thread (runtime)              reader
//!  Source::enumerate ─ items ─► Source::load × max_tasks ─ samples ─► client
//! ```
//!
//! Both stages are async and run on runtimes the engine builds, so a source
//! never picks a runtime flavor, sizes one, or watches for cancellation.
//!
//! Adding a source means implementing the trait in one file and adding one
//! `SourceType` variant; nothing in this module needs to change.

use crate::error::DatagoError;
use crate::image_processing::{ARAwareTransform, ImageEncoding};
use crate::pipeline::PipelineControl;
use crate::structs::{DatagoEngine, Sample};
use crate::worker_utils::{max_tasks_from_env, run_worker};
use log::{debug, info};
use serde::de::DeserializeOwned;
use std::future::Future;
use std::sync::Arc;

/// What the client configuration hands to every source.
pub(crate) struct SourceContext {
    pub image_transform: Arc<Option<ARAwareTransform>>,
    pub encoding: ImageEncoding,
    pub max_connections: usize,
}

/// Where samples come from. One value is shared by the feeder thread and all
/// the loading tasks, so it is the natural home for state they have in common
/// (a connection pool, the parsed configuration).
pub(crate) trait Source: Send + Sync + Sized + 'static {
    /// Names the source in logs and configuration errors.
    const NAME: &'static str;

    /// Deserialized from the client's `source_config`.
    type Config: DeserializeOwned;

    /// One unit of work: enumerated by the feeder, loaded by a worker task.
    type Item: Send + 'static;

    /// Check the configuration and build the source. An error rejects the
    /// configuration before any queue or thread exists.
    fn new(config: Self::Config, context: &SourceContext) -> Result<Self, String>;

    /// Send every candidate item, then return. A failed send means the worker
    /// has retired: return `Ok`. An error fails the pass while it is still
    /// running. The engine drops this future once the pass is cancelled, so
    /// blocking work must go to `spawn_blocking` and stop on a failed send.
    fn enumerate(
        &self,
        items: kanal::Sender<Self::Item>,
    ) -> impl Future<Output = Result<(), String>> + Send;

    /// Prepare one sample. `Err(())` skips this item without failing the pass
    /// or counting toward the limit; log the reason before returning it.
    fn load(&self, item: Self::Item) -> impl Future<Output = Result<Sample, ()>> + Send;

    /// How many `load` calls may be in flight at once.
    fn max_tasks(&self) -> usize {
        max_tasks_from_env(num_cpus::get())
    }

    /// Threads of the runtime running `enumerate`. One is plenty to page
    /// through an API; raise it when enumeration does CPU work of its own.
    fn enumeration_threads(&self) -> usize {
        1
    }

    /// How many enumerated items may wait for a loading task.
    fn queue_capacity(samples_buffer: usize) -> usize {
        samples_buffer * 2
    }
}

/// Sources shard their samples over `world_size` ranks; one rank by default.
pub(crate) fn default_world_size() -> usize {
    1
}

/// Reject a `rank` outside of `0..world_size`.
pub(crate) fn check_partition(rank: usize, world_size: usize) -> Result<(), String> {
    if world_size == 0 {
        return Err("world_size must be greater than 0".into());
    }
    if rank >= world_size {
        return Err(format!(
            "rank {rank} must be less than world_size {world_size}"
        ));
    }
    Ok(())
}

/// Start one pass over `S`, or reject its configuration before any queue or
/// thread exists.
pub(crate) fn launch<S: Source>(
    source_config: &serde_json::Value,
    context: &SourceContext,
    samples_buffer: usize,
    limit: usize,
) -> Result<DatagoEngine, DatagoError> {
    let rejected =
        |error: String| DatagoError::Configuration(format!("{} source: {error}", S::NAME));
    let config = serde_json::from_value(source_config.clone())
        .map_err(|error| rejected(error.to_string()))?;
    let source = S::new(config, context).map_err(rejected)?;
    info!("Using {} as source", S::NAME);
    Ok(start_engine(Arc::new(source), samples_buffer, limit))
}

/// Wire the pipeline: a feeder thread enumerating items and a worker thread
/// loading them into the returned engine's sample queue.
fn start_engine<S: Source>(source: Arc<S>, samples_buffer: usize, limit: usize) -> DatagoEngine {
    // Both queues are bounded: every stage pauses once the reader stops pulling.
    let (items_tx, items_rx) = kanal::bounded(S::queue_capacity(samples_buffer));
    let (samples_tx, samples_rx) = kanal::bounded(samples_buffer);
    let control = PipelineControl::new(samples_rx.clone());

    let feeder = control.spawn_feeder(items_tx, source.enumeration_threads(), {
        let source = source.clone();
        move |items_tx| async move { source.enumerate(items_tx).await }
    });
    // Loading is mostly CPU work: no point in more threads than tasks.
    let max_tasks = source.max_tasks();
    let worker_threads = num_cpus::get().min(max_tasks);
    let worker = control.spawn_worker(
        items_rx,
        samples_tx,
        worker_threads,
        move |runtime, items, samples, cancel| {
            let report = runtime.block_on(run_worker(
                items,
                samples,
                max_tasks,
                limit,
                cancel,
                move |item| {
                    let source = source.clone();
                    async move { source.load(item).await }
                },
            ))?;
            debug!("{} source: {} samples delivered", S::NAME, report.delivered);
            Ok(report.end)
        },
    );

    DatagoEngine {
        control,
        samples_rx,
        feeder: Some(feeder),
        worker: Some(worker),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::{ImagePayload, PythonImagePayload};
    use serde::Deserialize;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    const READ_TIMEOUT: Duration = Duration::from_secs(5);

    #[derive(Default, Deserialize)]
    #[serde(default)]
    struct CountingConfig {
        count: usize,
        fail_enumeration: bool,
        panic_on: Option<usize>,
        reject: bool,
    }

    /// Enumerates `0..count`; multiples of three are unreadable.
    struct CountingSource {
        config: CountingConfig,
        in_flight: AtomicUsize,
        most_in_flight: AtomicUsize,
    }

    impl CountingSource {
        const MAX_TASKS: usize = 3;
    }

    impl Source for CountingSource {
        const NAME: &'static str = "counting";
        type Config = CountingConfig;
        type Item = usize;

        fn new(config: Self::Config, _context: &SourceContext) -> Result<Self, String> {
            if config.reject {
                return Err("injected configuration error".into());
            }
            Ok(Self {
                config,
                in_flight: AtomicUsize::new(0),
                most_in_flight: AtomicUsize::new(0),
            })
        }

        async fn enumerate(&self, items: kanal::Sender<usize>) -> Result<(), String> {
            for item in 0..self.config.count {
                if items.as_async().send(item).await.is_err() {
                    return Ok(());
                }
            }
            if self.config.fail_enumeration {
                return Err("injected enumeration failure".into());
            }
            Ok(())
        }

        async fn load(&self, item: usize) -> Result<Sample, ()> {
            let in_flight = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.most_in_flight.fetch_max(in_flight, Ordering::SeqCst);
            tokio::task::yield_now().await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);

            assert_ne!(Some(item), self.config.panic_on, "injected loading panic");
            if item.is_multiple_of(3) {
                return Err(());
            }
            Ok(Sample {
                id: item.to_string(),
                source: Self::NAME.into(),
                image: PythonImagePayload::new(ImagePayload::default()),
                attributes: HashMap::new(),
                duplicate_state: 0,
                masks: HashMap::new(),
                additional_images: HashMap::new(),
                latents: HashMap::new(),
                coca_embedding: vec![],
                tags: vec![],
            })
        }

        fn max_tasks(&self) -> usize {
            Self::MAX_TASKS
        }
    }

    fn context() -> SourceContext {
        SourceContext {
            image_transform: Arc::new(None),
            encoding: ImageEncoding::default(),
            max_connections: 1,
        }
    }

    fn start(config: serde_json::Value, limit: usize) -> DatagoEngine {
        launch::<CountingSource>(&config, &context(), 2, limit).unwrap()
    }

    fn join(mut engine: DatagoEngine) {
        engine.feeder.take().unwrap().join().unwrap();
        engine.worker.take().unwrap().join().unwrap();
    }

    /// Read until the sample queue closes, as the client does.
    fn drain(engine: &DatagoEngine) -> Vec<usize> {
        let mut ids = Vec::new();
        while let Ok(Some(sample)) = engine.samples_rx.recv_timeout(READ_TIMEOUT) {
            ids.push(sample.id.parse().unwrap());
        }
        ids.sort_unstable();
        ids
    }

    #[test]
    fn delivers_every_loadable_item_then_closes_the_queue() {
        let engine = start(json!({"count": 10}), 100);
        assert_eq!(drain(&engine), [1, 2, 4, 5, 7, 8]);
        assert!(engine.control.error().is_none());
        // Exhaustion is the queue closing, not a marker sample.
        assert!(engine.samples_rx.recv().is_err());
    }

    #[test]
    fn an_empty_source_is_exhausted_not_failed() {
        let engine = start(json!({"count": 0}), 100);
        assert!(drain(&engine).is_empty());
        assert!(engine.control.error().is_none());
    }

    #[test]
    fn stops_at_the_limit_and_retires_the_feeder() {
        // Far more items than the bounded queues hold: the feeder can only
        // finish once reaching the limit closes the item queue under it.
        let engine = start(json!({"count": 100_000}), 5);
        assert_eq!(drain(&engine).len(), 5);
        assert!(engine.control.error().is_none());
        join(engine);
    }

    #[test]
    fn enumeration_failure_is_a_source_error() {
        let engine = start(json!({"count": 3, "fail_enumeration": true}), 100);
        drain(&engine);
        assert_eq!(
            engine.control.error(),
            Some(DatagoError::Source("injected enumeration failure".into()))
        );
    }

    #[test]
    fn loading_panic_is_a_worker_error() {
        let engine = start(json!({"count": 20, "panic_on": 13}), 100);
        drain(&engine);
        assert!(matches!(
            engine.control.error(),
            Some(DatagoError::Worker(message)) if message.contains("sample task failed")
        ));
    }

    #[test]
    fn invalid_configuration_names_the_source_and_starts_nothing() {
        let error =
            launch::<CountingSource>(&json!({"count": "many"}), &context(), 2, 1).unwrap_err();
        assert!(matches!(
            error,
            DatagoError::Configuration(message) if message.starts_with("counting source: ")
        ));
    }

    #[test]
    fn a_source_can_reject_its_configuration() {
        let error =
            launch::<CountingSource>(&json!({"reject": true}), &context(), 2, 1).unwrap_err();
        assert_eq!(
            error,
            DatagoError::Configuration("counting source: injected configuration error".into())
        );
    }

    #[test]
    fn concurrency_is_bounded_by_the_source() {
        let config = CountingConfig {
            count: 300,
            ..Default::default()
        };
        let source = Arc::new(CountingSource::new(config, &context()).unwrap());
        let engine = start_engine(source.clone(), 2, 1000);
        assert_eq!(drain(&engine).len(), 200);
        let most_in_flight = source.most_in_flight.load(Ordering::SeqCst);
        assert!(
            (1..=CountingSource::MAX_TASKS).contains(&most_in_flight),
            "{most_in_flight} tasks in flight"
        );
    }

    #[test]
    fn stopping_releases_a_feeder_and_worker_blocked_on_full_queues() {
        let engine = start(json!({"count": 100_000}), 100_000);
        // Nobody reads: both queues fill up and both producers block.
        engine.close();
        assert!(engine.control.error().is_none());
        join(engine);
    }
}
