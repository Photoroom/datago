//! Scheduling policy shared by source workers. Source tasks prepare samples;
//! only this coordinator publishes them and counts successful deliveries.

use crate::structs::Sample;
use std::future::Future;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Resolve the maximum number of in-flight processing tasks. `DATAGO_MAX_TASKS`
/// overrides `default`; unset or unparsable values fall back to it. An explicit
/// `0` is honored and serializes processing.
pub fn max_tasks_from_env(default: usize) -> usize {
    parse_max_tasks(std::env::var("DATAGO_MAX_TASKS").ok(), default)
}

fn parse_max_tasks(value: Option<String>, default: usize) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Process at most `max_tasks` items concurrently and publish at most `limit`
/// successful samples. Failed items do not consume the delivery budget.
///
/// Input, task completion, and output backpressure all observe cancellation.
/// After input ends we keep selecting task completion against cancellation;
/// there is no separate, uncancellable final drain.
pub async fn run_worker<M, F, Fut>(
    metadata_rx: kanal::Receiver<M>,
    samples_tx: kanal::Sender<Option<Sample>>,
    max_tasks: usize,
    limit: usize,
    cancel: CancellationToken,
    is_end: impl Fn(&M) -> bool,
    process: F,
) -> Result<usize, String>
where
    M: Send + 'static,
    F: Fn(M) -> Fut,
    Fut: Future<Output = Result<Sample, ()>> + Send + 'static,
{
    let mut tasks = JoinSet::new();
    let mut input_finished = false;
    let mut delivered = 0;
    let mut failure = None;
    // DATAGO_MAX_TASKS=0 historically means serial processing.
    let max_tasks = max_tasks.max(1);
    // A Kanal recv future can own an arriving message before its final poll.
    // Dropping it when another select branch wins loses that message. Keep the
    // same pinned future until it completes, then replace it with a fresh one.
    // Kanal 0.1.1's stream reuses a completed signal and can read a value twice.
    let metadata = metadata_rx.as_async().recv();
    tokio::pin!(metadata);

    while delivered < limit && (!input_finished || !tasks.is_empty()) {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = tasks.join_next(), if !tasks.is_empty() => {
                match result {
                    Some(Ok(Ok(sample))) => {
                        let sent = tokio::select! {
                            biased;
                            _ = cancel.cancelled() => break,
                            sent = samples_tx.as_async().send(Some(sample)) => sent,
                        };
                        if sent.is_err() {
                            break;
                        }
                        delivered += 1;
                    }
                    Some(Ok(Err(()))) => {} // Source logged the failed item; try another.
                    Some(Err(error)) => {
                        failure = Some(format!("sample task failed: {error}"));
                        break;
                    }
                    None => {}
                }
            }
            received = &mut metadata,
                if !input_finished && tasks.len() < max_tasks => {
                metadata.set(metadata_rx.as_async().recv());
                match received {
                    Ok(item) if !is_end(&item) => { tasks.spawn(process(item)); }
                    _ => {
                        input_finished = true;
                        // The engine retains another receiver; explicitly close
                        // this queue so the feeder can always finish.
                        let _ = metadata_rx.close();
                    }
                }
            }
        }
    }

    let _ = metadata_rx.close();
    // Reaching the delivery limit must discard excess results, not publish them.
    // Aborted spawn_blocking IO can outlive its future; runtime teardown uses a
    // separate bounded shutdown for those non-cancellable native operations.
    tasks.abort_all();
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            if !error.is_cancelled() && failure.is_none() {
                failure = Some(format!("sample task failed: {error}"));
            }
        }
    }
    tokio::select! {
        biased;
        _ = cancel.cancelled() => {}
        _ = samples_tx.as_async().send(None) => {}
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(delivered),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_max_tasks, run_worker};
    use crate::structs::{ImagePayload, PythonImagePayload, Sample};
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    fn sample(id: usize) -> Sample {
        Sample {
            id: id.to_string(),
            source: "test".into(),
            image: PythonImagePayload::new(ImagePayload::default()),
            attributes: Default::default(),
            duplicate_state: 0,
            masks: Default::default(),
            additional_images: Default::default(),
            latents: Default::default(),
            coca_embedding: vec![],
            tags: vec![],
        }
    }

    #[tokio::test]
    async fn only_successful_deliveries_count_toward_exact_limit() {
        let (metadata_tx, metadata_rx) = kanal::bounded(32);
        let (samples_tx, samples_rx) = kanal::bounded(32);
        for id in 0..20 {
            metadata_tx.send(Some(id)).unwrap();
        }
        metadata_tx.send(None).unwrap();
        let delivered = run_worker(
            metadata_rx,
            samples_tx,
            4,
            5,
            CancellationToken::new(),
            Option::is_none,
            |id| async move {
                let id = id.unwrap();
                if id % 3 == 0 {
                    Err(())
                } else {
                    Ok(sample(id))
                }
            },
        )
        .await
        .unwrap();
        let mut ids = Vec::new();
        while let Some(sample) = samples_rx.recv().unwrap() {
            let id = sample.id.parse::<usize>().unwrap();
            assert_ne!(id % 3, 0);
            ids.push(id);
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(delivered, 5);
        assert_eq!(ids.len(), 5);
    }

    #[tokio::test]
    async fn cancellation_after_end_of_input_aborts_pending_work() {
        let (metadata_tx, metadata_rx) = kanal::bounded(2);
        let (samples_tx, _samples_rx) = kanal::bounded(2);
        metadata_tx.send(Some(1)).unwrap();
        metadata_tx.send(None).unwrap();
        let cancel = CancellationToken::new();
        let worker_cancel = cancel.clone();
        let worker = tokio::spawn(async move {
            run_worker(
                metadata_rx,
                samples_tx,
                4,
                10,
                worker_cancel,
                Option::is_none,
                |_| std::future::pending::<Result<Sample, ()>>(),
            )
            .await
        });
        // Closing the queue proves the worker entered its final completion phase.
        tokio::time::timeout(Duration::from_secs(2), async {
            while !metadata_tx.is_disconnected() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        cancel.cancel();
        let delivered = tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(delivered, 0);
    }

    #[tokio::test]
    async fn cancellation_releases_output_backpressure() {
        let (metadata_tx, metadata_rx) = kanal::bounded(2);
        let (samples_tx, _samples_rx) = kanal::bounded(1);
        samples_tx.send(Some(sample(0))).unwrap(); // Consumer is not draining.
        metadata_tx.send(Some(1)).unwrap();
        let cancel = CancellationToken::new();
        let worker_cancel = cancel.clone();
        let (started_tx, started_rx) = kanal::bounded(1);
        let worker = tokio::spawn(async move {
            run_worker(
                metadata_rx,
                samples_tx,
                1,
                10,
                worker_cancel,
                Option::is_none,
                move |id| {
                    started_tx.send(()).unwrap();
                    async move { Ok(sample(id.unwrap())) }
                },
            )
            .await
        });
        started_rx.as_async().recv().await.unwrap();
        tokio::task::yield_now().await;
        cancel.cancel();
        let delivered = tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(delivered, 0);
    }

    #[tokio::test]
    async fn completed_task_does_not_discard_arriving_metadata() {
        let (metadata_tx, metadata_rx) = kanal::bounded(2);
        let (samples_tx, samples_rx) = kanal::bounded(4);
        let prepared = std::sync::Arc::new(tokio::sync::Notify::new());
        metadata_tx.send(Some(0)).unwrap();
        let mut worker = Box::pin(run_worker(
            metadata_rx,
            samples_tx,
            2,
            10,
            CancellationToken::new(),
            Option::is_none,
            |id| {
                let prepared = prepared.clone();
                async move {
                    if id == Some(0) {
                        prepared.notify_one();
                    }
                    Ok(sample(id.unwrap()))
                }
            },
        ));

        // Start the first task and leave the coordinator awaiting more metadata.
        assert!(futures::poll!(worker.as_mut()).is_pending());
        prepared.notified().await;
        // Both task completion and the metadata receive are now ready. The
        // biased select handles completion first; the receive must survive it.
        metadata_tx.send(Some(1)).unwrap();
        assert!(futures::poll!(worker.as_mut()).is_pending());
        metadata_tx.send(None).unwrap();

        assert_eq!(worker.await.unwrap(), 2);
        let mut ids = Vec::new();
        while let Some(sample) = samples_rx.recv().unwrap() {
            ids.push(sample.id);
        }
        ids.sort();
        assert_eq!(ids, ["0", "1"]);
    }

    #[tokio::test]
    async fn repeated_pending_receive_does_not_replay_metadata() {
        // Copy metadata exposes a replay without double-freeing an owned payload.
        let (metadata_tx, metadata_rx) = kanal::bounded(2);
        let (samples_tx, _samples_rx) = kanal::bounded(4);
        let processed = std::sync::Mutex::new(Vec::new());
        let mut worker = Box::pin(run_worker(
            metadata_rx,
            samples_tx,
            4,
            10,
            CancellationToken::new(),
            Option::is_none,
            |id: Option<usize>| {
                processed.lock().unwrap().push(id.unwrap());
                std::future::pending::<Result<Sample, ()>>()
            },
        ));

        // Force two separate waits on an empty queue. A reused receive signal
        // must not report its previous delivery as another completed receive.
        assert!(futures::poll!(worker.as_mut()).is_pending());
        metadata_tx.send(Some(7)).unwrap();
        assert!(futures::poll!(worker.as_mut()).is_pending());
        assert!(futures::poll!(worker.as_mut()).is_pending());
        metadata_tx.close().unwrap();
        assert_eq!(*processed.lock().unwrap(), [7]);
    }

    #[test]
    fn unset_and_invalid_max_tasks_fall_back_to_default() {
        // The original bug: an unset DATAGO_MAX_TASKS was parsed as 0, which
        // serialized the HTTP/WDS workers instead of using the CPU default.
        assert_eq!(parse_max_tasks(None, 16), 16);
        assert_eq!(parse_max_tasks(Some(String::new()), 16), 16);
        assert_eq!(parse_max_tasks(Some("garbage".to_string()), 16), 16);
    }

    #[test]
    fn explicit_max_tasks_is_honored() {
        assert_eq!(parse_max_tasks(Some("8".to_string()), 16), 8);
        assert_eq!(parse_max_tasks(Some("0".to_string()), 16), 0);
    }
}
