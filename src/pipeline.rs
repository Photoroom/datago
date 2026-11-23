//! One terminal outcome per engine generation. Errors are recorded before queues
//! close, so readers cannot mistake a failed producer for normal exhaustion.

use crate::error::{panic_message, DatagoError};
use crate::structs::Sample;
use crate::worker_utils::WorkerEnd;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::runtime::{Builder, Runtime};
use tokio_util::sync::CancellationToken;

const RUNTIME_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug)]
enum Outcome {
    Running,
    Exhausted,
    LimitReached,
    Stopped,
    Failed(DatagoError),
}

#[derive(Clone, Debug)]
pub(crate) struct PipelineControl {
    outcome: Arc<Mutex<Outcome>>,
    samples_rx: kanal::Receiver<Option<Sample>>,
    cancel: CancellationToken,
}

impl PipelineControl {
    pub(crate) fn new(samples_rx: kanal::Receiver<Option<Sample>>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Outcome::Running)),
            samples_rx,
            cancel: CancellationToken::new(),
        }
    }

    pub(crate) fn error(&self) -> Option<DatagoError> {
        match &*self.outcome.lock().unwrap() {
            Outcome::Failed(error) => Some(error.clone()),
            _ => None,
        }
    }

    pub(crate) fn stop(&self) {
        self.transition(Outcome::Stopped);
        // Explicit stop discards the buffer even if the worker already finished.
        self.cancel.cancel();
        let _ = self.samples_rx.close();
    }

    fn finish_worker(&self, result: Result<WorkerEnd, String>) {
        match result {
            Ok(WorkerEnd::Exhausted) => self.transition(Outcome::Exhausted),
            Ok(WorkerEnd::LimitReached) => self.transition(Outcome::LimitReached),
            Ok(WorkerEnd::Cancelled) => self.stop(),
            Err(message) => self.transition(Outcome::Failed(DatagoError::Worker(message))),
        }
    }

    fn transition(&self, end: Outcome) {
        let mut outcome = self.outcome.lock().unwrap();
        if !matches!(*outcome, Outcome::Running) {
            return;
        }
        *outcome = end;
        match *outcome {
            // On success the last sample sender drops normally so readers can
            // still drain buffered samples.
            Outcome::LimitReached => self.cancel.cancel(),
            Outcome::Failed(_) => {
                self.cancel.cancel();
                let _ = self.samples_rx.close();
            }
            _ => {}
        }
    }

    pub(crate) fn spawn_feeder<T: Send + 'static>(
        &self,
        sender: kanal::Sender<T>,
        work: impl FnOnce(kanal::Sender<T>, &CancellationToken) -> Result<(), String> + Send + 'static,
    ) -> JoinHandle<()> {
        let control = self.clone();
        thread::spawn(move || {
            if let Err(message) = catch_stage(|| work(sender.clone(), &control.cancel)) {
                control.transition(Outcome::Failed(DatagoError::Source(message)));
            }
            // Publish any failure before the worker can observe end of input.
            drop(sender);
        })
    }

    pub(crate) fn spawn_worker<T: Send + 'static>(
        &self,
        items: kanal::Receiver<T>,
        samples: kanal::Sender<Option<Sample>>,
        work: impl FnOnce(
                &Runtime,
                &kanal::Receiver<T>,
                kanal::Sender<Option<Sample>>,
                &CancellationToken,
            ) -> Result<WorkerEnd, String>
            + Send
            + 'static,
    ) -> JoinHandle<()> {
        let control = self.clone();
        thread::spawn(move || {
            let runtime = catch_stage(|| {
                Builder::new_multi_thread()
                    .worker_threads(num_cpus::get())
                    .enable_all()
                    .build()
                    .map_err(|error| format!("cannot start the worker runtime: {error}"))
            });
            let result = match &runtime {
                Ok(runtime) => {
                    catch_stage(|| work(runtime, &items, samples.clone(), &control.cancel))
                }
                Err(error) => Err(error.clone()),
            };
            control.finish_worker(result);

            // Release blocked sends and signal end of output before waiting for
            // native operations that may outlive cancellation.
            drop(items);
            drop(samples);
            if let Ok(runtime) = runtime {
                runtime.shutdown_timeout(RUNTIME_SHUTDOWN_TIMEOUT);
            }
        })
    }
}

fn catch_stage<T>(work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(work))
        .unwrap_or_else(|panic| Err(format!("panicked: {}", panic_message(panic))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feeder_panics_are_recorded_before_readers_wake() {
        let (_samples_tx, samples_rx) = kanal::bounded(1);
        let (metadata_tx, _metadata_rx) = kanal::bounded::<()>(1);
        let control = PipelineControl::new(samples_rx);
        let feeder = control.spawn_feeder(metadata_tx, |_, _| panic!("injected feeder panic"));
        assert!(control
            .samples_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .is_err());
        feeder.join().unwrap();
        assert!(
            matches!(control.error(), Some(DatagoError::Source(message)) if message.contains("injected feeder panic"))
        );
        assert!(control.cancel.is_cancelled());
    }

    #[test]
    fn explicit_stop_wins_over_a_later_failure() {
        let (_samples_tx, samples_rx) = kanal::bounded(1);
        let (metadata_tx, _metadata_rx) = kanal::bounded::<()>(1);
        let control = PipelineControl::new(samples_rx);
        control.stop();
        let feeder = control.spawn_feeder(metadata_tx, |_, _| Err("late feeder failure".into()));
        feeder.join().unwrap();
        assert!(control.error().is_none());
    }

    #[test]
    fn feeder_error_after_worker_limit_is_not_a_failure() {
        let (samples_tx, samples_rx) = kanal::bounded(1);
        let (metadata_tx, _metadata_rx) = kanal::bounded::<()>(1);
        let control = PipelineControl::new(samples_rx);
        samples_tx.send(None).unwrap();

        control.finish_worker(Ok(WorkerEnd::LimitReached));
        let feeder = control.spawn_feeder(metadata_tx, |_, _| Err("late metadata error".into()));
        feeder.join().unwrap();
        assert!(control.error().is_none());
        assert!(control.cancel.is_cancelled());
        drop(samples_tx);
        assert!(matches!(control.samples_rx.recv(), Ok(None)));
        assert!(control.samples_rx.recv().is_err());
    }

    #[test]
    fn feeder_error_before_worker_limit_is_a_failure() {
        let (_samples_tx, samples_rx) = kanal::bounded(1);
        let (metadata_tx, metadata_rx) = kanal::bounded::<()>(1);
        let control = PipelineControl::new(samples_rx);
        // Queue closure alone must not suppress a source error.
        drop(metadata_rx);
        let feeder =
            control.spawn_feeder(metadata_tx, |_, _| Err("metadata request failed".into()));
        feeder.join().unwrap();
        assert!(matches!(control.error(), Some(DatagoError::Source(message))
            if message == "metadata request failed"));
        assert!(control.cancel.is_cancelled());
    }
}
