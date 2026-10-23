//! One terminal outcome per engine generation. Errors are recorded before queues
//! close, so readers cannot mistake a failed producer for normal exhaustion.

use crate::error::{panic_message, DatagoError};
use crate::structs::{MetadataReceiver, Sample};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
enum Outcome {
    Running,
    Stopped,
    Failed(DatagoError),
}

#[derive(Clone, Debug)]
pub(crate) struct PipelineControl {
    outcome: Arc<Mutex<Outcome>>,
    samples_rx: kanal::Receiver<Option<Sample>>,
    metadata_rx: MetadataReceiver,
    cancel: CancellationToken,
}

impl PipelineControl {
    pub(crate) fn new(
        samples_rx: kanal::Receiver<Option<Sample>>,
        metadata_rx: MetadataReceiver,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Outcome::Running)),
            samples_rx,
            metadata_rx,
            cancel,
        }
    }

    pub(crate) fn error(&self) -> Option<DatagoError> {
        match &*self.outcome.lock().unwrap() {
            Outcome::Failed(error) => Some(error.clone()),
            _ => None,
        }
    }

    pub(crate) fn mark_stopped(&self) {
        let mut outcome = self.outcome.lock().unwrap();
        if matches!(*outcome, Outcome::Running) {
            *outcome = Outcome::Stopped;
        }
    }

    fn fail(&self, error: DatagoError) {
        // A worker reaching its limit closes metadata to retire the feeder.
        // A resulting download/send failure is expected, not a failed pass.
        if matches!(error, DatagoError::Source(_)) && self.metadata_rx.is_disconnected() {
            return;
        }
        let mut outcome = self.outcome.lock().unwrap();
        if !matches!(*outcome, Outcome::Running) {
            return;
        }
        *outcome = Outcome::Failed(error);
        self.close();
    }

    fn close(&self) {
        self.cancel.cancel();
        self.metadata_rx.close();
        let _ = self.samples_rx.close();
    }

    pub(crate) fn spawn_feeder<T: Send + 'static>(
        &self,
        keep_alive: kanal::Sender<T>,
        work: impl FnOnce() -> Result<(), String> + Send + 'static,
    ) -> JoinHandle<()> {
        self.spawn(keep_alive, work, DatagoError::Source)
    }

    pub(crate) fn spawn_worker(
        &self,
        keep_alive: kanal::Sender<Option<Sample>>,
        work: impl FnOnce() -> Result<(), String> + Send + 'static,
    ) -> JoinHandle<()> {
        self.spawn(keep_alive, work, DatagoError::Worker)
    }

    fn spawn<T: Send + 'static>(
        &self,
        keep_alive: kanal::Sender<T>,
        work: impl FnOnce() -> Result<(), String> + Send + 'static,
        classify: fn(String) -> DatagoError,
    ) -> JoinHandle<()> {
        let control = self.clone();
        thread::spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(work))
                .unwrap_or_else(|panic| Err(format!("panicked: {}", panic_message(panic))));
            if let Err(message) = result {
                control.fail(classify(message));
            }
            // Hold a sender until the producer's result is recorded. Dropping
            // its last sender then wakes readers on a successful end-of-stream.
            drop(keep_alive);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control() -> (
        PipelineControl,
        kanal::Sender<serde_json::Value>,
        kanal::Sender<Option<Sample>>,
    ) {
        let (samples_tx, samples_rx) = kanal::bounded(1);
        let (metadata_tx, metadata_rx) = kanal::bounded(1);
        (
            PipelineControl::new(
                samples_rx,
                MetadataReceiver::Json(metadata_rx),
                CancellationToken::new(),
            ),
            metadata_tx,
            samples_tx,
        )
    }

    #[test]
    fn feeder_panics_are_recorded_before_readers_wake() {
        let (control, metadata_tx, _samples_tx) = control();
        let feeder = control.spawn_feeder(metadata_tx, || panic!("injected feeder panic"));
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
        let (control, metadata_tx, _samples_tx) = control();
        control.mark_stopped();
        let feeder = control.spawn_feeder(metadata_tx, || Err("late feeder failure".into()));
        feeder.join().unwrap();
        assert!(control.error().is_none());
    }
}
