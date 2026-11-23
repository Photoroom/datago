//! One terminal outcome per engine generation. Errors are recorded before queues
//! close, so readers cannot mistake a failed producer for normal exhaustion.

use crate::error::{panic_message, DatagoError};
use crate::structs::Sample;
use crate::worker_utils::WorkerEnd;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use tokio_util::sync::CancellationToken;

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

    pub(crate) fn cancellation(&self) -> &CancellationToken {
        &self.cancel
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

    pub(crate) fn finish_worker(&self, result: Result<WorkerEnd, String>) {
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
        self.spawn(
            sender,
            move |items, control| work(items, control.cancellation()),
            DatagoError::Source,
        )
    }

    pub(crate) fn spawn_worker(
        &self,
        sender: kanal::Sender<Option<Sample>>,
        work: impl FnOnce(kanal::Sender<Option<Sample>>, &Self) -> Result<(), String> + Send + 'static,
    ) -> JoinHandle<()> {
        self.spawn(sender, work, DatagoError::Worker)
    }

    fn spawn<T: Send + 'static>(
        &self,
        keep_alive: kanal::Sender<T>,
        work: impl FnOnce(kanal::Sender<T>, &Self) -> Result<(), String> + Send + 'static,
        classify: fn(String) -> DatagoError,
    ) -> JoinHandle<()> {
        let control = self.clone();
        thread::spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| work(keep_alive.clone(), &control)))
                .unwrap_or_else(|panic| Err(format!("panicked: {}", panic_message(panic))));
            if let Err(message) = result {
                control.transition(Outcome::Failed(classify(message)));
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
