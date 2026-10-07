//! Small cancellation-aware helpers shared by the asynchronous source workers.

use crate::structs::Sample;
use kanal::Sender;
use tokio::task::{JoinError, JoinSet};

pub enum NextTask<T> {
    Completed(Option<Result<T, JoinError>>),
    OutputClosed,
}

const OUTPUT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

/// Wait for a task, but keep observing the downstream receiver. Channel closure
/// is the cross-source cancellation signal; polling is bounded to 20 ms because
/// kanal's synchronous Sender does not expose an async `closed()` future.
pub async fn join_next_or_output_closed<T: Send + 'static>(
    tasks: &mut JoinSet<T>,
    output: &Sender<Option<Sample>>,
) -> NextTask<T> {
    loop {
        tokio::select! {
            result = tasks.join_next() => return NextTask::Completed(result),
            _ = tokio::time::sleep(OUTPUT_POLL_INTERVAL) => {
                if output.is_closed() {
                    tasks.abort_all();
                    return NextTask::OutputClosed;
                }
            }
        }
    }
}
