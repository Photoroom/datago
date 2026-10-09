//! Small cancellation-aware helpers shared by the asynchronous source workers.

use tokio::task::{JoinError, JoinSet};
use tokio_util::sync::CancellationToken;

pub enum NextTask<T> {
    Completed(Option<Result<T, JoinError>>),
    Cancelled,
}

/// Wait for the next task to finish, cancel all remaining work as soon as the
/// engine's cancellation token flips. `stop()` flips the token (see
/// `DatagoClient::take_engine_if`), so a stalled download or decode is dropped
/// instead of being awaited to completion.
pub async fn join_next_or_cancelled<T: Send + 'static>(
    tasks: &mut JoinSet<T>,
    cancel: &CancellationToken,
) -> NextTask<T> {
    tokio::select! {
        result = tasks.join_next() => NextTask::Completed(result),
        _ = cancel.cancelled() => {
            tasks.abort_all();
            NextTask::Cancelled
        }
    }
}
