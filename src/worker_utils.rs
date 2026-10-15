//! Small cancellation-aware helpers shared by the asynchronous source workers.

use tokio::task::{JoinError, JoinSet};
use tokio_util::sync::CancellationToken;

pub enum NextTask<T> {
    Completed(Option<Result<T, JoinError>>),
    Cancelled,
}

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

#[cfg(test)]
mod tests {
    use super::parse_max_tasks;

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
