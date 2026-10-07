"""Run real SIGINT tests out of process so a GIL regression cannot hang pytest."""

import json
import os
import select
import signal
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="uses POSIX FIFOs and SIGINT"
)

SCRIPT = textwrap.dedent("""
    import json
    import sys
    import threading
    import time
    from datago import DatagoClient

    config = json.dumps({
        "source_type": "file",
        "source_config": {"root_path": sys.argv[1]},
        "limit": 1,
        "samples_buffer_size": 1,
    })
    mode = sys.argv[2]

    def read():
        client = DatagoClient(config)
        client.start()
        # Let the native file worker enter the FIFO's blocking open().
        time.sleep(0.2)
        print("READY", flush=True)
        if mode == "stop":
            client.stop()
        elif mode == "convert":
            client.get_sample_auto_convert()
        else:
            client.get_sample()

    try:
        if mode == "background":
            threading.Thread(target=read, daemon=True).start()
            # SIGINT must reach Python even when the native reader is on another thread.
            time.sleep(3600)
        else:
            read()
    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
""")


@pytest.mark.parametrize("mode", ["direct", "convert", "background", "stop"])
def test_sigint_interrupts_native_waits(tmp_path, mode):
    os.mkfifo(tmp_path / "blocked.png")
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", SCRIPT, str(tmp_path), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "DATAGO_MAX_TASKS": "1"},
    )
    try:
        assert child.stdout is not None
        readable, _, _ = select.select([child.stdout], [], [], 10)
        assert readable, "reader did not start"
        assert child.stdout.readline().strip() == "READY"
        # The marker is just before the native call. Avoid signalling before entry.
        time.sleep(0.2)
        child.send_signal(signal.SIGINT)
        output, errors = child.communicate(timeout=4)
        assert child.returncode == 0, errors
        assert "INTERRUPTED" in output, (output, errors)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_stop_unblocks_backpressured_queues(tmp_path):
    from PIL import Image

    for index in range(32):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{index}.png")
    config = {
        "source_type": "file",
        "source_config": {"root_path": str(tmp_path)},
        "limit": 32,
        "samples_buffer_size": 1,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import json, sys, time
            from datago import DatagoClient
            client = DatagoClient(sys.argv[1])
            for _ in range(3):
                client.start()
                time.sleep(0.2)  # Fill both bounded queues without consuming samples.
                client.stop()
                client.stop()
            print("STOPPED")
        """),
            json.dumps(config),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "DATAGO_MAX_TASKS": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "STOPPED" in result.stdout
