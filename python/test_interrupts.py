"""Exercise GIL release, re-entrant stop, and cross-thread cancellation out of process."""

import json
import os
import select
import signal
import subprocess
import sys
import textwrap
from time import monotonic

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="uses local HTTP and SIGINT"
)

SCRIPT = textwrap.dedent("""
    import json
    import os
    import signal
    import sys
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from datago import DatagoClient

    request_started = threading.Event()
    release_response = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_started.set()
            release_response.wait(30)
            body = b'{"results": [], "next": null}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["DATAROOM_API_URL"] = f"http://127.0.0.1:{server.server_port}/"
    os.environ["DATAROOM_API_KEY"] = "test"
    config = json.dumps({
        "source_type": "db",
        "source_config": {"sources": "test", "page_size": 1},
        "limit": 1,
        "samples_buffer_size": 1,
    })
    mode = sys.argv[1]
    client = DatagoClient(config)

    try:
        client.start()
        assert request_started.wait(10), "Datago feeder did not enter the blocked HTTP request"
        if mode == "background":
            reader_started = threading.Event()

            def background_read():
                reader_started.set()
                client.get_sample()

            reader = threading.Thread(target=background_read)
            reader.start()
            assert reader_started.wait(10)

            signal_handled = threading.Event()
            def stop_from_signal(*_):
                client.stop()
                signal_handled.set()

            signal.signal(signal.SIGINT, stop_from_signal)
            print("READY", flush=True)
            assert signal_handled.wait(4), "main-thread SIGINT handler did not run"
            reader.join(timeout=2)
            assert not reader.is_alive(), "signal handler did not stop the background reader"
            print("READER_STOPPED", flush=True)
        elif mode == "cross-thread-stop":
            reader_started = threading.Event()

            def background_read():
                reader_started.set()
                client.get_sample()

            reader = threading.Thread(target=background_read)
            reader.start()
            assert reader_started.wait(10)
            print("READY", flush=True)
            client.stop()
            reader.join(timeout=2)
            assert not reader.is_alive(), "stop() did not cancel the background reader"
            print("READER_STOPPED", flush=True)
        elif mode == "handler":
            signal.signal(signal.SIGINT, lambda *_: client.stop())
            print("READY", flush=True)
            sample = client.get_sample()
            assert sample is None
            print("STOPPED", flush=True)
        elif mode == "drop":
            print("READY", flush=True)
            del client
            client = None
            print("DROPPED", flush=True)
        else:
            print("READY", flush=True)
            if mode == "convert":
                client.get_sample_auto_convert()
            else:
                client.get_sample()
    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
    finally:
        if client is not None:
            client.stop()
        release_response.set()
        server.shutdown()
        server.server_close()
""")


@pytest.mark.parametrize(
    "mode", ["direct", "convert", "background", "handler", "cross-thread-stop", "drop"]
)
def test_sigint_interrupts_native_waits(mode):
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", SCRIPT, mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env={**os.environ, "DATAGO_MAX_TASKS": "1"},
    )
    try:
        assert child.stdout is not None
        marker = bytearray()
        deadline = monotonic() + 10
        while not marker.endswith(b"\n") and monotonic() < deadline:
            readable, _, _ = select.select([child.stdout], [], [], 0.1)
            if readable:
                # Read one byte at a time: TextIO.readline() may buffer subsequent
                # output, which communicate() cannot see after the marker.
                marker.extend(os.read(child.stdout.fileno(), 1))
        assert bytes(marker) == b"READY\n", bytes(marker)
        if mode not in {"cross-thread-stop", "drop"}:
            child.send_signal(signal.SIGINT)
        output, errors = child.communicate(timeout=4)
        output = output.decode()
        errors = errors.decode()
        assert child.returncode == 0, errors
        if mode == "handler":
            assert "STOPPED" in output, (output, errors)
        elif mode in {"background", "cross-thread-stop"}:
            assert "READER_STOPPED" in output, (output, errors)
        elif mode == "drop":
            assert "DROPPED" in output, (output, errors)
        else:
            assert "INTERRUPTED" in output, (output, errors)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_stop_is_idempotent_and_explicit_start_restarts(tmp_path):
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
            import json, sys
            from datago import DatagoClient
            client = DatagoClient(sys.argv[1])
            for _ in range(3):
                client.start()
                client.stop()
                client.stop()
            client.start()
            assert client.get_sample() is not None
            client.stop()
            assert client.get_sample() is None  # Reads do not silently restart.
            client.start()  # An explicit start begins a new pass.
            assert client.get_sample() is not None
            client.stop()
            print("STOPPED")
        """),
            json.dumps(config),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "DATAGO_MAX_TASKS": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "STOPPED" in result.stdout
