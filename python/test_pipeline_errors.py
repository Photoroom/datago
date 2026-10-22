"""Fatal source failures are observable; exhaustion and stop are normal results."""

import io
import json
import tarfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from datago import DatagoClient
from PIL import Image


@contextmanager
def serve(handler):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, body = handler(self.path)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The client may cancel an in-flight request.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def db_client(monkeypatch, url):
    monkeypatch.setenv("DATAROOM_API_URL", url + "/")
    monkeypatch.setenv("DATAROOM_API_KEY", "test")
    return DatagoClient(
        json.dumps(
            {
                "source_type": "db",
                "source_config": {"sources": "mock", "page_size": 10},
                "samples_buffer_size": 2,
                "limit": 10,
            }
        )
    )


@pytest.mark.parametrize("read", ["get_sample", "get_sample_auto_convert"])
@pytest.mark.parametrize(
    "status,body", [(401, b"unauthorized"), (200, b"invalid JSON"), (200, b"{}")]
)
def test_metadata_errors_raise_runtime_error(monkeypatch, read, status, body):
    with serve(lambda _: (status, body)) as url:
        client = db_client(monkeypatch, url)
        try:
            with pytest.raises(RuntimeError, match="source failed"):
                getattr(client, read)(timeout=3)
            assert client.get_sample() is None
        finally:
            client.stop()


def test_failure_after_a_successful_page_is_not_exhaustion(monkeypatch):
    release = threading.Event()
    with serve(lambda path: response(path)) as url:

        def response(path):
            if path == "/next":
                assert release.wait(5)
                return 401, b"unauthorized"
            return 200, json.dumps(
                {
                    "results": [{"id": "one", "source": "mock", "attributes": {}}],
                    "next": url + "/next",
                }
            ).encode()

        client = db_client(monkeypatch, url)
        try:
            assert client.get_sample(timeout=3).id == "one"
            release.set()
            with pytest.raises(RuntimeError, match="source failed"):
                client.get_sample(timeout=3)
        finally:
            release.set()
            client.stop()


@pytest.mark.parametrize("read", ["get_sample", "get_sample_auto_convert"])
def test_timeout_cancels_the_pass_and_stop_stays_normal(monkeypatch, read):
    started, release = threading.Event(), threading.Event()

    def response(_):
        started.set()
        release.wait(5)
        return 200, b'{"results": [], "next": null}'

    with serve(response) as url:
        client = db_client(monkeypatch, url)
        try:
            client.start()
            assert started.wait(3)
            with pytest.raises(TimeoutError):
                getattr(client, read)(timeout=0.01)
            assert client.get_sample() is None
            client.start()
            reader_result = []
            reader = threading.Thread(
                target=lambda: reader_result.append(client.get_sample())
            )
            reader.start()
            client.stop()
            reader.join(timeout=2)
            assert not reader.is_alive()
            assert reader_result == [None]
        finally:
            release.set()
            client.stop()


def test_failed_shard_is_not_an_empty_dataset(monkeypatch):
    monkeypatch.setenv("DATAGO_MAX_RETRIES", "1")
    with serve(lambda _: (404, b"not found")) as url:
        client = DatagoClient(
            json.dumps(
                {
                    "source_type": "webdataset",
                    "source_config": {
                        "url": url + "/shard-{0..0}.tar",
                        "concurrent_downloads": 1,
                    },
                    "limit": 10,
                    "samples_buffer_size": 2,
                }
            )
        )
        try:
            with pytest.raises(RuntimeError, match="source failed"):
                client.get_sample(timeout=3)
        finally:
            client.stop()


def test_successful_shard_limit_does_not_turn_feeder_shutdown_into_error(monkeypatch):
    monkeypatch.setenv("DATAGO_MAX_RETRIES", "1")
    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="PNG")
    image = output.getvalue()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for index in range(100):
            info = tarfile.TarInfo(f"{index}.png")
            info.size = len(image)
            tar.addfile(info, io.BytesIO(image))
    with serve(lambda _: (200, archive.getvalue())) as url:
        client = DatagoClient(
            json.dumps(
                {
                    "source_type": "webdataset",
                    "source_config": {
                        "url": url + "/shard-{0..0}.tar",
                        "reference_image_type": "png",
                        "concurrent_downloads": 1,
                    },
                    "limit": 1,
                    "samples_buffer_size": 1,
                }
            )
        )
        try:
            # Legacy workers can overshoot limits; this PR preserves that policy.
            assert client.get_sample(timeout=3) is not None
            while client.get_sample(timeout=3) is not None:
                pass
        finally:
            client.stop()


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_read_timeout_is_a_value_error(tmp_path, timeout):
    client = DatagoClient(
        json.dumps(
            {
                "source_type": "file",
                "source_config": {"root_path": str(tmp_path)},
                "limit": 1,
                "samples_buffer_size": 1,
            }
        )
    )
    try:
        with pytest.raises(ValueError, match="timeout"):
            client.get_sample(timeout=timeout)
    finally:
        client.stop()


@pytest.mark.parametrize(
    "body,failed",
    [
        (b'{"kind":"storage#objects"}', False),
        (b'{"items":[]}', False),
        (b"{}", True),
        (b'{"items":{}}', True),
    ],
)
def test_shard_listing_distinguishes_empty_from_invalid(body, failed):
    with serve(lambda _: (200, body)) as url:
        client = DatagoClient(
            json.dumps(
                {
                    "source_type": "webdataset",
                    "source_config": {"url": url + "/listing"},
                    "limit": 1,
                    "samples_buffer_size": 1,
                }
            )
        )
        try:
            if failed:
                with pytest.raises(RuntimeError, match="source failed"):
                    client.get_sample(timeout=3)
            else:
                assert client.get_sample(timeout=3) is None
        finally:
            client.stop()
