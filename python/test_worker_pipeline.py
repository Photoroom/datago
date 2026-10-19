"""Public client guarantees shared by the file, HTTP, and WDS coordinators."""

import io
import json
import os
import subprocess
import sys
import tarfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from datago import DatagoClient
from PIL import Image


@pytest.mark.parametrize("tasks", ["0", "1", "4"])
def test_raw_client_enforces_exact_limit_with_failed_items(
    tmp_path, monkeypatch, tasks
):
    monkeypatch.setenv("DATAGO_MAX_TASKS", tasks)
    (tmp_path / "broken.png").write_bytes(b"not an image")
    for i in range(30):
        Image.new("RGB", (3, 2)).save(tmp_path / f"{i}.png")
    client = DatagoClient(
        json.dumps(
            {
                "source_type": "file",
                "source_config": {"root_path": str(tmp_path)},
                "limit": 10,
                "samples_buffer_size": 2,
            }
        )
    )
    try:
        ids = []
        while (sample := client.get_sample()) is not None:
            ids.append(sample.id)
        assert len(ids) == len(set(ids)) == 10
        assert client.get_sample() is None
    finally:
        client.stop()


def test_stop_cancels_http_download_after_metadata_ends():
    script = textwrap.dedent("""
        import io, json, os, threading, time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from PIL import Image
        from datago import DatagoClient, live_cleanup_threads

        started = threading.Event()
        release = threading.Event()
        output = io.BytesIO()
        Image.new("RGB", (2, 2)).save(output, format="PNG")
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/image.png":
                    started.set()
                    release.wait(5)
                    body = output.getvalue()
                else:
                    body = json.dumps({"results": [{
                        "id": "one", "source": "mock", "attributes": {},
                        "image_direct_url": f"http://127.0.0.1:{self.server.server_port}/image.png",
                    }], "next": None}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        os.environ["DATAROOM_API_URL"] = f"http://127.0.0.1:{server.server_port}/"
        os.environ["DATAROOM_API_KEY"] = "test"
        os.environ["DATAGO_MAX_TASKS"] = "4"
        client = DatagoClient(json.dumps({
            "source_type": "db", "source_config": {"sources": "mock", "page_size": 10},
            "limit": 10, "samples_buffer_size": 2,
        }))
        try:
            client.start()
            assert started.wait(5)
            # One item, then EOS: the coordinator is waiting for that download.
            time.sleep(0.1)
            client.stop()
            deadline = time.monotonic() + 1
            while live_cleanup_threads() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert live_cleanup_threads() == 0, "download survived cancellation"
        finally:
            release.set()
            client.stop()
            server.shutdown()
            server.server_close()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", ["db", "webdataset"])
def test_network_sources_enforce_exact_limit(source, monkeypatch):
    """Exercise the real source adapters without credentials or external data."""
    image = io.BytesIO()
    Image.new("RGB", (3, 2), (40, 80, 120)).save(image, format="PNG")
    pixels = image.getvalue()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for i in range(20):
            info = tarfile.TarInfo(f"{i:04}.png")
            info.size = len(pixels)
            tar.addfile(info, io.BytesIO(pixels))
    tar_bytes = archive.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/images/"):
                body = json.dumps(
                    {
                        "results": [
                            {
                                "id": str(i),
                                "source": "mock",
                                "attributes": {},
                                "image_direct_url": base_url + "image.png",
                            }
                            for i in range(20)
                        ],
                        "next": None,
                    }
                ).encode()
            elif self.path.endswith(".tar"):
                body = tar_bytes
            else:
                body = pixels
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base_url = f"http://127.0.0.1:{server.server_port}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("DATAROOM_API_URL", base_url)
    monkeypatch.setenv("DATAROOM_API_KEY", "test")
    monkeypatch.setenv("DATAGO_MAX_TASKS", "4")
    source_config = (
        {"sources": "mock", "page_size": 20}
        if source == "db"
        else {
            "url": base_url + "shard-{0..0}.tar",
            "concurrent_downloads": 1,
            "reference_image_type": "png",
        }
    )
    client = DatagoClient(
        json.dumps(
            {
                "source_type": source,
                "source_config": source_config,
                "limit": 10,
                "samples_buffer_size": 2,
                "image_format": "numpy",
            }
        )
    )
    try:
        samples = []
        while (sample := client.get_sample_auto_convert()) is not None:
            samples.append(sample)
        assert len(samples) == len({sample["id"] for sample in samples}) == 10
        assert all(sample["image"].shape == (2, 3, 3) for sample in samples)
    finally:
        client.stop()
        server.shutdown()
        server.server_close()
