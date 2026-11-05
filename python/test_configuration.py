"""Configuration failures retain actionable field names at the public API."""

import json
import subprocess
import sys

import pytest
from datago import DatagoClient


@pytest.mark.parametrize("method", ["start", "get_sample", "get_sample_auto_convert"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("image_format", "tensor"),
        ("image_format", 42),
        ("samples_buffer_size", 0),
        ("limit", 0),
    ],
)
def test_invalid_config_names_the_field(tmp_path, method, field, value):
    config = {
        "source_type": "file",
        "source_config": {"root_path": str(tmp_path)},
        "limit": 1,
        "samples_buffer_size": 1,
        field: value,
    }
    client = DatagoClient(json.dumps(config))
    try:
        with pytest.raises(ValueError, match=field):
            getattr(client, method)()
    finally:
        client.stop()


def test_unknown_client_fields_are_logged_and_remain_compatible(tmp_path):
    config = {
        "source_type": "file",
        "source_config": {"root_path": str(tmp_path)},
        "limit": 1,
        "samples_buffer_size": 1,
        "unused_option": True,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from datago import DatagoClient, initialize_logging; "
                "initialize_logging('warn'); client = DatagoClient(sys.argv[1]); "
                "client.start(); assert client.get_sample() is None; client.stop()"
            ),
            json.dumps(config),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (
        'Unknown client configuration field "unused_option" is ignored' in result.stderr
    )
