"""Image metadata and ndarray views should avoid copying pixel storage."""

import json
import os

import numpy as np
from datago import DatagoClient
from PIL import Image


def client_for(directory, image: Image.Image) -> DatagoClient:
    directory.mkdir(parents=True, exist_ok=True)
    image.save(os.path.join(directory, "sample.png"))
    config = {
        "source_type": "file",
        "source_config": {"root_path": str(directory), "random_sampling": False},
        "limit": 1,
        "samples_buffer_size": 1,
    }
    return DatagoClient(json.dumps(config))


def test_raw_numpy_array_is_a_readonly_zero_copy_view_with_independent_lifetime(
    tmp_path, monkeypatch
):
    pixels = np.arange(12 * 9 * 3, dtype=np.uint8).reshape(9, 12, 3)
    client = client_for(tmp_path, Image.fromarray(pixels))
    sample = client.get_sample()
    assert sample is not None
    wrapper = sample.image

    # Metadata hot paths do not decode/materialize a PIL image.
    def unexpected(*args, **kwargs):
        raise AssertionError("raw image metadata should not invoke PIL")

    monkeypatch.setattr(Image, "open", unexpected)
    assert wrapper.width == 12
    assert wrapper.height == 9
    assert wrapper.size == (12, 9)
    assert wrapper.mode == "RGB"
    assert wrapper.palette is None

    view = memoryview(wrapper)
    assert view.readonly
    array = wrapper.to_numpy_array()
    alias = np.frombuffer(view, dtype=np.uint8).reshape(9, 12, 3)
    assert not array.flags.owndata
    assert not array.flags.writeable
    assert array.__array_interface__["data"][0] == alias.__array_interface__["data"][0]
    np.testing.assert_array_equal(array, pixels)
    payload_array = sample.image.get_payload().to_numpy_array()
    assert (
        payload_array.__array_interface__["data"][0]
        == array.__array_interface__["data"][0]
    )

    # A view retains the exporter and Rust allocation after sample/client drop.
    del wrapper, sample
    client.stop()
    del view, alias
    np.testing.assert_array_equal(array, pixels)
    np.testing.assert_array_equal(payload_array, pixels)
    del array, payload_array


def test_image_payload_clone_shares_storage_but_explicit_bytes_are_owned(tmp_path):
    client = client_for(tmp_path, Image.new("RGB", (5, 4), (11, 22, 33)))
    sample = client.get_sample()
    assert sample is not None
    payload = sample.image.get_payload()
    clone = sample.image.get_payload()
    assert payload.data == clone.data
    assert isinstance(payload.data, bytes)
    original = bytes(payload.data)
    payload.data = bytes([0] * len(payload.data))
    assert bytes(clone.data) == original
    client.stop()
