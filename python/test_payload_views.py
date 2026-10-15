"""Image metadata and ndarray views should avoid copying pixel storage."""

import json
import os

import numpy as np
import pytest
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


def test_buffer_view_outlives_data_reassignment(tmp_path):
    """A view on the mutable payload wrapper must keep the old pixels alive when
    `data` is reassigned (this used to be a use-after-free)."""
    pixels = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    client = client_for(tmp_path, Image.fromarray(pixels))
    sample = client.get_sample()
    assert sample is not None

    payload = sample.image.get_payload()
    view = memoryview(payload)
    expected = bytes(view)

    del sample
    client.stop()

    payload.data = bytes(len(expected))  # replaces the wrapper's storage
    assert bytes(view) == expected


def test_to_pil_image_matches_pixels(tmp_path):
    pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    client = client_for(tmp_path, Image.fromarray(pixels))
    sample = client.get_sample()
    assert sample is not None

    np.testing.assert_array_equal(np.asarray(sample.image.to_pil_image()), pixels)
    # __call__ and attribute delegation share the same conversion path.
    np.testing.assert_array_equal(np.asarray(sample.image()), pixels)
    client.stop()


def test_to_pil_image_grayscale(tmp_path):
    pixels = (np.arange(4 * 6, dtype=np.uint8) % 250).reshape(4, 6)
    client = client_for(tmp_path, Image.fromarray(pixels, mode="L"))
    sample = client.get_sample()
    assert sample is not None

    pil = sample.image.to_pil_image()
    assert pil.mode == "L"
    np.testing.assert_array_equal(np.asarray(pil), pixels)
    client.stop()


def test_encoded_to_numpy_array_decodes(tmp_path):
    """Encoded payloads must decode to a numpy array (this path used to raise)."""
    directory = tmp_path / "encoded"
    directory.mkdir()
    Image.new("RGB", (9, 5), (40, 80, 120)).save(directory / "a.png")
    client_config = {
        "source_type": "file",
        "source_config": {"root_path": str(directory)},
        "limit": 1,
        "samples_buffer_size": 1,
        "image_config": {
            "crop_and_resize": True,
            "default_image_size": 1024,
            "downsampling_ratio": 32,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
            "pre_encode_images": True,
        },
    }
    client = DatagoClient(json.dumps(client_config))
    sample = client.get_sample()
    assert sample is not None
    payload = sample.image.get_payload()
    assert payload.channels == -1  # encoded
    array = sample.image.to_numpy_array()
    assert array.ndim == 3
    assert array.shape[2] == 3
    # Documented contract: for encoded payloads the buffer protocol exposes the
    # compressed stream, not pixels; only to_numpy_array() decodes.
    assert memoryview(sample.image).nbytes == len(payload.data)
    assert memoryview(sample.image).nbytes != array.size
    client.stop()


def _file_client(directory, image, **extra):
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / "sample.png")
    config = {
        "source_type": "file",
        "source_config": {"root_path": str(directory)},
        "limit": 1,
        "samples_buffer_size": 1,
        **extra,
    }
    return DatagoClient(json.dumps(config))


def _write_rgb16_png(path, pixels):
    """Write a 16-bit RGB PNG (Pillow has no RGB16 mode to save one)."""
    import binascii
    import struct
    import zlib

    height, width, _ = pixels.shape

    def chunk(tag, data):
        crc = binascii.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + pixels[y].astype(">u2").tobytes() for y in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_image_format_numpy_returns_zero_copy_ndarray(tmp_path):
    pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    client = _file_client(
        tmp_path / "np", Image.fromarray(pixels), image_format="numpy"
    )
    sample = client.get_sample_auto_convert()
    assert sample is not None
    image = sample["image"]
    assert isinstance(image, np.ndarray)
    assert not image.flags.owndata
    assert not image.flags.writeable
    np.testing.assert_array_equal(image, pixels)
    client.stop()


def test_grayscale_numpy_matches_pil_shape(tmp_path):
    pixels = (np.arange(4 * 6, dtype=np.uint8) % 250).reshape(4, 6)
    client = _file_client(
        tmp_path / "gray_np", Image.fromarray(pixels, mode="L"), image_format="numpy"
    )
    sample = client.get_sample_auto_convert()
    assert sample is not None
    image = sample["image"]
    assert image.shape == (4, 6)  # same as np.asarray(PIL "L"), not (4, 6, 1)
    np.testing.assert_array_equal(image, pixels)
    client.stop()


def test_image_format_defaults_to_pil(tmp_path):
    client = _file_client(tmp_path / "pil_default", Image.new("RGB", (4, 3), (1, 2, 3)))
    sample = client.get_sample_auto_convert()
    assert sample is not None
    assert isinstance(sample["image"], Image.Image)
    client.stop()


def test_image_format_pil_alias(tmp_path):
    client = _file_client(
        tmp_path / "pil_alias",
        Image.new("RGB", (4, 3), (1, 2, 3)),
        image_format="PIL",
    )
    sample = client.get_sample_auto_convert()
    assert sample is not None
    assert isinstance(sample["image"], Image.Image)
    client.stop()


def test_numpy_preserves_16bit_grayscale(tmp_path):
    pixels = ((np.arange(3 * 4, dtype=np.uint16) * 5000) % 65535).reshape(3, 4)
    client = _file_client(
        tmp_path / "g16", Image.fromarray(pixels), image_format="numpy"
    )
    sample = client.get_sample_auto_convert()
    assert sample is not None
    image = sample["image"]
    assert image.dtype == np.uint16
    assert image.shape == (3, 4)
    np.testing.assert_array_equal(image, pixels)
    client.stop()


def test_numpy_preserves_16bit_rgb(tmp_path):
    pixels = ((np.arange(3 * 4 * 3, dtype=np.uint16) * 5000) % 65535).reshape(3, 4, 3)
    directory = tmp_path / "rgb16"
    directory.mkdir()
    _write_rgb16_png(directory / "rgb.png", pixels)
    client = DatagoClient(
        json.dumps(
            {
                "source_type": "file",
                "source_config": {"root_path": str(directory)},
                "limit": 1,
                "samples_buffer_size": 1,
                "image_format": "numpy",
            }
        )
    )
    sample = client.get_sample_auto_convert()
    assert sample is not None
    image = sample["image"]
    assert image.dtype == np.uint16
    assert image.shape == (3, 4, 3)
    np.testing.assert_array_equal(image, pixels)
    client.stop()


def test_pil_16bit_grayscale_uses_i16(tmp_path):
    pixels = ((np.arange(3 * 4, dtype=np.uint16) * 5000) % 65535).reshape(3, 4)
    client = _file_client(tmp_path / "g16_pil", Image.fromarray(pixels))
    sample = client.get_sample()
    assert sample is not None
    pil = sample.image.to_pil_image()
    assert pil.mode == "I;16"
    np.testing.assert_array_equal(np.asarray(pil), pixels)
    client.stop()


def test_pil_16bit_rgb_raises(tmp_path):
    pixels = ((np.arange(3 * 4 * 3, dtype=np.uint16) * 5000) % 65535).reshape(3, 4, 3)
    directory = tmp_path / "rgb16_pil"
    directory.mkdir()
    _write_rgb16_png(directory / "rgb.png", pixels)
    client = DatagoClient(
        json.dumps(
            {
                "source_type": "file",
                "source_config": {"root_path": str(directory)},
                "limit": 1,
                "samples_buffer_size": 1,
            }
        )
    )
    sample = client.get_sample()
    assert sample is not None
    with pytest.raises(ValueError):
        sample.image.to_pil_image()
    client.stop()
