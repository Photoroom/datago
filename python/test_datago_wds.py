"""
Test suite for WebDataset (WDS) functionality in Datago.

This module tests that Datago correctly serves images and attributes from WebDataset sources.
"""

import os

from dataset import DatagoIterDataset
from PIL import Image

# Test buckets - using the same ones as benchmark_webdataset.py
TEST_BUCKETS = {
    "pd12m": {
        "url": "https://huggingface.co/datasets/sayakpaul/pd12m-full/resolve/main/{00155..02480}.tar",
        "source": "PD12M",
    },
    "fakein": {
        "url": "https://storage.googleapis.com/webdataset/fake-imagenet/imagenet-train-{000000..001281}.tar",
        "source": "FakeIN",
    },
}


def test_wds_basic_functionality():
    """Test basic WDS functionality - that we can get samples with proper structure."""
    limit = 5  # Small limit for quick testing

    # Use the PD12M bucket for testing
    bucket_config = TEST_BUCKETS["pd12m"]

    client_config = {
        "source_type": "webdataset",
        "source_config": {
            "url": bucket_config["url"],
            "shuffle": True,
            "concurrent_downloads": 4,  # Reduced for testing
            "auth_token": os.environ.get("HF_TOKEN", default=""),
        },
        "prefetch_buffer_size": 32,
        "samples_buffer_size": 32,
        "limit": limit,
    }

    # Test with return_python_types=True to get proper Python objects
    dataset = DatagoIterDataset(client_config, return_python_types=True)

    count = 0
    for sample in dataset:
        count += 1

        # Basic structure checks
        assert "id" in sample, "Sample should contain 'id' field"
        assert sample["id"] != "", "Sample ID should not be empty"

        # Check that we have an image
        assert "image" in sample, "Sample should contain 'image' field"
        assert sample["image"] is not None, "Image should not be None"

        # If it's a PIL Image, check its properties
        if isinstance(sample["image"], Image.Image):
            assert sample["image"].width > 0, "Image should have positive width"
            assert sample["image"].height > 0, "Image should have positive height"
            assert sample["image"].mode in ["RGB", "RGBA", "L"], (
                f"Image should have valid mode, got {sample['image'].mode}"
            )

        # Check for attributes if present
        if "attributes" in sample:
            assert isinstance(sample["attributes"], dict), "Attributes should be a dictionary"
            # Attributes should be non-empty if present
            if sample["attributes"]:
                assert len(sample["attributes"]) > 0, "Attributes dictionary should not be empty"

        # We should get at least the basic fields
        assert len(sample) >= 2, "Sample should contain at least id and image"

        if count >= limit:
            break

    assert count == limit, f"Expected {limit} samples, got {count}"


def test_wds_image_properties():
    """Test that images from WDS have proper properties and can be processed."""
    limit = 3

    bucket_config = TEST_BUCKETS["pd12m"]

    client_config = {
        "source_type": "webdataset",
        "source_config": {
            "url": bucket_config["url"],
            "shuffle": True,
            "concurrent_downloads": 4,
            "auth_token": os.environ.get("HF_TOKEN", default=""),
        },
        "prefetch_buffer_size": 32,
        "samples_buffer_size": 32,
        "limit": limit,
    }

    dataset = DatagoIterDataset(client_config, return_python_types=True)

    for sample in dataset:
        if "image" in sample and sample["image"] is not None:
            image = sample["image"]

            # Test that we can get image properties
            if isinstance(image, Image.Image):
                width, height = image.size
                assert width > 0 and height > 0, "Image should have valid dimensions"

                # Test that we can convert to different modes
                rgb_image = image.convert("RGB")
                assert rgb_image.mode == "RGB", "Image should convert to RGB mode"

                # Test that we can get thumbnail
                thumbnail = image.copy()
                thumbnail.thumbnail((100, 100))
                assert thumbnail.size[0] <= 100 and thumbnail.size[1] <= 100, "Thumbnail should be resized"

                # Test that image data is valid by trying to get pixel data
                pixels = image.get_flattened_data()
                assert len(pixels) > 0, "Image should have pixel data"

                break  # Just test one image


def test_wds_with_image_processing():
    """Test WDS with image processing configuration (crop and resize)."""
    limit = 3

    bucket_config = TEST_BUCKETS["pd12m"]

    client_config = {
        "source_type": "webdataset",
        "source_config": {
            "url": bucket_config["url"],
            "shuffle": True,
            "concurrent_downloads": 4,
            "auth_token": os.environ.get("HF_TOKEN", default=""),
        },
        "image_config": {
            "crop_and_resize": True,
            "default_image_size": 256,
            "downsampling_ratio": 16,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
        },
        "prefetch_buffer_size": 32,
        "samples_buffer_size": 32,
        "limit": limit,
    }

    dataset = DatagoIterDataset(client_config, return_python_types=True)

    for sample in dataset:
        if "image" in sample and sample["image"] is not None:
            image = sample["image"]

            if isinstance(image, Image.Image):
                # With crop_and_resize=True, images should be processed
                width, height = image.size
                assert width > 0 and height > 0, "Processed image should have valid dimensions"

                # The processed image should be in RGB mode
                assert image.mode == "RGB", f"Processed image should be RGB, got {image.mode}"

                break  # Just test one image


def test_wds_attributes_structure():
    """Test that WDS attributes are properly structured when present."""
    limit = 5

    bucket_config = TEST_BUCKETS["pd12m"]

    client_config = {
        "source_type": "webdataset",
        "source_config": {
            "url": bucket_config["url"],
            "shuffle": True,
            "concurrent_downloads": 4,
            "auth_token": os.environ.get("HF_TOKEN", default=""),
        },
        "prefetch_buffer_size": 32,
        "samples_buffer_size": 32,
        "limit": limit,
    }

    dataset = DatagoIterDataset(client_config, return_python_types=True)

    for sample in dataset:
        if "attributes" in sample and sample["attributes"]:
            attributes = sample["attributes"]

            # Attributes should be a dictionary
            assert isinstance(attributes, dict), "Attributes should be a dictionary"

            # Check that we can access attribute values
            for key, value in attributes.items():
                # Values should be JSON-serializable types
                assert isinstance(key, str), "Attribute keys should be strings"
                assert isinstance(value, (str, int, float, bool, list, dict)), (
                    f"Attribute values should be JSON-serializable, got {type(value)}"
                )

            break  # Just test one sample with attributes

    # Note: Not all samples may have attributes


def test_wds_sample_consistency():
    """Test that WDS samples have consistent structure across multiple samples."""
    limit = 10

    bucket_config = TEST_BUCKETS["pd12m"]

    client_config = {
        "source_type": "webdataset",
        "source_config": {
            "url": bucket_config["url"],
            "shuffle": True,
            "concurrent_downloads": 4,
            "auth_token": os.environ.get("HF_TOKEN", default=""),
        },
        "prefetch_buffer_size": 32,
        "samples_buffer_size": 32,
        "limit": limit,
    }

    dataset = DatagoIterDataset(client_config, return_python_types=True)

    first_sample = True

    for sample in dataset:
        current_keys = set(sample.keys())

        if first_sample:
            first_sample = False
        else:
            # All samples should have at least the core fields (id, image)
            required_keys = {"id", "image"}
            assert required_keys.issubset(current_keys), \
                f"Sample missing required keys. Expected at least {required_keys}, got {current_keys}"

        # Check that we don't have any unexpected None values for core fields
        assert sample.get("id") != "", "Sample ID should not be empty"
        assert sample.get("image") is not None, "Sample image should not be None"
