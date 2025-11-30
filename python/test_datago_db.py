from datago import DatagoClient
import pytest
import os
import json

from raw_types import (
    raw_array_to_pil_image,
    raw_array_to_numpy,
    get_image_mode,
    decode_image_payload,
)
from dataset import DatagoIterDataset


def get_test_source() -> str:
    test_source = os.getenv("DATAROOM_TEST_SOURCE", "COYO")
    assert test_source is not None, "Please set DATAROOM_TEST_SOURCE"
    return test_source


def get_json_config():
    client_config = {
        "source_type": "db",
        "source_config": {
            "page_size": 10,
            "sources": get_test_source(),
            "require_images": True,
            "has_masks": "segmentation_mask",
            "has_latents": "masked_image",
            "has_attributes": "caption_coca,caption_cogvlm,caption_moondream",
            "return_latents": "masked_image",
            "rank": 0,
            "world_size": 1,
        },
        "image_config": {
            "crop_and_resize": False,
            "default_image_size": 512,
            "downsampling_ratio": 16,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
            "pre_encode_images": False,
        },
        "prefetch_buffer_size": 64,
        "samples_buffer_size": 10,
        "limit": 10,
    }
    return client_config


def test_get_sample_db():
    # Check that we can instantiate a client and get a sample, nothing more
    client_config = get_json_config()

    client = DatagoClient(json.dumps(client_config))
    data = client.get_sample()
    assert data.id != ""


N_SAMPLES = 3


def test_caption_and_image():
    client_config = get_json_config()
    dataset = DatagoIterDataset(client_config, return_python_types=False)

    def check_image(img, channels=3):
        assert img.height > 0
        assert img.width > 0

        payload = img.get_payload()
        assert img.height <= payload.original_height
        assert img.width <= payload.original_width
        assert img.channels == channels

    for i, sample in enumerate(dataset):
        assert sample.source != ""
        assert sample.id != ""

        attributes = json.loads(sample.attributes)
        assert len(attributes["caption_coca"]) != len(attributes["caption_cogvlm"]), (
            "Caption lengths should not be equal"
        )

        check_image(sample.image, 3)
        check_image(sample.additional_images["masked_image"], 3)
        check_image(sample.masks["segmentation_mask"], 1)

        # Check the image decoding
        assert raw_array_to_pil_image(sample.image).mode == "RGB", "Image should be RGB"
        assert (
            raw_array_to_pil_image(sample.additional_images["masked_image"]).mode
            == "RGB"
        ), "Image should be RGB"
        assert raw_array_to_pil_image(sample.masks["segmentation_mask"]).mode == "L", (
            "Mask should be L"
        )

        if i > N_SAMPLES:
            break


def test_image_resize():
    client_config = get_json_config()
    client_config["image_config"]["crop_and_resize"] = True
    dataset = DatagoIterDataset(client_config, return_python_types=False)

    for i, sample in enumerate(dataset):
        # Assert that all the images in the sample have the same size
        assert (
            sample.image.height
            == sample.additional_images["masked_image"].height
            == sample.masks["segmentation_mask"].height
            and sample.image.height > 0
        )
        assert (
            sample.image.width
            == sample.additional_images["masked_image"].width
            == sample.masks["segmentation_mask"].width
            and sample.image.width > 0
        )
        if i > N_SAMPLES:
            break


def test_has_tags():
    client_config = get_json_config()
    client_config["source_config"]["tags"] = "v4_trainset_hq"

    dataset = DatagoIterDataset(client_config, return_python_types=False)
    sample = next(iter(dataset))

    assert "v4_trainset_hq" in sample.tags, "v4_trainset_hq should be in the tags"


def test_empty_image():
    client_config = get_json_config()
    client_config["source_config"]["require_images"] = False

    dataset = DatagoIterDataset(client_config, return_python_types=False)

    # Just check that accessing the sample in python does not crash
    _ = next(iter(dataset))


def test_jpeg_compression():
    # Check that JPEG compression works as expected
    client_config = get_json_config()
    client_config["image_config"]["pre_encode_images"] = True
    client_config["image_config"]["encode_format"] = "jpeg"
    client_config["image_config"]["jpeg_quality"] = 92
    dataset = DatagoIterDataset(client_config, return_python_types=False)

    sample = next(iter(dataset))

    # When images are encoded, channels is set to -1 to signal encoded format
    assert sample.image.channels == -1, "Image should be encoded (channels == -1)"
    assert sample.additional_images["masked_image"].channels == -1, (
        "Additional image should be encoded"
    )
    assert sample.masks["segmentation_mask"].channels == -1, "Mask should be encoded"

    # Test that raw_array_to_pil_image returns ImagePayload for encoded images
    image_result = raw_array_to_pil_image(sample.image)
    assert not hasattr(image_result, "mode"), (
        "Should return ImagePayload, not PIL Image"
    )
    assert hasattr(image_result, "data"), "Should have data attribute"
    assert hasattr(image_result, "channels"), "Should have channels attribute"
    assert image_result.channels == -1, "Should be encoded ImagePayload"

    # Test proper decoding using decode_image_payload
    decoded_image = decode_image_payload(image_result)
    assert hasattr(decoded_image, "mode"), "Decoded image should be PIL Image"
    assert decoded_image.mode == "RGB", "Image should decode to RGB"
    assert decoded_image.size == (sample.image.width, sample.image.height), (
        "Size should match"
    )

    # Test additional images and masks
    additional_result = raw_array_to_pil_image(sample.additional_images["masked_image"])
    decoded_additional = decode_image_payload(additional_result)
    assert decoded_additional.mode == "RGB", "Additional image should decode to RGB"

    mask_result = raw_array_to_pil_image(sample.masks["segmentation_mask"])
    decoded_mask = decode_image_payload(mask_result)
    assert decoded_mask.mode == "L", "Mask should decode to L"


def test_png_compression():
    # Check that PNG compression works (default behavior)
    client_config = get_json_config()
    client_config["image_config"]["pre_encode_images"] = True
    # Don't specify encode_format - should default to PNG
    dataset = DatagoIterDataset(client_config, return_python_types=False)

    sample = next(iter(dataset))

    # When images are encoded, channels is set to -1 to signal encoded format
    assert sample.image.channels == -1, "Image should be encoded (channels == -1)"

    # Test that raw_array_to_pil_image returns ImagePayload for encoded images
    image_result = raw_array_to_pil_image(sample.image)
    assert not hasattr(image_result, "mode"), (
        "Should return ImagePayload, not PIL Image"
    )
    assert hasattr(image_result, "data"), "Should have data attribute"
    assert hasattr(image_result, "channels"), "Should have channels attribute"
    assert image_result.channels == -1, "Should be encoded ImagePayload"

    # Test proper decoding using decode_image_payload
    decoded_image = decode_image_payload(image_result)
    assert hasattr(decoded_image, "mode"), "Decoded image should be PIL Image"
    assert decoded_image.mode == "RGB", "Image should decode to RGB"
    assert decoded_image.size == (sample.image.width, sample.image.height), (
        "Size should match"
    )


def test_original_image():
    # Check that the images are transmitted as expected
    client_config = get_json_config()
    client_config["image_config"]["pre_encode_images"] = False
    client_config["image_config"]["crop_and_resize"] = False
    dataset = DatagoIterDataset(client_config, return_python_types=False)

    sample = next(iter(dataset))

    assert raw_array_to_pil_image(sample.image).mode == "RGB", "Image should be RGB"
    assert (
        raw_array_to_pil_image(sample.additional_images["masked_image"]).mode == "RGB"
    ), "Image should be RGB"
    assert raw_array_to_pil_image(sample.masks["segmentation_mask"]).mode == "L", (
        "Mask should be L"
    )


def test_duplicate_state():
    client_config = get_json_config()
    client_config["source_config"]["return_duplicate_state"] = True
    dataset = DatagoIterDataset(client_config, return_python_types=False)

    sample = next(iter(dataset))
    assert sample.duplicate_state in [
        0,
        1,
        2,
    ], "Duplicate state should be 0, 1 or 2"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
