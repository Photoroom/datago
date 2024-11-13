from datago import datago
import pytest
import os
import json
from go_types import go_array_to_pil_image, go_array_to_numpy


def get_test_source() -> str:
    test_source = os.getenv("DATAROOM_TEST_SOURCE", "COYO")
    assert test_source is not None, "Please set DATAROOM_TEST_SOURCE"
    return test_source


def get_json_config():
    client_config = {
        "source_type": datago.SourceTypeDB,
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
        "samples_buffer_size": 128,
        "concurrency": 1,
        "limit": 10,
    }
    return client_config


def get_dataset(client_config: str):
    client = datago.GetClientFromJSON(json.dumps(client_config))

    class Dataset:
        def __init__(self, client):
            self.client = client

        def __iter__(self):
            return self

        def __next__(self):
            new_sample = self.client.GetSample()
            if new_sample.ID == "":
                raise StopIteration
            return new_sample

    return Dataset(client)


def test_get_sample_db():
    # Check that we can instantiate a client and get a sample, nothing more
    client_config = datago.GetDatagoConfig()
    client_config.SamplesBufferSize = 10

    source_config = datago.GetSourceDBConfig()
    source_config.Sources = get_test_source()
    client_config.SourceConfig = source_config

    client = datago.GetClient(client_config)
    data = client.GetSample()
    assert data.ID != ""


N_SAMPLES = 3


def test_caption_and_image():
    client_config = get_json_config()
    dataset = get_dataset(client_config)

    def check_image(img, channels=3):
        assert img.Height > 0
        assert img.Width > 0

        assert img.Height <= img.OriginalHeight
        assert img.Width <= img.OriginalWidth
        assert img.Channels == channels

    for i, sample in enumerate(dataset):
        assert sample.Source != ""
        assert sample.ID != ""

        assert len(sample.Attributes["caption_coca"]) != len(
            sample.Attributes["caption_cogvlm"]
        ), "Caption lengths should not be equal"

        check_image(sample.Image, 3)
        check_image(sample.AdditionalImages["masked_image"], 3)
        check_image(sample.Masks["segmentation_mask"], 1)

        # Check the image decoding
        assert go_array_to_pil_image(sample.Image).mode == "RGB", "Image should be RGB"
        assert (
            go_array_to_pil_image(sample.AdditionalImages["masked_image"]).mode == "RGB"
        ), "Image should be RGB"
        assert (
            go_array_to_pil_image(sample.Masks["segmentation_mask"]).mode == "L"
        ), "Mask should be L"

        if i > N_SAMPLES:
            break


def test_image_resize():
    client_config = get_json_config()
    client_config["image_config"]["crop_and_resize"] = True
    dataset = get_dataset(client_config)

    for i, sample in enumerate(dataset):
        # Assert that all the images in the sample have the same size
        assert (
            sample.Image.Height
            == sample.AdditionalImages["masked_image"].Height
            == sample.Masks["segmentation_mask"].Height
            and sample.Image.Height > 0
        )
        assert (
            sample.Image.Width
            == sample.AdditionalImages["masked_image"].Width
            == sample.Masks["segmentation_mask"].Width
            and sample.Image.Width > 0
        )
        if i > N_SAMPLES:
            break


def test_has_tags():
    client_config = get_json_config()
    client_config["source_config"]["tags"] = "v4_trainset_hq"

    dataset = get_dataset(client_config)
    sample = next(iter(dataset))

    assert "v4_trainset_hq" in sample.Tags, "v4_trainset_hq should be in the tags"


def no_test_jpg_compression():
    # Check that the images are compressed as expected
    client_config = get_json_config()
    client_config["image_config"]["pre_encode_images"] = True
    dataset = get_dataset(client_config)

    sample = next(iter(dataset))

    assert go_array_to_pil_image(sample.Image).mode == "RGB", "Image should be RGB"
    assert (
        go_array_to_pil_image(sample.AdditionalImages["masked_image"]).mode == "RGB"
    ), "Image should be RGB"
    assert (
        go_array_to_pil_image(sample.Masks["segmentation_mask"]).mode == "L"
    ), "Mask should be L"

    # Check the embeddings decoding
    assert (
        go_array_to_numpy(sample.CocaEmbedding) is not None
    ), "Embedding should be set"


def test_duplicate_state():
    client_config = get_json_config()
    client_config["source_config"]["return_duplicate_state"] = True
    dataset = get_dataset(client_config)

    sample = next(iter(dataset))
    assert sample.DuplicateState in [
        0,
        1,
        2,
    ], "Duplicate state should be 0, 1 or 2"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
