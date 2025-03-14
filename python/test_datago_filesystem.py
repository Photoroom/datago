from PIL import Image
from datago import DatagoClient
import json
import tempfile
import pytest
import random


@pytest.mark.parametrize("pre_encode_images", [False, True])
def test_get_sample_filesystem(pre_encode_images: bool):
    limit = 10

    with tempfile.TemporaryDirectory() as tmpdirname:
        cwd = tmpdirname

        for i in range(limit):
            # Prepare an ephemeral test set
            mode = "RGB" if random.random() > 0.5 else "RGBA"
            img = Image.new(mode, (100, 100))

            # Randomly make the image 16 bits
            if random.random() > 0.5:
                img = img.convert("I;16")

            img.save(cwd + f"/test_{i}.png")

        # Check that we can instantiate a client and get a sample, nothing more
        client_config = {
            "source_type": "file",
            "source_config": {
                "root_path": cwd,
            },
            "image_config": {
                "crop_and_resize": False,
                "default_image_size": 512,
                "downsampling_ratio": 16,
                "min_aspect_ratio": 0.5,
                "max_aspect_ratio": 2.0,
                "pre_encode_images": pre_encode_images,
            },
            "limit": limit,
            "prefetch_buffer_size": 64,
            "samples_buffer_size": 10,
            "rank": 0,
            "world_size": 1,
        }

        client = DatagoClient(json.dumps(client_config))
        count = 0
        for i in range(limit):
            data = client.get_sample()
            if not data:
                break
            count += 1
            assert data.id != ""
            assert data.image.width == 100
            assert data.image.height == 100

            if pre_encode_images:
                assert data.image.bit_depth == 8

        assert count == limit


if __name__ == "__main__":
    test_get_sample_filesystem(True)
