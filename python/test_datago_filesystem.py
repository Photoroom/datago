from PIL import Image
from datago import DatagoClient
import json
import tempfile
import pytest
import random


def generate_tmp_files(dir, limit):
    for i in range(limit):
        # Prepare an ephemeral test set
        mode = "RGB" if random.random() > 0.5 else "RGBA"
        img = Image.new(mode, (100, 100))

        # Randomly make the image 16 bits
        if random.random() > 0.5:
            img = img.convert("I;16")

        img.save(dir + f"/test_{i}.png")


@pytest.mark.parametrize("pre_encode_images", [False, True])
def test_get_sample_filesystem(pre_encode_images: bool):
    limit = 10

    with tempfile.TemporaryDirectory() as tmpdirname:
        generate_tmp_files(tmpdirname, limit)

        # Check that we can instantiate a client and get a sample, nothing more
        client_config = {
            "source_type": "file",
            "source_config": {
                "root_path": tmpdirname,
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


def test_random_walk():
    limit = 50

    with tempfile.TemporaryDirectory() as tmpdirname:
        generate_tmp_files(tmpdirname, limit)

        # Check that we can instantiate a client and get a sample, nothing more
        client_config = {
            "source_type": "file",
            "source_config": {
                "root_path": tmpdirname,
                "random_order": True,
            },
            "image_config": {
                "crop_and_resize": False,
                "default_image_size": 512,
                "downsampling_ratio": 16,
                "min_aspect_ratio": 0.5,
                "max_aspect_ratio": 2.0,
                "pre_encode_images": False,
            },
            "limit": limit,
            "prefetch_buffer_size": 64,
            "samples_buffer_size": 10,
            "rank": 0,
            "world_size": 1,
        }

        results = []
        n_runs = 3
        for _ in range(n_runs):
            client = DatagoClient(json.dumps(client_config))
            run_results = []
            for i in range(limit):
                data = client.get_sample()
                if not data:
                    break
                run_results.append(data.id)

            results.append(run_results)

        # Check that the results are all different
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                assert results[i] != results[j]


if __name__ == "__main__":
    test_get_sample_filesystem(True)
