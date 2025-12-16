from PIL import Image
from datago import DatagoClient
import json
import tempfile
import pytest
from io import BytesIO


def generate_tmp_files(dir: str, limit: int, rgb16: bool = False, rgba: bool = False):
    for i in range(limit):
        # Prepare an ephemeral test set
        mode = "RGBA" if rgba else "RGB"
        img = Image.new(mode, (100, 100))

        # Randomly make the image 16 bits
        if rgb16:
            img = img.convert("I;16")

        img.save(dir + f"/test_{i}.png")


@pytest.mark.parametrize(
    ["pre_encode_images", "rgb16", "rgba"],
    [(a, b, c) for a in [True, False] for b in [True, False] for c in [True, False]],
)
def test_get_sample_filesystem(pre_encode_images: bool, rgb16: bool, rgba: bool):
    limit = 10

    with tempfile.TemporaryDirectory() as tmpdirname:
        generate_tmp_files(tmpdirname, limit, rgb16, rgba)

        # Check that we can instantiate a client and get a sample, nothing more
        client_config = {
            "source_type": "file",
            "source_config": {
                "root_path": tmpdirname,
                "rank": 0,
                "world_size": 1,
                "random_sampling": True,
            },
            "image_config": {
                "crop_and_resize": False,
                "default_image_size": 512,
                "downsampling_ratio": 16,
                "min_aspect_ratio": 0.5,
                "max_aspect_ratio": 2.0,
                "pre_encode_images": pre_encode_images,
                "image_to_rgb8": rgb16 or rgba,
            },
            "limit": limit,
            "prefetch_buffer_size": 64,
            "samples_buffer_size": 10,
        }

        client = DatagoClient(json.dumps(client_config))
        count = 0
        for _ in range(limit):
            data = client.get_sample()
            if not data:
                break
            count += 1
            assert data.id != ""

            image_payload = data.image.get_payload()
            assert image_payload.width == 100
            assert image_payload.height == 100

            if rgb16:
                assert image_payload.bit_depth == 8

            # Open the image in python scope and check properties
            if pre_encode_images:
                test_image = Image.open(BytesIO(bytes(image_payload.data)))
                assert test_image.width == 100
                assert test_image.height == 100
                assert test_image.mode == "RGB"

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
                "rank": 0,
                "world_size": 1,
                "random_sampling": True,
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
    test_get_sample_filesystem(True, True, True)
