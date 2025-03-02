from PIL import Image
from datago import DatagoClient
import json
import tempfile


def test_get_sample_filesystem():
    samples = 10

    with tempfile.TemporaryDirectory() as tmpdirname:
        cwd = tmpdirname
        # Dump a sample image to the filesystem
        for i in range(samples):
            img = Image.new("RGB", (100, 100))
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
                "pre_encode_images": False,
            },
            "limit": samples,
            "prefetch_buffer_size": 64,
            "samples_buffer_size": 10,
            "rank": 0,
            "world_size": 1,
        }

        client = DatagoClient(json.dumps(client_config))
        for i in range(samples):
            data = client.get_sample()
            assert data.id != ""
            assert data.image.width == 100
            assert data.image.height == 100


if __name__ == "__main__":
    test_get_sample_filesystem()
