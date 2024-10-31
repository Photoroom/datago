from datago import datago
import pytest
import os
from PIL import Image


def get_test_source():
    return os.getenv("DATAROOM_TEST_SOURCE")


def test_get_sample_db():
    # Check that we can instantiate a client and get a sample, nothing more
    config = datago.GetDefaultConfig()
    config.source = get_test_source()
    config.sample = 10
    client = datago.GetClient(config)
    data = client.GetSample()
    assert data.ID != ""


def test_get_sample_filesystem():
    cwd = os.getcwd()

    try:
        # Dump a sample image to the filesystem
        img = Image.new("RGB", (100, 100))
        img.save(cwd + "/test.png")

        # Check that we can instantiate a client and get a sample, nothing more
        config = datago.GetDefaultConfig()
        config.SourceType = "filesystem"
        config.Sources = cwd
        config.sample = 1

        client = datago.GetClient(config)
        data = client.GetSample()
        assert data.ID != ""
    finally:
        os.remove(cwd + "/test.png")


# TODO: Backport all the image correctness tests

if __name__ == "__main__":
    pytest.main(["-v", __file__])
