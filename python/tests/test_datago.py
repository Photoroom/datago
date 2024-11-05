from datago import datago
import pytest
import os
from PIL import Image


def get_test_source():
    return os.getenv("DATAROOM_TEST_SOURCE")


def test_get_sample_db():
    # Check that we can instantiate a client and get a sample, nothing more
    client_config = datago.DatagoConfig()
    client_config.SetDefaults()
    client_config.SamplesBufferSize = 10

    source_config = datago.GeneratorDBConfig()
    source_config.SetDefaults()
    source_config.Sources = get_test_source()

    client = datago.GetClient(client_config)
    data = client.GetSample()
    assert data.ID != ""


def no_test_get_sample_filesystem():
    cwd = os.getcwd()

    try:
        # Dump a sample image to the filesystem
        img = Image.new("RGB", (100, 100))
        img.save(cwd + "/test.png")

        # Check that we can instantiate a client and get a sample, nothing more
        client_config = datago.DatagoConfig()
        client_config.SetDefaults()
        client_config.SourceType = "filesystem"
        client_config.SamplesBufferSize = 1

        source_config = datago.GeneratorFileSystemConfig()
        source_config.RootPath = cwd
        source_config.PageSize = 1

        client = datago.GetClient(client_config, source_config)
        data = client.GetSample()
        assert data.ID != ""
    finally:
        os.remove(cwd + "/test.png")


# TODO: Backport all the image correctness tests

if __name__ == "__main__":
    pytest.main(["-v", __file__])
