import os
from PIL import Image
from datago import datago


# FIXME: Would need to generate more fake data to test this
def no_test_get_sample_filesystem():
    cwd = os.getcwd()

    try:
        # Dump a sample image to the filesystem
        img = Image.new("RGB", (100, 100))
        img.save(cwd + "/test.png")

        # Check that we can instantiate a client and get a sample, nothing more
        client_config = datago.GetDatagoConfig()
        client_config.SourceType = "filesystem"
        client_config.SamplesBufferSize = 1

        source_config = datago.SourceFileSystemConfig()
        source_config.RootPath = cwd
        source_config.PageSize = 1

        client = datago.GetClient(client_config, source_config)
        data = client.GetSample()
        assert data.ID != ""
    finally:
        os.remove(cwd + "/test.png")
