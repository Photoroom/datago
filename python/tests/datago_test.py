from datago import datago
import pytest
import os


def get_test_source():
    return os.getenv("DATAROOM_TEST_SOURCE")


def test_get_sample():
    # Check that we can instantiate a client and get a sample, nothing more
    config = datago.GetDefaultConfig()
    config.source = get_test_source()
    config.sample = 10
    client = datago.GetClient(config)
    data = client.GetSample()
    assert data.ID != ""


# TODO: Backport all the image correctness tests

if __name__ == "__main__":
    pytest.main(["-v", __file__])
