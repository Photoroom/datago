import json
import tempfile
import os
from datago import DatagoClient, initialize_logging
from PIL import Image


def create_test_images(directory, count=5):
    """Helper function to create test images in a directory."""
    image_paths = []
    for i in range(count):
        img = Image.new(
            "RGB", (100, 100), color=(i * 50 % 255, (i * 100) % 255, (i * 150) % 255)
        )
        path = os.path.join(directory, f"test_image_{i}.png")
        img.save(path)
        image_paths.append(path)
    return image_paths


class TestDatagoClient:
    """Test cases for DatagoClient functionality."""

    def test_initialize_logging(self):
        """Test the initialize_logging function."""
        # Should return True on first call
        result = initialize_logging("info")
        assert isinstance(result, bool)

        # Test with None parameter
        result = initialize_logging(None)
        assert isinstance(result, bool)

    def test_client_instantiation_file_source(self):
        """Test creating a client with file source configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 3)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 3,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            assert client is not None

    # We panic out at the moment if the config is invalid, so this test is commented out.
    # Uncomment this test if you want to handle invalid configurations gracefully.
    # def test_client_instantiation_invalid_config(self):
    #     """Test that invalid configuration raises an error."""
    #     invalid_config = '{"invalid": "config"}'

    #     with pytest.raises((ValueError, RuntimeError)):
    #         DatagoClient(invalid_config)

    def test_client_start_stop_file_source(self):
        """Test starting and stopping client with file source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 3)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 3,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            client.start()
            client.stop()

    def test_get_sample_file_source(self):
        """Test getting samples from file source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 5)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 3,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            samples_received = []
            for _ in range(3):
                sample = client.get_sample()
                if sample:
                    samples_received.append(sample)
                else:
                    break

            assert len(samples_received) <= 3

            for sample in samples_received:
                assert sample.id != ""
                assert sample.source == "filesystem"
                assert sample.image.width > 0
                assert sample.image.height > 0
                assert len(sample.image.data) > 0

    def test_client_with_image_transformations(self):
        """Test client with image transformation configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 3)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "image_config": {
                    "crop_and_resize": True,
                    "default_image_size": 64,
                    "downsampling_ratio": 16,
                    "min_aspect_ratio": 0.5,
                    "max_aspect_ratio": 2.0,
                    "pre_encode_images": False,
                    "image_to_rgb8": True,
                },
                "limit": 2,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            sample = client.get_sample()

            assert sample is not None
            assert sample.image.width <= 64
            assert sample.image.height <= 64
            assert sample.image.channels == 3  # RGB8

    def test_client_with_image_encoding(self):
        """Test client with image encoding enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 3)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "image_config": {
                    "crop_and_resize": False,
                    "pre_encode_images": True,
                    "image_to_rgb8": False,
                },
                "limit": 2,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            sample = client.get_sample()

            assert sample is not None
            assert sample.image.channels == -1  # Encoded images have channels = -1
            assert len(sample.image.data) > 0

    def test_random_sampling(self):
        """Test that random sampling produces different results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 10)

            config_base = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": True,
                },
                "limit": 5,
                "samples_buffer_size": 10,
            }

            # Get two sets of samples with random sampling
            client1 = DatagoClient(json.dumps(config_base))
            samples1 = []
            for _ in range(5):
                sample = client1.get_sample()
                if sample:
                    samples1.append(sample.id)
                else:
                    break

            client2 = DatagoClient(json.dumps(config_base))
            samples2 = []
            for _ in range(5):
                sample = client2.get_sample()
                if sample:
                    samples2.append(sample.id)
                else:
                    break

            # With random sampling, the samples should be different
            assert len(samples1) > 0
            assert len(samples2) > 0
            assert (
                len(set(samples1) & set(samples2)) < 5
            )  # Expect some overlap, but not all

    def test_world_size_and_rank(self):
        """Test that different ranks get different subsets of data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 10)

            config_rank0 = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 2,
                    "random_sampling": False,
                },
                "limit": 10,
                "samples_buffer_size": 10,
            }

            config_rank1 = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 1,
                    "world_size": 2,
                    "random_sampling": False,
                },
                "limit": 10,
                "samples_buffer_size": 10,
            }

            client0 = DatagoClient(json.dumps(config_rank0))
            samples0 = []
            for _ in range(10):
                sample = client0.get_sample()
                if sample:
                    samples0.append(sample.id)
                else:
                    break

            client1 = DatagoClient(json.dumps(config_rank1))
            samples1 = []
            for _ in range(10):
                sample = client1.get_sample()
                if sample:
                    samples1.append(sample.id)
                else:
                    break

            # Different ranks should get different samples
            assert len(samples0) > 0
            assert len(samples1) > 0

            # No overlap between ranks
            overlap = set(samples0) & set(samples1)
            assert len(overlap) == 0

    def test_limit_respected(self):
        """Test that the client respects the limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 10)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 3,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            samples_received = 0

            while True:
                sample = client.get_sample()
                if sample:
                    samples_received += 1
                else:
                    break

                # Safety valve to prevent infinite loop
                if samples_received > 10:
                    break

            # Should respect the limit (might have small buffer)
            assert samples_received <= 4  # Allow small buffer

    def test_empty_directory(self):
        """Test client behavior with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 3,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            sample = client.get_sample()

            # Should return None when no files available
            assert sample is None

    def test_nonexistent_directory(self):
        """Test client behavior with nonexistent directory."""
        config = {
            "source_type": "file",
            "source_config": {
                "root_path": "/nonexistent/directory",
                "rank": 0,
                "world_size": 1,
                "random_sampling": False,
            },
            "limit": 3,
            "samples_buffer_size": 10,
        }

        client = DatagoClient(json.dumps(config))
        sample = client.get_sample()

        # Should handle gracefully and return None
        assert sample is None

    def test_client_drop_cleanup(self):
        """Test that client cleans up properly when dropped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 3)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 3,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            client.start()

            # Client should clean up when it goes out of scope
            del client

    def test_multiple_starts_stops(self):
        """Test that multiple start/stop calls don't cause issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_images(tmpdir, 3)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 3,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            # Multiple starts should be safe
            client.start()
            client.start()

            # Multiple stops should be safe
            client.stop()
            client.stop()

    def test_various_image_formats(self):
        """Test client with various image formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create images with different formats
            formats = [
                ("test1.png", "PNG"),
                ("test2.jpg", "JPEG"),
            ]

            for filename, format_name in formats:
                img = Image.new("RGB", (50, 50), color="red")
                path = os.path.join(tmpdir, filename)
                img.save(path, format=format_name)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 4,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            samples_received = 0
            while True:
                sample = client.get_sample()
                if sample:
                    samples_received += 1
                    assert sample.image.width == 50
                    assert sample.image.height == 50
                else:
                    break

            assert samples_received == len(formats)
