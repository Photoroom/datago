import pytest
import json
import tempfile
import os
from datago import DatagoClient
from PIL import Image
import threading
import time


def create_test_image(path, size=(100, 100), color="red"):
    """Helper to create a test image."""
    img = Image.new("RGB", size, color=color)
    img.save(path)


class TestDatagoEdgeCases:
    """Test edge cases and error conditions for DatagoClient."""

    def test_corrupted_image_files(self):
        """Test behavior with corrupted image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a corrupted image file
            corrupted_path = os.path.join(tmpdir, "corrupted.png")
            with open(corrupted_path, "wb") as f:
                f.write(b"This is not a valid image file")

            # Create one valid image
            valid_path = os.path.join(tmpdir, "valid.png")
            create_test_image(valid_path)

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 5,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            # Should skip corrupted files and process valid ones
            samples = []
            for _ in range(5):
                sample = client.get_sample()
                if sample:
                    samples.append(sample)
                else:
                    break

            # Should get at least the valid image
            assert len(samples) >= 1
            for sample in samples:
                assert sample.image.width > 0
                assert sample.image.height > 0

    def test_very_large_images(self):
        """Test handling of large images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large image
            large_image_path = os.path.join(tmpdir, "large.png")
            create_test_image(large_image_path, size=(2000, 2000))

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
                    "default_image_size": 512,
                    "downsampling_ratio": 16,
                    "min_aspect_ratio": 0.5,
                    "max_aspect_ratio": 2.0,
                    "pre_encode_images": False,
                    "image_to_rgb8": True,
                },
                "limit": 1,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            sample = client.get_sample()

            assert sample is not None
            assert sample.image.original_width == 2000
            assert sample.image.original_height == 2000
            # Should be resized
            assert sample.image.width <= 512
            assert sample.image.height <= 512

    def test_very_small_images(self):
        """Test handling of very small images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a tiny image
            tiny_image_path = os.path.join(tmpdir, "tiny.png")
            create_test_image(tiny_image_path, size=(1, 1))

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 1,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            sample = client.get_sample()

            assert sample is not None
            assert sample.image.width == 1
            assert sample.image.height == 1

    def test_extreme_aspect_ratios(self):
        """Test handling of images with extreme aspect ratios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create very wide image
            wide_path = os.path.join(tmpdir, "wide.png")
            create_test_image(wide_path, size=(1000, 10))

            # Create very tall image
            tall_path = os.path.join(tmpdir, "tall.png")
            create_test_image(tall_path, size=(10, 1000))

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
                    "default_image_size": 224,
                    "downsampling_ratio": 16,
                    "min_aspect_ratio": 0.5,
                    "max_aspect_ratio": 2.0,
                    "pre_encode_images": False,
                    "image_to_rgb8": False,
                },
                "limit": 2,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            samples = []
            for _ in range(2):
                sample = client.get_sample()
                if sample:
                    samples.append(sample)
                else:
                    break

            assert len(samples) == 2
            for sample in samples:
                # Should be cropped to fit within aspect ratio constraints
                aspect_ratio = sample.image.width / sample.image.height
                assert 0.5 <= aspect_ratio <= 2.0

    def test_concurrent_access(self):
        """Test concurrent access to the same client."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test images
            for i in range(10):
                path = os.path.join(tmpdir, f"test_{i}.png")
                create_test_image(path, color=(i * 25, i * 25, i * 25))

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 10,
                "samples_buffer_size": 20,
            }

            client = DatagoClient(json.dumps(config))
            results = []
            errors = []

            def worker():
                try:
                    for _ in range(3):
                        sample = client.get_sample()
                        if sample:
                            results.append(sample.id)
                        time.sleep(0.01)  # Small delay
                except Exception as e:
                    errors.append(e)

            # Start multiple threads
            threads = []
            for _ in range(3):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Should not have any errors
            assert len(errors) == 0
            # Should have gotten some results
            assert len(results) > 0

    def test_invalid_world_size_rank_combinations(self):
        """Test invalid world_size and rank combinations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, "test.png"))

            # Test rank >= world_size
            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 2,
                    "world_size": 2,  # rank should be < world_size
                    "random_sampling": False,
                },
                "limit": 1,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))
            _sample = client.get_sample()

            # Should handle gracefully (might return None or work with adjusted parameters)
            # The exact behavior depends on implementation

    def test_very_large_buffer_sizes(self):
        """Test with very large buffer sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, "test.png"))

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 1,
                "samples_buffer_size": 10000,  # Very large buffer
            }

            client = DatagoClient(json.dumps(config))
            sample = client.get_sample()

            assert sample is not None

    def test_mixed_file_types(self):
        """Test directory with mix of image and non-image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image files
            create_test_image(os.path.join(tmpdir, "image1.png"))
            create_test_image(os.path.join(tmpdir, "image2.jpg"))

            # Create non-image files
            with open(os.path.join(tmpdir, "text.txt"), "w") as f:
                f.write("This is a text file")

            with open(os.path.join(tmpdir, "data.json"), "w") as f:
                json.dump({"key": "value"}, f)

            os.makedirs(os.path.join(tmpdir, "subdirectory"))
            create_test_image(os.path.join(tmpdir, "subdirectory", "image3.png"))

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 10,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            samples = []
            for _ in range(10):
                sample = client.get_sample()
                if sample:
                    samples.append(sample)
                else:
                    break

            # Should only process image files (including in subdirectories)
            assert len(samples) == 3
            for sample in samples:
                assert sample.image.width > 0
                assert sample.image.height > 0

    def test_special_characters_in_filenames(self):
        """Test handling of special characters in filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create images with special characters in names
            special_names = [
                "image with spaces.png",
                "image-with-dashes.png",
                "image_with_underscores.png",
                "image.with.dots.png",
                "image@symbol.png",
            ]

            for name in special_names:
                try:
                    path = os.path.join(tmpdir, name)
                    create_test_image(path)
                except OSError:
                    # Skip if filesystem doesn't support the filename
                    continue

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 10,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            samples = []
            for _ in range(10):
                sample = client.get_sample()
                if sample:
                    samples.append(sample)
                else:
                    break

            # Should handle special characters in filenames
            assert len(samples) > 0

    def test_readonly_directory(self):
        """Test behavior with read-only directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, "test.png"))

            # Make directory read-only
            try:
                os.chmod(tmpdir, 0o555)

                config = {
                    "source_type": "file",
                    "source_config": {
                        "root_path": tmpdir,
                        "rank": 0,
                        "world_size": 1,
                        "random_sampling": False,
                    },
                    "limit": 1,
                    "samples_buffer_size": 10,
                }

                client = DatagoClient(json.dumps(config))
                sample = client.get_sample()

                # Should still be able to read files
                assert sample is not None

            finally:
                # Restore permissions for cleanup
                os.chmod(tmpdir, 0o755)

    def test_deep_directory_structure(self):
        """Test with deeply nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directories
            deep_path = tmpdir
            for i in range(5):
                deep_path = os.path.join(deep_path, f"level_{i}")
                os.makedirs(deep_path, exist_ok=True)

            # Create image in deep path
            create_test_image(os.path.join(deep_path, "deep_image.png"))

            # Create image in root
            create_test_image(os.path.join(tmpdir, "root_image.png"))

            config = {
                "source_type": "file",
                "source_config": {
                    "root_path": tmpdir,
                    "rank": 0,
                    "world_size": 1,
                    "random_sampling": False,
                },
                "limit": 5,
                "samples_buffer_size": 10,
            }

            client = DatagoClient(json.dumps(config))

            samples = []
            for _ in range(5):
                sample = client.get_sample()
                if sample:
                    samples.append(sample)
                else:
                    break

            # Should find images in nested directories
            assert len(samples) == 2
