import json
import os
import time

import numpy as np
import typer
from PIL import Image
from benchmark_defaults import IMAGE_CONFIG
from dataset import DatagoIterDataset
from tqdm import tqdm


def _passthrough(x):
    return x


def _to_numpy(image):
    return np.asarray(image)


def _image_nbytes(image) -> int:
    """Raw pixel payload size in bytes (metadata excluded)."""
    # ImageFolder yields (image, target) tuples; unwrap if needed.
    if isinstance(image, (tuple, list)):
        image = image[0]
    if isinstance(image, np.ndarray):
        return int(image.nbytes)
    # PIL.Image: 1 byte per band for the 8-bit modes datago produces.
    return image.width * image.height * len(image.getbands())


def benchmark(
    root_path: str = typer.Option(
        os.getenv("DATAGO_TEST_FILESYSTEM", ""), help="The source to test out"
    ),
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(
        False, help="Crop and resize the images on the fly"
    ),
    compare_torch: bool = typer.Option(True, help="Compare against torch dataloader"),
    num_workers: int = typer.Option(os.cpu_count(), help="Number of workers to use"),
    sweep: bool = typer.Option(False, help="Sweep over the number of workers"),
    image_format: str = typer.Option(
        "pil", help="Materialized image type for both loaders: 'pil' or 'numpy'"
    ),
):
    image_format = image_format.lower()
    if image_format not in ("pil", "numpy"):
        raise typer.BadParameter("image_format must be 'pil' or 'numpy'")

    if sweep:
        results_sweep = {}
        num_workers = 1
        while num_workers <= (os.cpu_count() or 16):
            results_sweep[num_workers] = benchmark(
                root_path,
                limit,
                crop_and_resize,
                compare_torch,
                num_workers,
                False,
                image_format,
            )
            num_workers *= 2

        with open("benchmark_results_filesystem.json", "w") as f:
            json.dump(results_sweep, f, indent=2)

        return results_sweep

    print(
        f"Running benchmark for {root_path} - {limit} samples - {num_workers} workers "
        f"- image_format={image_format}"
    )

    # This setting is not exposed in the config, but an env variable can be used instead
    os.environ["DATAGO_MAX_TASKS"] = str(num_workers)

    client_config = {
        "source_type": "file",
        "source_config": {
            "root_path": root_path,
            "rank": 0,
            "world_size": 1,
        },
        "prefetch_buffer_size": 256,
        "samples_buffer_size": 256,
        "limit": limit,
        # "pil" (default) or "numpy", which hands back a zero-copy read-only ndarray
        "image_format": image_format,
    }

    if crop_and_resize:
        client_config["image_config"] = IMAGE_CONFIG

    # Make sure in the following that we compare apples to apples, meaning in that case
    # that we materialize the payloads in the python scope in the expected format
    # (PIL.Image for images and masks for instance, numpy arrays for latents)
    datago_dataset = DatagoIterDataset(client_config, return_python_types=True)
    start = time.time()  # Note that the datago dataset will start walking the filesystem at construction time

    img = None
    count = 0
    total_bytes = 0
    for sample in tqdm(
        datago_dataset, desc=f"Datago ({image_format})", dynamic_ncols=True
    ):
        assert sample["id"] != ""
        img = sample["image"]
        total_bytes += _image_nbytes(img)

        if count < limit - 1:
            del img
            img = None  # Help with memory pressure

        count += 1

    assert count == limit, f"Expected {limit} samples, got {count}"
    elapsed = time.time() - start
    fps = limit / elapsed
    bandwidth_mbps = total_bytes / elapsed / 1e6
    results = {
        "datago": {
            "fps": fps,
            "count": count,
            "bytes": total_bytes,
            "bandwidth_mbps": bandwidth_mbps,
            "image_format": image_format,
        }
    }
    print(
        f"Datago - FPS {fps:.2f} - BW {bandwidth_mbps:.1f} MB/s - "
        f"workers {num_workers} - image_format {image_format}"
    )
    del datago_dataset

    # Save the last image as a test. This goes through Image.fromarray() whether
    # the payload is a PIL image or a (zero-copy) numpy buffer, so a corrupt
    # buffer would fail here.
    assert img is not None, "No image - benchmark did not run"
    last_image = np.asarray(img)
    if last_image.dtype != np.uint8:
        # Wider dtypes (uint16/float32) can't be saved from an array directly;
        # scale a preview down to 8-bit. The full-depth data is untouched.
        if np.issubdtype(last_image.dtype, np.floating):
            last_image = np.clip(last_image, 0.0, 1.0) * 255.0
        else:
            last_image = (
                last_image.astype(np.float64) / np.iinfo(last_image.dtype).max * 255.0
            )
        last_image = np.rint(last_image).astype(np.uint8)
    # PIL wants (H, W) for single-channel and (H, W, C) otherwise; squeeze a
    # trailing singleton axis defensively for either loader.
    if last_image.ndim == 3 and last_image.shape[2] == 1:
        last_image = last_image[:, :, 0]
    Image.fromarray(last_image).save("benchmark_last_image.png")

    # Let's compare against a classic pytorch dataloader
    if compare_torch:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        # Materialize the torch side in the same format as datago for fairness.
        if crop_and_resize:
            resize = transforms.Resize(
                (1024, 1024), interpolation=transforms.InterpolationMode.LANCZOS
            )
            transform = (
                transforms.Compose([resize, _to_numpy])
                if image_format == "numpy"
                else resize
            )
        else:
            transform = _to_numpy if image_format == "numpy" else None

        # Create the ImageFolder dataset
        dataset = datasets.ImageFolder(
            root=root_path, transform=transform, allow_empty=True
        )

        # Create a DataLoader to allow for multiple workers
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_passthrough,
        )

        # Iterate over the DataLoader
        start = time.time()
        n_images = 0
        total_bytes = 0
        for batch in tqdm(
            dataloader, desc=f"Torch ({image_format})", dynamic_ncols=True
        ):
            n_images += len(batch)
            for item in batch:
                total_bytes += _image_nbytes(item)
            if n_images > limit:
                break

            del batch  # Help with memory pressure, same as above
        elapsed = time.time() - start
        fps = n_images / elapsed
        bandwidth_mbps = total_bytes / elapsed / 1e6
        results["torch"] = {
            "fps": fps,
            "count": n_images,
            "bytes": total_bytes,
            "bandwidth_mbps": bandwidth_mbps,
            "image_format": image_format,
        }
        print(
            f"Torch - FPS {fps:.2f} - BW {bandwidth_mbps:.1f} MB/s - "
            f"workers {num_workers} - image_format {image_format}"
        )

    return results


if __name__ == "__main__":
    typer.run(benchmark)
