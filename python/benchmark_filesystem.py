import json
import os
import time

import typer
from benchmark_defaults import IMAGE_CONFIG
from dataset import DatagoIterDataset
from tqdm import tqdm


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
):
    if sweep:
        results_sweep = {}
        num_workers = 1

        while num_workers <= (os.cpu_count() or 16):
            results_sweep[num_workers] = benchmark(
                root_path, limit, crop_and_resize, compare_torch, num_workers, False
            )
            num_workers *= 2

        with open("benchmark_results_filesystem.json", "w") as f:
            json.dump(results_sweep, f, indent=2)

        return results_sweep

    print(
        f"Running benchmark for {root_path} - {limit} samples - {num_workers} workers"
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
    for sample in tqdm(datago_dataset, desc="Datago", dynamic_ncols=True):
        assert sample["id"] != ""
        img = sample["image"]

        if count < limit - 1:
            del img
            img = None  # Help with memory pressure

        count += 1

    assert count == limit, f"Expected {limit} samples, got {count}"
    fps = limit / (time.time() - start)
    results = {"datago": {"fps": fps, "count": count}}
    print(f"Datago - FPS {fps:.2f} - workers {num_workers}")
    del datago_dataset

    # Save the last image as a test
    assert img is not None, "No image - benchmark did not run"
    img.save("benchmark_last_image.png")

    # Let's compare against a classic pytorch dataloader
    if compare_torch:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        # Define the transformations to apply to each image
        transform = (
            transforms.Compose(
                [
                    transforms.Resize(
                        (1024, 1024), interpolation=transforms.InterpolationMode.LANCZOS
                    ),
                ]
            )
            if crop_and_resize
            else None
        )

        # Create the ImageFolder dataset
        dataset = datasets.ImageFolder(
            root=root_path, transform=transform, allow_empty=True
        )

        # Create a DataLoader to allow for multiple workers
        # Use available CPU count for num_workers
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )

        # Iterate over the DataLoader
        start = time.time()
        n_images = 0
        for batch in tqdm(dataloader, desc="Torch", dynamic_ncols=True):
            n_images += len(batch)
            if n_images > limit:
                break

            del batch  # Help with memory pressure, same as above
        fps = n_images / (time.time() - start)
        results["torch"] = {"fps": fps, "count": n_images}
        print(f"Torch - FPS {fps:.2f} - workers {num_workers}")

    return results


if __name__ == "__main__":
    typer.run(benchmark)
