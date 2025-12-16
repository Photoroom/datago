import json
import os
import time

import typer
from benchmark_defaults import IMAGE_CONFIG
from dataset import DatagoIterDataset
from tqdm import tqdm


def benchmark(
    limit: int = typer.Option(10, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(
        True, help="Crop and resize the images on the fly"
    ),
    compare_wds: bool = typer.Option(True, help="Compare against torch dataloader"),
    num_workers: int = typer.Option(
        16,
        help="Number of processes to use",
    ),
    sweep: bool = typer.Option(False, help="Sweep over the number of processes"),
):
    if sweep:
        results = {}
        for num_workers in range(2, max(64, (os.cpu_count() or 1)), 8):
            results[num_workers] = benchmark(limit, crop_and_resize, compare_wds, num_workers, False)

        # Save results to a json file
        with open("benchmark_results_wds.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

    # URL of the test bucket
    # bucket = "https://storage.googleapis.com/webdataset/fake-imagenet"
    # dataset = "/imagenet-train-{000000..001281}.tar"

    bucket = "https://huggingface.co/datasets/sayakpaul/pd12m-full/resolve/"
    dataset = "main/{00155..02480}.tar"
    url = bucket + dataset

    print(
        f"Benchmarking Datago WDS path on {url}.\nRunning benchmark for {limit} samples"
    )
    client_config = {
        "source_type": "webdataset",
        "source_config": {
            "url": url,
            "shuffle": True,
            "max_concurrency": num_workers,  # Number of concurrent TarballSample downloads and dispatch
            "auth_token": os.environ.get("HF_TOKEN", default=""),
        },
        "prefetch_buffer_size": 256,
        "samples_buffer_size": 256,
        "limit": limit,
    }

    if crop_and_resize:
        # Optionally add a custom image config to crop and resize the images on the fly
        client_config["image_config"] = IMAGE_CONFIG

    # # Make sure in the following that we compare apples to apples, meaning in that case
    # # that we materialize the payloads in the python scope in the expected format
    # # (PIL.Image for images and masks for instance, numpy arrays for latents)
    datago_dataset = DatagoIterDataset(client_config, return_python_types=True)
    start = time.time()  # Note that the datago dataset will start preparing samples (up to the requested buffer size) at construction time

    img, count = None, 0
    for sample in tqdm(datago_dataset, desc="Datago", dynamic_ncols=True):
        assert sample["id"] != ""
        img = sample["image"]
        count += 1

    assert count == limit, f"Expected {limit} samples, got {count}"
    fps = limit / (time.time() - start)
    print(f"-- Datago WDS FPS {fps:.2f} - workers {num_workers}")
    results = {"datago": {"fps": fps, "count": count}}
    del datago_dataset

    # Save the last image as a test
    assert img is not None, "No image - benchmark did not run"
    img.save("benchmark_last_image.png")

    # Let's compare against a classic webdataset dataloader
    if compare_wds:
        import webdataset as wds
        from torch.utils.data import DataLoader
        from torchvision import transforms

        print("\nBenchmarking webdataset library dataloader")
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

        def custom_transform(sample):
            if "jpg" in sample:
                sample["jpg"] = transform(sample["jpg"])
            if "png" in sample:
                sample["png"] = transform(sample["png"])
            return sample

        # Create a WebDataset instance
        dataset = (
            wds.WebDataset(url, shardshuffle=False)
            .shuffle(256)  # Optional: shuffle with buffer
            .decode("pil")  # Decode images using PIL
            .map(custom_transform)
            # .to_tuple("png", "cls")  # Map keys to output tuple
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=2,
            collate_fn=lambda x: x,
        )

        # Iterate over the DataLoader
        start = time.time()
        for n_images, _ in enumerate(tqdm(dataloader, desc="WDS", dynamic_ncols=True)):
            if n_images > limit:
                break
        fps = n_images / (time.time() - start)
        print(f"-- Webdataset lib FPS ({num_workers} processes) {fps:.2f}")

        results["webdataset"] = {"fps": fps, "count": n_images}
        return results

if __name__ == "__main__":
    typer.run(benchmark)
