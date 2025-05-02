import time
from tqdm import tqdm
import typer
from dataset import DatagoIterDataset
import os


def benchmark(
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(True, help="Crop and resize the images on the fly"),
    compare_wds: bool = typer.Option(True, help="Compare against torch dataloader"),
):
    # URL of the test bucket
    # bucket = "https://storage.googleapis.com/webdataset/fake-imagenet"
    # dataset = "/imagenet-train-{000000..001281}.tar"

    bucket = "https://huggingface.co/datasets/sayakpaul/pd12m-full/resolve/"
    dataset = "main/{00155..02480}.tar"

    url = bucket + dataset

    print(f"Benchmarking Datago WDS path on {url}.\nRunning benchmark for {limit} samples")
    client_config = {
        "source_type": "webdataset",
        "source_config": {
            "url": url,
            "shuffle": True,
            "max_concurrency": 8,  # Number of concurrent tarball downloads and dispatch
            "auth_token": os.environ.get("HF_TOKEN", default=""),
        },
        "image_config": {
            "crop_and_resize": crop_and_resize,
            "default_image_size": 1024,
            "downsampling_ratio": 32,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
            "pre_encode_images": False,
        },
        "prefetch_buffer_size": 256,
        "samples_buffer_size": 256,
        "limit": limit,
        "rank": 0,
        "world_size": 1,
    }

    # # Make sure in the following that we compare apples to apples, meaning in that case
    # # that we materialize the payloads in the python scope in the expected format
    # # (PIL.Image for images and masks for instance, numpy arrays for latents)
    datago_dataset = DatagoIterDataset(client_config, return_python_types=True)
    start = time.time()  # Note that the datago dataset will start walking the filesystem at construction time

    img, count = None, 0
    for sample in tqdm(datago_dataset, dynamic_ncols=True):
        assert sample["id"] != ""
        img = sample["image"]
        count += 1

    assert count == limit, f"Expected {limit} samples, got {count}"
    fps = limit / (time.time() - start)
    print(f"-- Datago WDS FPS {fps:.2f}")
    del datago_dataset

    # Save the last image as a test
    assert img is not None, "No image - benchmark did not run"
    img.save("benchmark_last_image.png")

    # Let's compare against a classic webdataset dataloader
    if compare_wds:
        from torchvision import transforms
        from torch.utils.data import DataLoader
        import webdataset as wds

        print("\nBenchmarking webdataset library dataloader")
        # Define the transformations to apply to each image
        transform = (
            transforms.Compose(
                [
                    transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.LANCZOS),
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
            .shuffle(1000)  # Optional: shuffle with buffer
            .decode("pil")  # Decode images using PIL
            .map(custom_transform)
            # .to_tuple("png", "cls")  # Map keys to output tuple
        )

        n_processes = 16  # Change to use whatever feels right for your machine
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=n_processes,
            prefetch_factor=2,
            collate_fn=lambda x: x,
        )

        # Iterate over the DataLoader
        start = time.time()
        n_images = 0
        for _ in dataloader:
            n_images += 1
            if n_images > limit:
                break
        fps = n_images / (time.time() - start)
        print(f"-- Webdataset lib FPS ({n_processes} processes) {fps:.2f}")


if __name__ == "__main__":
    typer.run(benchmark)
