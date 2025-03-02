import time
from tqdm import tqdm
import os
import typer
from dataset import DatagoIterDataset


def benchmark(
    root_path: str = typer.Option(
        os.getenv("DATAGO_TEST_FILESYSTEM", ""), help="The source to test out"
    ),
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(
        True, help="Crop and resize the images on the fly"
    ),
    compare_torch: bool = typer.Option(True, help="Compare against torch dataloader"),
):
    print(f"Running benchmark for {root_path} - {limit} samples")
    client_config = {
        "source_type": "file",
        "source_config": {
            "root_path": root_path,
        },
        "image_config": {
            "crop_and_resize": crop_and_resize,
            "default_image_size": 1024,
            "downsampling_ratio": 32,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
            "pre_encode_images": False,
        },
        "prefetch_buffer_size": 128,
        "samples_buffer_size": 64,
        "limit": limit,
        "rank": 0,
        "world_size": 1,
    }

    # Make sure in the following that we compare apples to apples, meaning in that case
    # that we materialize the payloads in the python scope in the expected format
    # (PIL.Image for images and masks for instance, numpy arrays for latents)
    datago_dataset = DatagoIterDataset(client_config, return_python_types=True)
    start = time.time()  # Note that the datago dataset will start walking the filesystem at construction time

    img = None
    for sample in tqdm(datago_dataset, dynamic_ncols=True):
        assert sample["id"] != ""
        img = sample["image"]

    fps = limit / (time.time() - start)
    print(f"Datago FPS {fps:.2f}")
    del datago_dataset

    # Save the last image as a test
    assert img is not None, "No image - benchmark did not run"
    img.save("benchmark_last_image.png")

    # Let's compare against a classic pytorch dataloader
    if compare_torch:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        print("Benchmarking torch dataloader")
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
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=lambda x: x
        )

        # Iterate over the DataLoader
        start = time.time()
        n_images = 0
        for batch in dataloader:
            n_images += len(batch)
            if n_images > limit:
                break
        fps = n_images / (time.time() - start)
        print(f"Torch FPS {fps:.2f}")


if __name__ == "__main__":
    typer.run(benchmark)
