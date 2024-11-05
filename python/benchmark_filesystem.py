from datago import datago  # type: ignore
import time
import typer
from tqdm import tqdm
from go_types import go_array_to_pil_image
import os
import json


def benchmark(
    root_path: str = typer.Option(
        os.getenv("DATAROOM_TEST_FILESYSTEM", ""), help="The source to test out"
    ),
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(
        True, help="Crop and resize the images on the fly"
    ),
    concurrency: int = typer.Option(32, help="The number of concurrent workers"),
    compare_torch: bool = typer.Option(True, help="Compare against torch dataloader"),
):
    print(f"Running benchmark for {root_path} - {limit} samples")
    client_config = {
        "source_type": datago.SourceTypeFileSystem,
        "source_config": {
            "page_size": 512,
            "root_path": root_path,
        },
        "image_config": {
            "crop_and_resize": crop_and_resize,
            "default_image_size": 512,
            "downsampling_ratio": 16,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
            "pre_encode_images": False,
        },
        "prefetch_buffer_size": 64,
        "samples_buffer_size": 128,
        "concurrency": concurrency,
    }

    client = datago.GetClientFromJSON(json.dumps(client_config))
    client.Start()
    start = time.time()

    # Make sure in the following that we compare apples to apples, meaning in that case
    # that we materialize the payloads in the python scope in the expected format
    # (PIL.Image for images and masks for instance, numpy arrays for latents)
    img = None
    for _ in tqdm(range(limit), dynamic_ncols=True):
        sample = client.GetSample()
        if sample.ID and hasattr(sample, "Image"):
            img = go_array_to_pil_image(sample.Image)

    fps = limit / (time.time() - start)
    print(f"Datago FPS {fps:.2f}")
    client.Stop()

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
                    transforms.Resize((512, 512)),
                ]
            )
            if crop_and_resize
            else None
        )

        # Create the ImageFolder dataset
        dataset = datasets.ImageFolder(root=root_path, transform=transform)

        # Create a DataLoader to load the images in batches
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=lambda x: x
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
