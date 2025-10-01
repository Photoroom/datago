import json
import time

import numpy as np
import typer
from benchmark_defaults import IMAGE_CONFIG
from datago import DatagoClient  # type: ignore
from PIL import Image
from raw_types import raw_array_to_numpy, raw_array_to_pil_image
from tqdm import tqdm


def benchmark(
    source: str = typer.Option("DATAGO_TEST_SOURCE", help="The source to test out"),
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(
        True, help="Crop and resize the images on the fly"
    ),
    require_images: bool = typer.Option(True, help="Request the original images"),
    require_embeddings: bool = typer.Option(False, help="Request embeddings"),
    test_masks: bool = typer.Option(True, help="Test masks"),
    encode_images: bool = typer.Option(True, help="Re-compress all the images in PNG"),
):
    print(f"Running benchmark for {source} - {limit} samples")
    client_config = {
        "source_type": "db",
        "source_config": {
            "page_size": 512,
            "sources": source,
            "require_images": require_images,
            "require_embeddings": require_embeddings,
            "has_masks": "segmentation_mask" if test_masks else "",
            "rank": 0,
            "world_size": 1,
        },
        "prefetch_buffer_size": 128,
        "samples_buffer_size": 64,
        "limit": limit,
    }

    if crop_and_resize or encode_images:
        client_config["image_config"] = IMAGE_CONFIG

    if encode_images:
        client_config["image_config"]["crop_and_resize"] = (  # type: ignore
            crop_and_resize  # You may want to encode images without resizing them
        )
        client_config["image_config"]["pre_encode_images"] = True  # type: ignore

    client = DatagoClient(json.dumps(client_config))
    client.start()  # Optional, but good practice to start the client to reduce latency to first sample (while you're instantiating models for instance)
    start = time.time()

    # Make sure in the following that we compare apples to apples, meaning in that case
    # that we materialize the payloads in the python scope in the expected format
    # (PIL.Image for images and masks for instance, numpy arrays for latents)
    img, mask, masked_image = None, None, None
    for _ in tqdm(range(limit), dynamic_ncols=True):
        sample = client.get_sample()
        if sample.id:
            # Bring the masks and image to PIL
            if hasattr(sample, "image"):
                img = raw_array_to_pil_image(sample.image)

            if hasattr(sample, "masks"):
                for _, mask_buffer in sample.masks.items():
                    mask = raw_array_to_pil_image(mask_buffer)

            if (
                hasattr(sample, "additional_images")
                and "masked_image" in sample.additional_images
            ):
                masked_image = raw_array_to_pil_image(
                    sample.AdditionalImages["masked_image"]
                )

            # Bring the latents to numpy
            if hasattr(sample, "latents"):
                for _, latent_buffer in sample.latents.items():
                    _latents = raw_array_to_numpy(latent_buffer)

            # Bring the embeddings to numpy
            if hasattr(sample, "coca_embedding"):
                _embedding = np.array(sample.coca_embedding)

    fps = limit / (time.time() - start)
    print(f"FPS {fps:.2f}")
    client.stop()

    def save_img(img, path):
        if isinstance(img, Image.Image):
            img.save(path)
        else:
            # save the raw array to disk
            with open(path, "wb") as f:
                f.write(img.data)

    # Save the last image as a test
    if img is not None:
        save_img(img, "benchmark_last_image.png")

    if mask is not None:
        save_img(mask, "benchmark_last_mask.png")

    if masked_image is not None:
        save_img(masked_image, "benchmark_last_masked_img.png")


if __name__ == "__main__":
    typer.run(benchmark)
