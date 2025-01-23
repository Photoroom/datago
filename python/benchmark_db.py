from datago import datago  # type: ignore
import time
from tqdm import tqdm
import numpy as np
from go_types import go_array_to_pil_image, go_array_to_numpy
import typer
import json


def benchmark(
    source: str = typer.Option("DATAGO_TEST_SOURCE", help="The source to test out"),
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(
        True, help="Crop and resize the images on the fly"
    ),
    require_images: bool = typer.Option(True, help="Request the original images"),
    require_embeddings: bool = typer.Option(False, help="Request embeddings"),
    test_masks: bool = typer.Option(True, help="Test masks"),
):
    print(f"Running benchmark for {source} - {limit} samples")
    client_config = {
        "source_type": datago.SourceTypeDB,
        "source_config": {
            "page_size": 512,
            "sources": source,
            "require_images": require_images,
            "require_embeddings": require_embeddings,
            "has_masks": "segmentation_mask" if test_masks else "",
            "rank": 0,
            "world_size": 1,
        },
        "image_config": {
            "crop_and_resize": crop_and_resize,
            "default_image_size": 512,
            "downsampling_ratio": 16,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
            "pre_encode_images": False,
        },
        "prefetch_buffer_size": 128,
        "samples_buffer_size": 64,
        "limit": limit,
    }

    client = datago.GetClientFromJSON(json.dumps(client_config))
    client.Start()  # Optional, but good practice to start the client to reduce latency to first sample (while you're instantiating models for instance)
    start = time.time()

    # Make sure in the following that we compare apples to apples, meaning in that case
    # that we materialize the payloads in the python scope in the expected format
    # (PIL.Image for images and masks for instance, numpy arrays for latents)
    img, mask, masked_image = None, None, None
    for _ in tqdm(range(limit), dynamic_ncols=True):
        sample = client.GetSample()
        if sample.ID:
            # Bring the masks and image to PIL
            if hasattr(sample, "Image"):
                img = go_array_to_pil_image(sample.Image)

            if hasattr(sample, "Masks"):
                for _, mask_buffer in sample.Masks.items():
                    mask = go_array_to_pil_image(mask_buffer)

            if (
                hasattr(sample, "AdditionalImages")
                and "masked_image" in sample.AdditionalImages
            ):
                masked_image = go_array_to_pil_image(
                    sample.AdditionalImages["masked_image"]
                )

            # Bring the latents to numpy
            if hasattr(sample, "Latents"):
                for _, latent_buffer in sample.Latents.items():
                    _latents = go_array_to_numpy(latent_buffer)

            # Bring the embeddings to numpy
            if hasattr(sample, "CocaEmbedding"):
                _embedding = np.array(sample.CocaEmbedding)

    fps = limit / (time.time() - start)
    print(f"FPS {fps:.2f}")
    client.Stop()

    # Save the last image as a test
    if img is not None:
        img.save("benchmark_last_image.png")

    if mask is not None:
        mask.save("benchmark_last_mask.png")

    if masked_image is not None:
        masked_image.save("benchmark_last_masked_image.png")


if __name__ == "__main__":
    typer.run(benchmark)
