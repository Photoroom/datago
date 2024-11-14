from datago import datago  # type: ignore
import time
from tqdm import tqdm
import numpy as np
from go_types import go_array_to_pil_image, go_array_to_numpy
import typer


def benchmark(
    source: str = typer.Option("DATAGO_TEST_SOURCE", help="The source to test out"),
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    crop_and_resize: bool = typer.Option(
        True, help="Crop and resize the images on the fly"
    ),
    require_images: bool = typer.Option(True, help="Request the original images"),
    require_embeddings: bool = typer.Option(False, help="Request embeddings"),
    test_masks: bool = typer.Option(True, help="Test masks"),
    test_latents: bool = typer.Option(True, help="Test latents"),
):
    print(f"Running benchmark for {source} - {limit} samples")

    # Get a generic client config
    client_config = datago.GetDatagoConfig()
    client_config.ImageConfig.CropAndResize = crop_and_resize

    # Specify the source parameters as you see fit
    source_config = datago.GetSourceDBConfig()
    source_config.Sources = source
    source_config.RequireImages = require_images
    source_config.RequireEmbeddings = require_embeddings
    source_config.HasMasks = "segmentation_mask" if test_masks else ""
    source_config.HasLatents = "caption_latent_t5xxl" if test_latents else ""

    # Get a new client instance, happy benchmarking
    client_config.SourceConfig = source_config
    client = datago.GetClient(client_config)

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
