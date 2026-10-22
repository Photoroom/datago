"""Break the file-serving benchmark down by consumer stage.

The point is to separate the Rust pipeline (decode + channel handoff) from the
Python-side materialization, which runs on a single thread while holding the GIL.

Stages:
  raw     - pull raw samples only (measures the Rust pipeline)
  numpy   - pull raw samples and build a read-only numpy view (zero-copy)
  pil     - pull raw samples and build a PIL image (one pixel copy)
  convert - get_sample_auto_convert(): full dict + PIL path used by the benchmark

Run:
  DATAGO_TEST_FILESYSTEM=$DATA uv run --python 3.14 --group dev python/bench_stages.py --limit 2000
"""

import json
import os
import time

import typer
from benchmark_defaults import IMAGE_CONFIG
from datago import DatagoClient


def measure(
    root_path: str,
    limit: int,
    num_workers: int,
    stage: str,
    crop_and_resize: bool = False,
) -> tuple[int, float]:
    # The benchmark controls datago's internal task pool through this env var.
    os.environ["DATAGO_MAX_TASKS"] = str(num_workers)

    client_config = {
        "source_type": "file",
        "source_config": {"root_path": root_path, "rank": 0, "world_size": 1},
        "prefetch_buffer_size": 256,
        "samples_buffer_size": 256,
        "limit": limit,
    }
    if crop_and_resize:
        client_config["image_config"] = {
            **IMAGE_CONFIG,
            "min_aspect_ratio": 1.0,
            "max_aspect_ratio": 1.0,
        }
    # Match the native example: include construction/startup, exclude cleanup.
    start = time.perf_counter()
    client = DatagoClient(json.dumps(client_config))
    client.start()
    count = 0
    try:
        for _ in range(limit):
            if stage == "convert":
                sample = client.get_sample_auto_convert()
            else:
                sample = client.get_sample()
                if sample is not None and stage == "numpy":
                    sample.image.to_numpy_array()
                elif sample is not None and stage == "pil":
                    sample.image.to_pil_image()
            if sample is None:
                break
            count += 1

        elapsed = time.perf_counter() - start
    finally:
        client.stop()
    if count != limit:
        raise ValueError(f"Expected {limit} samples, got {count}")
    return count, count / elapsed


def main(
    root_path: str = typer.Option(os.getenv("DATAGO_TEST_FILESYSTEM", "")),
    limit: int = typer.Option(2000, help="The number of samples to test on"),
    stages: str = typer.Option("raw,numpy,pil,convert"),
    workers: str = typer.Option("1,2,4,8,16"),
    crop_and_resize: bool = typer.Option(
        False, help="Resize and center-crop to 1024 x 1024"
    ),
):
    if limit < 1:
        raise typer.BadParameter("limit must be positive")
    stage_list = [stage.strip() for stage in stages.split(",")]
    if any(stage not in {"raw", "numpy", "pil", "convert"} for stage in stage_list):
        raise typer.BadParameter("stages must be raw, numpy, pil, or convert")
    try:
        worker_list = [int(value) for value in workers.split(",")]
    except ValueError as error:
        raise typer.BadParameter("workers must be comma-separated integers") from error
    if any(value < 0 for value in worker_list):
        raise typer.BadParameter("workers must be nonnegative")
    for num_workers in worker_list:
        for stage in stage_list:
            count, fps = measure(root_path, limit, num_workers, stage, crop_and_resize)
            print(
                f"{stage:8s} workers={num_workers:3d}  fps={fps:8.1f}  count={count}",
                flush=True,
            )


if __name__ == "__main__":
    typer.run(main)
