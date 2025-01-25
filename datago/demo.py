from datago import DatagoClient
import os
import json

config = {
    "source_config": {
        "sources": os.environ.get("DATAROOM_TEST_SOURCE", ""),
        "sources_ne": "",
        "require_images": True,
        "require_embeddings": True,
        "tags": "",
        "tags_ne": "",
        "has_attributes": "",
        "lacks_attributes": "",
        "has_masks": "",
        "lacks_masks": "",
        "has_latents": "",
        "lacks_latents": "",
        "min_short_edge": 0,
        "max_short_edge": 0,
        "min_pixel_count": -1,
        "max_pixel_count": -1,
        "duplicate_state": -1,
        "random_sampling": False,
        "page_size": 10,
    },
    "limit": 2,
    "rank": 0,
    "world_size": 1,
    "samples_buffer_size": 1,
}

client = DatagoClient(json.dumps(config))


sample = client.get_sample()
print(sample)
