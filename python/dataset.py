from datago import DatagoClient
import json
from typing import Dict, Any
from raw_types import raw_array_to_pil_image, raw_array_to_numpy


class DatagoIterDataset:
    def __init__(self, datago_config: Dict[str, Any], return_python_types: bool = True):
        self.client = DatagoClient(json.dumps(datago_config))
        self.client.start()
        self.return_python_types = return_python_types
        self.len = datago_config.get("limit", 1e9)
        print(self.len)
        print(datago_config)
        self.count = 0

    def __iter__(self):
        return self

    def __del__(self):
        self.client.stop()

    def __len__(self):
        return self.len

    @staticmethod
    def to_python_types(item, key):
        if key == "attributes":
            return json.loads(item)

        if isinstance(item, dict):
            # recurvisely convert the dictionary
            return {k: DatagoIterDataset.to_python_types(v, k) for k, v in item.items()}

        elif "image" in key:
            return raw_array_to_pil_image(item)
        elif "latent" in key:
            return raw_array_to_numpy(item)

        return item

    def __next__(self):
        self.count += 1
        if self.count > self.len:
            raise StopIteration

        sample = self.client.get_sample()
        if not sample or sample.id == "":
            raise StopIteration

        if self.return_python_types:
            # Convert the Rust types to Python types
            python_sample = {}
            for attr in filter(lambda x: "__" not in x, dir(sample)):
                python_sample[attr.lower()] = self.to_python_types(
                    getattr(sample, attr), attr
                )

            return python_sample

        return sample


if __name__ == "__main__":
    # Example config, using this for filesystem walkthrough would work just as well
    client_config = client_config = {
        "source_type": "db",
        "source_config": {
            "page_size": 10,
            "sources": "COYO",
            "require_images": True,
            "has_attributes": "caption_moondream",
        },
        "image_config": {
            "crop_and_resize": True,
            "default_image_size": 512,
            "downsampling_ratio": 16,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
            "pre_encode_images": False,
        },
        "prefetch_buffer_size": 64,
        "samples_buffer_size": 128,
        "limit": 10,
        "rank": 0,
        "world_size": 1,
    }
    dataset = DatagoIterDataset(client_config)
    for sample in dataset:
        print(sample)
