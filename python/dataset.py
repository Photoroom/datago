from datago import datago
import json
from typing import Dict, Any
from go_types import go_array_to_pil_image, go_array_to_numpy


class DatagoIterDataset:
    def __init__(self, datago_config: Dict[str, Any], return_python_types: bool = True):
        self.client = datago.GetClientFromJSON(json.dumps(datago_config))
        self.client.Start()
        self.return_python_types = return_python_types
        self.len = datago_config.get("limit", 1e9)
        print(self.len)
        print(datago_config)
        self.count = 0

    def __iter__(self):
        return self

    def __del__(self):
        self.client.Stop()

    def __len__(self):
        return self.len

    @staticmethod
    def to_python_types(item):
        if isinstance(item, datago.ImagePayload):
            return go_array_to_pil_image(item)
        elif isinstance(item, datago.LatentPayload):
            return go_array_to_numpy(item)
        elif isinstance(item, datago.Map_string_interface_):
            dict_item = dict(item)
            for key, value in filter(
                lambda x: isinstance(x[1], str) and x[1].startswith("%!s(float64"),
                dict_item.items(),
            ):
                dict_item[key] = float(value[12:-1])
            return dict_item
        elif isinstance(item, datago.go.Slice_string):
            return list(item)

        # TODO: Make this recursive, would be an elegant way of handling nested structures
        return item

    def __next__(self):
        self.count += 1
        if self.count > self.len:
            raise StopIteration

        sample = self.client.GetSample()
        if sample.ID == "":
            raise StopIteration

        if self.return_python_types:
            # Convert the Go types to Python types
            python_sample = {}
            for attr in filter(lambda x: "__" not in x, dir(sample)):
                python_sample[attr.lower()] = self.to_python_types(
                    getattr(sample, attr)
                )

            return python_sample

        return sample


if __name__ == "__main__":
    # Example config, using this for filesystem walkthrough would work just as well
    client_config = client_config = {
        "source_type": datago.SourceTypeDB,
        "source_config": {
            "page_size": 10,
            "sources": "COYO",
            "require_images": True,
            "has_attributes": "caption_moondream",
            "rank": 0,
            "world_size": 1,
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
    }
    dataset = DatagoIterDataset(client_config)
    for sample in dataset:
        print(sample)
