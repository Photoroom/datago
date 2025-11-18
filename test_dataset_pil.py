#!/usr/bin/env python3

import tempfile
import os
from PIL import Image
import sys
import json

# Add the current directory to Python path so we can import datago
sys.path.insert(0, '/home/lefaudeux/Git/datago')

try:
    from datago import DatagoIterDataset, initialize_logging

    # Create a temporary directory with test images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test image
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        img_path = os.path.join(tmpdir, 'test.png')
        img.save(img_path)

        # Create dataset config
        config = {
            'source_type': 'file',
            'source_config': {
                'root_path': tmpdir,
                'rank': 0,
                'world_size': 1,
                'random_sampling': False,
            },
            'limit': 1,
            'samples_buffer_size': 10,
        }

        # Test the dataset
        initialize_logging('warn')
        dataset = DatagoIterDataset(config)

        for sample in dataset:
            print('✓ Sample received from dataset')
            print(f'Sample ID: {sample["id"]}')

            # Test image access - should be automatic now
            image = sample["image"]
            print(f'Image type: {type(image)}')

            # Test accessing PIL attributes
            try:
                mode = image.mode
                size = image.size
                format = image.format
                print(f'✓ PIL Image attributes accessible:')
                print(f'  - mode: {mode}')
                print(f'  - size: {size}')
                print(f'  - format: {format}')
                print('✓ SUCCESS: Dataset automatically returns PIL Images!')
            except Exception as e:
                print(f'✗ ERROR accessing PIL attributes: {e}')
                import traceback
                traceback.print_exc()
            break

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
