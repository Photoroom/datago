#!/usr/bin/env python3

import tempfile
import os
from PIL import Image
import sys
import json

# Add the current directory to Python path so we can import datago
sys.path.insert(0, '/home/lefaudeux/Git/datago')

try:
    from datago import DatagoClient, initialize_logging

    # Create a temporary directory with test images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test image
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        img_path = os.path.join(tmpdir, 'test.png')
        img.save(img_path)

        # Create client config
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

        # Test the client
        initialize_logging('warn')
        client = DatagoClient(json.dumps(config))
        client.start()

        sample = client.get_sample()
        if sample:
            print('✓ Sample received successfully')
            print(f'Sample ID: {sample.id}')

            # Test the automatic conversion approach
            print("\n=== Testing Automatic Conversion ===")

            # Method 1: Direct access (should work with __getattr__)
            try:
                mode = sample.image.mode
                print(f'✓ Direct access works: mode = {mode}')
            except Exception as e:
                print(f'✗ Direct access failed: {e}')

            # Method 2: Callable approach
            try:
                pil_image = sample.image()
                mode = pil_image.mode
                size = pil_image.size
                print(f'✓ Callable approach works: {type(pil_image)} - mode={mode}, size={size}')
            except Exception as e:
                print(f'✗ Callable approach failed: {e}')

            # Method 3: Manual conversion via get_payload
            try:
                payload = sample.image.get_payload()
                pil_from_payload = payload.to_pil_image()
                print(f'✓ Manual conversion works: {type(pil_from_payload)}')
            except Exception as e:
                print(f'✗ Manual conversion failed: {e}')

            print("\n=== Summary ===")
            print("The PythonImagePayload object provides multiple ways to access the PIL image:")
            print("1. sample.image() - Call the object to get PIL image directly")
            print("2. sample.image.get_payload().to_pil_image() - Manual conversion")
            print("3. Python dataset code can automatically call sample.image() for seamless experience")
        else:
            print('✗ No sample received')

        client.stop()

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
