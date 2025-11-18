#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from datago import ImagePayload
from PIL import Image
import numpy as np

def test_pil_conversion():
    """Test the new PIL image conversion methods"""

    # Test 1: Raw RGB image
    print("Test 1: Raw RGB image conversion")
    width, height = 100, 50
    channels = 3

    # Create a simple test image (red gradient)
    data = bytearray()
    for y in range(height):
        for x in range(width):
            # Red gradient from left to right
            red = int((x / width) * 255)
            data.extend([red, 0, 0])  # R, G, B

    payload = ImagePayload()
    payload.data = data
    payload.width = width
    payload.height = height
    payload.channels = channels
    payload.bit_depth = 8
    payload.is_encoded = False
    payload.original_width = width
    payload.original_height = height

    # Convert to PIL image
    pil_image = payload.to_pil_image()
    print(f"Created PIL image: {pil_image.mode}, {pil_image.size}")
    assert pil_image.mode == "RGB"
    assert pil_image.size == (width, height)

    # Verify the gradient
    first_pixel = pil_image.getpixel((0, 0))
    last_pixel = pil_image.getpixel((width-1, 0))
    print(f"First pixel: {first_pixel}, Last pixel: {last_pixel}")
    assert first_pixel[0] == 0  # First pixel should be black (0, 0, 0)
    # Due to integer division, the last pixel should be 252, not 255
    assert last_pixel[0] == 252  # Last pixel should be bright red (252, 0, 0)

    # Test 2: Raw greyscale image
    print("\nTest 2: Raw greyscale image conversion")
    width, height = 50, 50
    channels = 1

    # Create a simple test image (greyscale gradient)
    data = bytearray()
    for y in range(height):
        for x in range(width):
            # Greyscale gradient
            value = int(((x + y) / (width + height)) * 255)
            data.append(value)

    payload = ImagePayload()
    payload.data = data
    payload.width = width
    payload.height = height
    payload.channels = channels
    payload.bit_depth = 8
    payload.is_encoded = False
    payload.original_width = width
    payload.original_height = height

    # Convert to PIL image
    pil_image = payload.to_pil_image()
    print(f"Created PIL image: {pil_image.mode}, {pil_image.size}")
    assert pil_image.mode == "L"
    assert pil_image.size == (width, height)

    # Test 3: Encoded image (simulate JPEG/PNG)
    print("\nTest 3: Encoded image conversion")

    # Create a simple PIL image and encode it
    test_img = Image.new('RGB', (64, 64), color=(128, 128, 192))
    import io
    img_byte_arr = io.BytesIO()
    test_img.save(img_byte_arr, format='PNG')
    encoded_data = img_byte_arr.getvalue()

    payload = ImagePayload()
    payload.data = encoded_data
    payload.width = 64
    payload.height = 64
    payload.channels = -1  # Indicates encoded image
    payload.bit_depth = 8
    payload.is_encoded = True
    payload.original_width = 64
    payload.original_height = 64

    # Convert to PIL image
    pil_image = payload.to_pil_image()
    print(f"Created PIL image from encoded data: {pil_image.mode}, {pil_image.size}")
    assert pil_image.mode == "RGB"
    assert pil_image.size == (64, 64)

    # Verify the color
    middle_pixel = pil_image.getpixel((32, 32))
    print(f"Middle pixel: {middle_pixel}")
    assert middle_pixel == (128, 128, 192)

    # Test 4: NumPy array conversion
    print("\nTest 4: NumPy array conversion")
    width, height = 25, 25
    channels = 3

    # Create test data
    data = bytearray([min(x + y, 255) for y in range(height) for x in range(width) for _ in range(channels)])

    payload = ImagePayload()
    payload.data = data
    payload.width = width
    payload.height = height
    payload.channels = channels
    payload.bit_depth = 8
    payload.is_encoded = False
    payload.original_width = width
    payload.original_height = height

    # Convert to numpy array
    np_array = payload.to_numpy_array()
    print(f"Created numpy array: {np_array.shape}, {np_array.dtype}")
    assert np_array.shape == (height, width, channels)
    assert np_array.dtype == np.uint8

    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_pil_conversion()
