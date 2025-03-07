from PIL import Image
from typing import Optional
import numpy as np


def uint8_array_to_numpy(raw_array):
    if len(raw_array.data) == 0:
        return None

    # By convention, arrays which are already serialized as jpg or png are not reshaped
    # We export them from Go with a Channels dimension of -1 to mark them as dimensionless.
    # Anything else is a valid number of channels and will thus lead to a reshape
    num_final_channels = max(raw_array.channels, 1)
    bit_depth = getattr(raw_array, "bit_depth", 8)

    length = (
        raw_array.width * raw_array.height * num_final_channels * bit_depth
        if raw_array.channels > 0
        else len(raw_array.data)
    )
    shape = (
        (raw_array.height, raw_array.width, raw_array.channels)
        if raw_array.channels > 0
        else (length,)
    )

    # Wrap the buffer around to create a numpy array. Strangely, shape needs to be passed twice
    # This is a zero-copy operation
    return np.frombuffer(raw_array.data, dtype=np.uint8).reshape(shape)


def raw_array_to_numpy(raw_array) -> Optional[np.ndarray]:
    if len(raw_array.data) == 0:
        return None

    # Generic numpy-serialized array
    try:
        return np.load(raw_array.data, allow_pickle=False)
    except ValueError:
        # Do not try to handle these, return None and we'll handle it in the caller
        print("Could not deserialize numpy array")
        return None


def raw_array_to_pil_image(raw_array) -> Optional[Image.Image]:
    if len(raw_array.data) == 0:
        return None

    if raw_array.channels <= 0:
        # Do not try to decode, we have a jpg or png buffer already
        return raw_array

    # Zero copy conversion of the image buffer from RAW to PIL.Image
    np_array = uint8_array_to_numpy(raw_array)
    h, w, c = np_array.shape

    # Greyscale image
    if c == 1:
        return Image.fromarray(np_array[:, :, 0], mode="L")

    if c == 4:
        return Image.frombuffer("RGBA", (w, h), np_array, "raw", "RGBA", 0, 1)

    assert c == 3, f"Expected 3 channels, got {c}"
    return Image.fromarray(np_array)
