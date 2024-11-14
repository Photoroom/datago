import ctypes
from PIL import Image
import io
from typing import Optional
import numpy as np
from datago import go


def uint8_array_to_numpy(go_array):
    # By convention, arrays which are already serialized as jpg or png are not reshaped
    # We export them from Go with a Channels dimension of -1 to mark them as dimensionless.
    # Anything else is a valid number of channels and will thus lead to a reshape
    num_final_channels = max(go_array.Channels, 1)
    length = (
        go_array.Width * go_array.Height * num_final_channels
        if go_array.Channels > 0
        else len(go_array.Data)
    )
    shape = (
        (go_array.Height, go_array.Width, go_array.Channels)
        if go_array.Channels > 0
        else (length,)
    )

    # Wrap the buffer around to create a numpy array. Strangely, shape needs to be passed twice
    # This is a zero-copy operation
    bytes_buffer = bytes(go.Slice_byte(go_array.Data))
    return np.frombuffer(bytes_buffer, dtype=np.uint8).reshape(shape)


def go_array_to_numpy(go_array) -> Optional[np.ndarray]:
    # Generic numpy-serialized array
    bytes_buffer = bytes(go.Slice_byte(go_array.Data))
    try:
        return np.load(bytes_buffer, allow_pickle=False)
    except ValueError:
        # Do not try to handle these, return None and we'll handle it in the caller
        print("Could not deserialize numpy array")
        return None


def go_array_to_pil_image(go_array):
    # Zero copy conversion of the image buffer from Go to PIL.Image
    np_array = uint8_array_to_numpy(go_array)
    if go_array.Channels <= 0:
        # Do not try to decode, we have a jpg or png buffer already
        return np_array

    h, w, c = np_array.shape

    # Greyscale image
    if c == 1:
        return Image.fromarray(np_array[:, :, 0], mode="L")

    if c == 4:
        return Image.frombuffer("RGBA", (w, h), np_array, "raw", "RGBA", 0, 1)

    assert c == 3, "Expected 3 channels"
    return Image.fromarray(np_array)
