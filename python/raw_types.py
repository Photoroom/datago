from PIL import Image
from typing import Optional, Union
import numpy as np
from datago import ImagePayload


def raw_array_to_numpy(raw_array: ImagePayload) -> Optional[np.ndarray]:
    if len(raw_array.data) == 0:
        return None

    # Generic numpy-serialized array
    try:
        return np.load(raw_array.data, allow_pickle=False)
    except ValueError:
        # Do not try to handle these, return None and we'll handle it in the caller
        print("Could not deserialize numpy array")
        return None


def decode_image_payload(payload: ImagePayload) -> Image.Image:
    """
    Decode an ImagePayload (encoded image) into a PIL Image.
    This is the proper way to decode encoded images for API users.
    """
    import io

    return Image.open(io.BytesIO(payload.data))


def get_image_mode(image_or_payload: Union[ImagePayload, Image.Image]) -> str:
    """
    Helper function to get the mode of an image, whether it's a PIL Image or ImagePayload.
    For ImagePayload objects (encoded images), we need to decode them first.
    """
    if hasattr(image_or_payload, "mode"):
        # It's a PIL Image
        return image_or_payload.mode
    else:
        # It's an ImagePayload (encoded image), decode it first
        return decode_image_payload(image_or_payload).mode
