"""Utility functions for encoding and decoding images to/from string formats.

This module provides a class `ImageStringEncoder` that facilitates the
conversion of NumPy image arrays to base64-encoded data URIs and vice-versa.
This is useful for transmitting image data over networks with other data.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

DATA_URI_PREFIX = "data:image/png;base64,"


class ImageStringEncoder:
    """Class for encoding and decoding images as data URIS."""

    @staticmethod
    def encode_image(image: np.ndarray, cap_size: bool = False) -> str:
        """Encode numpy image array to base64 string for transmission.

        Converts numpy arrays to PNG format and encodes as base64. For remote
        endpoints, automatically resizes large images to 224x224 to meet
        payload size limits.

        Args:
            image: Numpy array representing an RGB image.

        Returns:
            Base64 encoded data URI string of the PNG image.
        """
        pil_image = Image.fromarray(image)
        if cap_size and pil_image.size > (224, 224):
            # There is a limit on the image size for non-local endpoints
            # This is OK as almost all algorithms scale to 224x224
            pil_image = pil_image.resize((224, 224))
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return DATA_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def decode_image(encoded_image: str) -> np.ndarray:
        """Decode base64 data URI image string back to numpy array.

        Args:
            encoded_image: Base64 encoded image string.

        Returns:
            Numpy array representing the decoded image.
        """
        img_bytes = base64.b64decode(encoded_image.removeprefix(DATA_URI_PREFIX))
        buffer = BytesIO(img_bytes)
        pil_image = Image.open(buffer)
        return np.array(pil_image)
