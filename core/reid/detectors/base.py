from pathlib import Path

import cv2
import numpy as np


def resolve_image(image):
    """Resolve an image input to a numpy array in cv2 BGR format."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image}")
        return img
    if isinstance(image, np.ndarray):
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")
