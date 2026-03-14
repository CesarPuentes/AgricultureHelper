import numpy as np
import cv2
from plantcv import plantcv as pcv

# Disable interactive plotting for headless/Raspberry Pi environments
pcv.params.debug = "None"


def _create_green_mask(image_path):
    """
    Shared pipeline: read image → ExG index → threshold → noise fill.
    Returns (original_color_image, binary_mask).
    """
    original_color_image, _, _ = pcv.readimage(filename=image_path)

    b, g, r = cv2.split(original_color_image)
    b, g, r = b.astype(float), g.astype(float), r.astype(float)

    exg = np.clip((2 * g) - r - b, 0, 255).astype(np.uint8)

    mask = pcv.threshold.binary(gray_img=exg, threshold=20, object_type="light")
    mask = pcv.fill(bin_img=mask, size=200)

    return original_color_image, mask


def calculate_living_canopy(image_path):
    """
    Calculate the percentage of the image covered by living green tissue.
    """
    try:
        _, mask = _create_green_mask(image_path)
        total_pixels = mask.size
        living_pixels = np.count_nonzero(mask)
        return round((living_pixels / total_pixels) * 100, 2)
    except Exception as e:
        print(f"Error processing camera image: {e}")
        return None
