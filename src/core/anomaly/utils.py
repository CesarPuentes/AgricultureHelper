import os
from datetime import datetime
import cv2
import numpy as np
from plantcv import plantcv as pcv

def _plant_mask(image_path: str):
    """Read image → PlantCV LAB-a segmentation → (BGR image, binary mask)."""
    img, _, _ = pcv.readimage(filename=image_path)
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    thresh = pcv.threshold.otsu(gray_img=a, object_type='dark')
    mask = pcv.fill(bin_img=thresh, size=150)
    return img, mask


def _save_debug(image, name: str, output_dir: str) -> str:
    """Save diagnostic image, return path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{name}_{ts}.png")
    cv2.imwrite(path, image)
    return path


def _label(img, x, y, text, color):
    """Draw a labeled pill on the image."""
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(img, (x, y - sz[1] - 6), (x + sz[0] + 8, y + 4), color, -1)
    cv2.putText(img, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
