import numpy as np
import cv2
from plantcv import plantcv as pcv

# Disable interactive plotting for headless/Raspberry Pi environments
pcv.params.debug = "None"

# Minimum pixel area to consider a contour as a real plant (not noise)
_MIN_PLANT_AREA = 150


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


def _get_plant_contours(mask):
    """Return external contours whose area ≥ _MIN_PLANT_AREA."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= _MIN_PLANT_AREA]


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


def count_plants(image_path):
    """
    Estimate the number of individual plants using contour-based area division.

    Strategy:
      1. Extract green mask → find external contours.
      2. Use the 75th percentile of contour areas as the typical single-plant area.
      3. Total green area ÷ typical plant area → estimated plant count.

    Returns the estimated count (int), or None on failure.
    """
    try:
        _, mask = _create_green_mask(image_path)
        plant_contours = _get_plant_contours(mask)

        if not plant_contours:
            return 0

        areas = [a for _, a in plant_contours]
        typical_plant_area = np.percentile(areas, 75)
        total_green_area = sum(areas)

        return max(1, round(total_green_area / typical_plant_area))

    except Exception as e:
        print(f"Error counting plants: {e}")
        return None


def measure_plant_sizes(image_path):
    """
    Detect individual plant contours and measure each one's size.

    Returns a list of dicts with keys:
      label, area, height, width, perimeter, solidity.

    Large merged contours are reported as-is (one entry per contour).
    Returns None on failure.
    """
    try:
        img, mask = _create_green_mask(image_path)
        plant_contours = _get_plant_contours(mask)

        if not plant_contours:
            return []

        plants = []
        for i, (cnt, area) in enumerate(plant_contours, start=1):
            _, _, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, closed=True)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = (area / hull_area) if hull_area > 0 else 0

            plants.append({
                "label": f"plant_{i}",
                "area": float(area),
                "height": float(h),
                "width": float(w),
                "perimeter": round(perimeter, 1),
                "solidity": round(solidity, 3),
            })

        return plants

    except Exception as e:
        print(f"Error measuring plant sizes: {e}")
        return None
