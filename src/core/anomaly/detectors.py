import cv2
import numpy as np
from plantcv import plantcv as pcv
from .utils import _plant_mask, _save_debug, _label

def chlorosis_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    img, mask = _plant_mask(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green = cv2.bitwise_and(cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)), mask)
    stress = cv2.bitwise_and(cv2.inRange(hsv, (10, 40, 40), (35, 255, 255)), mask)

    g_px, s_px = cv2.countNonZero(green), cv2.countNonZero(stress)
    total = g_px + s_px
    ratio = (s_px / total * 100) if total > 0 else 0
    score = min(round(ratio * 2), 100)

    # Debug: green=healthy, red=stressed, dimmed=background
    debug = (img * 0.3).astype(np.uint8)
    debug[green > 0] = (0, 220, 0)
    debug[stress > 0] = (0, 0, 255)
    _label(debug, 10, 30, f"Chlorosis Score: {score}/100 | Green: {g_px}px  Stress: {s_px}px", (0, 0, 0))
    path = _save_debug(debug, "chlorosis", output_dir)

    return {"score": score, "green_px": g_px, "stress_px": s_px, "debug_path": path}


