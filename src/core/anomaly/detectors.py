import cv2
import numpy as np
from plantcv import plantcv as pcv
from .utils import _plant_mask, _save_debug, _label

def chlorosis_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    original_image, plant_mask = _plant_mask(image_path)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Threshold for healthy green and stressed yellow/brown
    green_healthy_mask = cv2.bitwise_and(cv2.inRange(hsv_image, (35, 40, 40), (85, 255, 255)), plant_mask)
    stress_yellow_mask = cv2.bitwise_and(cv2.inRange(hsv_image, (10, 40, 40), (35, 255, 255)), plant_mask)

    green_pixel_count = cv2.countNonZero(green_healthy_mask)
    stress_pixel_count = cv2.countNonZero(stress_yellow_mask)
    total_plant_pixels = green_pixel_count + stress_pixel_count
    
    stress_ratio = (stress_pixel_count / total_plant_pixels * 100) if total_plant_pixels > 0 else 0
    calculated_score = min(round(stress_ratio * 2), 100)

    # Create visualization images: dimmed background, color-coded status
    # Healthy (green) vs Stressed (red)
    visualization_image = (original_image * 0.3).astype(np.uint8)
    visualization_image[green_healthy_mask > 0] = (0, 220, 0)
    visualization_image[stress_yellow_mask > 0] = (0, 0, 255)
    
    _label(visualization_image, 10, 30, f"Chlorosis Score: {calculated_score}/100 | Green: {green_pixel_count}px  Stress: {stress_pixel_count}px", (0, 0, 0))
    debug_path = _save_debug(visualization_image, "chlorosis", output_dir)

    return {
        "score": calculated_score, 
        "green_px": green_pixel_count, 
        "stress_px": stress_pixel_count, 
        "debug_path": debug_path
    }


