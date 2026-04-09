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


# ---------------------------------------------------------------------------
# Texture Anomaly — Deactivated but preserved
# ---------------------------------------------------------------------------
def texture_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    """
    img, mask = _plant_mask(image_path)
    
    # 1. Variance-based Texture (Expert Method for Mildew/Fuzz)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stdev = pcv.stdev_filter(img=gray, ksize=11)
    _, texture_thresh = cv2.threshold(stdev, 28, 255, cv2.THRESH_BINARY)
    texture_spots = cv2.bitwise_and(texture_thresh, mask)

    # 2. Color-based Anomaly (Specific Lesions / Mildew Patches)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pale_spots = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 140), (180, 115, 255)), mask)
    dark_spots = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 0), (180, 255, 60)), mask)
    
    # Combine and Clean (Removes edge noise/shadows)
    raw_anomaly = cv2.bitwise_or(texture_spots, cv2.bitwise_or(pale_spots, dark_spots))
    anomaly = pcv.fill(bin_img=raw_anomaly, size=15)

    plant_px = cv2.countNonZero(mask)
    anomaly_px = cv2.countNonZero(anomaly)
    pct = (anomaly_px / plant_px * 100) if plant_px > 0 else 0
    score = min(round(pct * 4), 100)

    # Debug: heatmap of anomalous spots
    debug = (img * 0.3).astype(np.uint8)
    debug[mask > 0] = (img * 0.6).astype(np.uint8)[mask > 0]
    debug[texture_spots > 0] = (255, 255, 0)   # Cyan = High variance
    debug[pale_spots > 0] = (255, 255, 255)    # White = Pale/White patches
    debug[dark_spots > 0] = (0, 0, 255)       # Red = Dark lesions
    _label(debug, 10, 30, f"Texture Score: {score}/100 | Anomaly: {anomaly_px}px ({pct:.1f}%)", (0, 0, 0))
    path = _save_debug(debug, "texture", output_dir)
    return {"score": score, "anomaly_px": anomaly_px, "anomaly_pct": round(pct, 2), "debug_path": path}
    """
    return {"score": 0, "anomaly_px": 0, "anomaly_pct": 0, "debug_path": None}


# ---------------------------------------------------------------------------
# Hole Detection — Deactivated but preserved
# ---------------------------------------------------------------------------
def hole_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    """
    img, mask = _plant_mask(image_path)
    rois = pcv.roi.auto_grid(mask=mask, nrows=6, ncols=4)
    labeled_mask, _ = pcv.create_labels(mask=mask, rois=rois, roi_type="partial")
    global_holes = np.zeros_like(mask)
    labels = np.unique(labeled_mask)
    for label in labels:
        if label == 0: continue
        local_mask = np.zeros_like(mask)
        local_mask[labeled_mask == label] = 255
        filled = pcv.fill_holes(local_mask)
        local_holes = cv2.bitwise_and(cv2.bitwise_not(local_mask), filled)
        global_holes = cv2.bitwise_or(global_holes, local_holes)
    contours, _ = cv2.findContours(global_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    real_holes = [c for c in contours if cv2.contourArea(c) > 30]
    hole_area = sum(cv2.contourArea(c) for c in real_holes)
    plant_area = cv2.countNonZero(mask)
    pct = (hole_area / plant_area * 100) if plant_area > 0 else 0
    score = min(round(pct * 10), 100)
    debug = (img * 0.4).astype(np.uint8)
    debug[mask > 0] = (img * 0.7).astype(np.uint8)[mask > 0]
    cv2.drawContours(debug, real_holes, -1, (255, 0, 255), -1)
    cv2.drawContours(debug, real_holes, -1, (255, 255, 255), 1)
    _label(debug, 10, 30, f"Hole Score: {score}/100 | {len(real_holes)} holes, {pct:.2f}% area", (0, 0, 0))
    path = _save_debug(debug, "holes", output_dir)
    return {"score": score, "hole_count": len(real_holes), "hole_area_pct": round(pct, 2), "debug_path": path}
    """
    return {"score": 0, "hole_count": 0, "hole_area_pct": 0, "debug_path": None}
