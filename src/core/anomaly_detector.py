"""
anomaly_detector.py — Detección de Anomalías por Visión Computacional
---------------------------------------------------------------------
Three pure-OpenCV techniques that flag whether a plant needs attention
without naming the disease. Each returns a score 0–100.

  1. Chlorosis Ratio   — Yellow/brown vs green tissue
  2. Texture Anomaly   — Non-green spots on leaf surface
  3. Hole Detection    — Internal holes from pest chewing
"""

import os
from datetime import datetime

import cv2
import numpy as np
from plantcv import plantcv as pcv

from src.core.vision_extractor import calculate_living_canopy

pcv.params.debug = "None"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Chlorosis Ratio — Stressed (yellow/brown) vs healthy green
# ---------------------------------------------------------------------------

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
# 2. Texture Anomaly — Non-green spots on leaf surface (white mildew, dark lesions)
# ---------------------------------------------------------------------------

def texture_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    img, mask = _plant_mask(image_path)
    
    # 1. Variance-based Texture (Expert Method for Mildew/Fuzz)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Lowered threshold (28) to detect subtle mildew texture
    stdev = pcv.stdev_filter(img=gray, ksize=11)
    _, texture_thresh = cv2.threshold(stdev, 28, 255, cv2.THRESH_BINARY)
    texture_spots = cv2.bitwise_and(texture_thresh, mask)

    # 2. Color-based Anomaly (Specific Lesions / Mildew Patches)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # White/Pale spots (Mildew): Saturation up to 115, Value floor 140
    pale_spots = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 140), (180, 115, 255)), mask)
    # Dark necrotic lesions
    dark_spots = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 0), (180, 255, 60)), mask)
    
    # Combine and Clean (Removes edge noise/shadows)
    raw_anomaly = cv2.bitwise_or(texture_spots, cv2.bitwise_or(pale_spots, dark_spots))
    anomaly = pcv.fill(bin_img=raw_anomaly, size=15)

    plant_px = cv2.countNonZero(mask)
    anomaly_px = cv2.countNonZero(anomaly)
    pct = (anomaly_px / plant_px * 100) if plant_px > 0 else 0
    
    # Scaling: 20% anomalous area is a high score (80)
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


# ---------------------------------------------------------------------------
# 3. Hole Detection — Internal contours from pest chewing
# ---------------------------------------------------------------------------

def hole_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    img, mask = _plant_mask(image_path)

    # 1. Define ROI Grid (Standard 6x4 tray)
    # Using pcv.roi.auto_grid to define the 24 pots
    rois = pcv.roi.auto_grid(mask=mask, nrows=6, ncols=4)
    
    # 2. Isolate individual rosettes
    # pcv.create_labels segments the global mask into 24 distinct clusters based on these ROIs.
    # This is crucial: Gaps BETWEEN rosettes will now be outside the cluster boundaries
    # and won't be filled by the topological algorithm.
    labeled_mask, _ = pcv.create_labels(mask=mask, rois=rois, roi_type="partial")
    
    global_holes = np.zeros_like(mask)
    labels = np.unique(labeled_mask)

    # 3. Analyze each plant individually for internal holes
    for label in labels:
        if label == 0: continue  # Skip background
        
        # Create a local mask for a single pot/plant cluster
        local_mask = np.zeros_like(mask)
        local_mask[labeled_mask == label] = 255
        
        # Topological fill strictly finds internal holes within THIS plant's geometry
        filled = pcv.fill_holes(local_mask)
        local_holes = cv2.bitwise_and(cv2.bitwise_not(local_mask), filled)
        
        # Aggregate back to global holes mask
        global_holes = cv2.bitwise_or(global_holes, local_holes)

    # 4. Filter and Quantify
    contours, _ = cv2.findContours(global_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filtering small noise; real pest damage usually targets area > 30px
    real_holes = [c for c in contours if cv2.contourArea(c) > 30]
    hole_area = sum(cv2.contourArea(c) for c in real_holes)
    
    plant_area = cv2.countNonZero(mask)
    pct = (hole_area / plant_area * 100) if plant_area > 0 else 0
    # Scaling: 5% total leaf area loss is a very high score (50)
    score = min(round(pct * 10), 100)

    # Debug: magenta holes on dimmed image
    debug = (img * 0.4).astype(np.uint8)
    debug[mask > 0] = (img * 0.7).astype(np.uint8)[mask > 0]
    cv2.drawContours(debug, real_holes, -1, (255, 0, 255), -1)  # Fill holes
    cv2.drawContours(debug, real_holes, -1, (255, 255, 255), 1)  # Outline
    _label(debug, 10, 30, f"Hole Score: {score}/100 | {len(real_holes)} holes, {pct:.2f}% area", (0, 0, 0))
    path = _save_debug(debug, "holes", output_dir)

    return {"score": score, "hole_count": len(real_holes), "hole_area_pct": round(pct, 2), "debug_path": path}



# ---------------------------------------------------------------------------
# Composite — Single-image anomaly assessment
# ---------------------------------------------------------------------------

def calculate_anomaly_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    """
    Run chlorosis + texture + holes on one image.
    Returns individual scores + weighted composite (0–100).
    """
    chlor = chlorosis_score(image_path, output_dir)
    # text = texture_score(image_path, output_dir=output_dir)
    # holes = hole_score(image_path, output_dir)
    
    # Deactivated as per user request
    text = {"score": 0, "anomaly_px": 0, "anomaly_pct": 0, "debug_path": None}
    holes = {"score": 0, "hole_count": 0, "hole_area_pct": 0, "debug_path": None}
    
    canopy_res = calculate_living_canopy(image_path, save_map=True, plant_id=f"anomaly_{os.path.basename(image_path).split('.')[0]}", output_dir=output_dir)
    if isinstance(canopy_res, tuple):
        canopy_coverage, canopy_map_info = canopy_res
        canopy_map_path = canopy_map_info["map_path"]
    else:
        canopy_coverage = canopy_res or 0.0
        canopy_map_path = None

    # Composite now only reflects chlorosis (weighted 100%)
    composite = chlor["score"]
    needs_attention = composite > 40

    return {
        "anomaly_score": composite,
        "needs_attention": needs_attention,
        "canopy_coverage": canopy_coverage,
        "canopy_map": canopy_map_path,
        "chlorosis": chlor,
        "texture": text,
        "holes": holes,
        "debug_dir": output_dir,
    }
