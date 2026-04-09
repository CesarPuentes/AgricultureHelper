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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Spots that are ON the plant but NOT green and NOT yellow (already counted in chlorosis)
    # This catches: white mildew, dark necrotic lesions, grey mold
    white_spots = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 160), (180, 50, 255)), mask)
    dark_spots = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 0), (180, 255, 60)), mask)
    anomaly = cv2.bitwise_or(white_spots, dark_spots)

    plant_px = cv2.countNonZero(mask)
    anomaly_px = cv2.countNonZero(anomaly)
    pct = (anomaly_px / plant_px * 100) if plant_px > 0 else 0
    score = min(round(pct * 5), 100)  # 20% anomalous → score 100

    # Debug: heatmap of anomalous spots
    debug = (img * 0.3).astype(np.uint8)
    debug[mask > 0] = (img * 0.6).astype(np.uint8)[mask > 0]
    debug[white_spots > 0] = (255, 255, 0)   # Cyan = white mildew
    debug[dark_spots > 0] = (0, 0, 255)       # Red = dark lesions
    _label(debug, 10, 30, f"Texture Score: {score}/100 | Anomaly: {anomaly_px}px ({pct:.1f}%)", (0, 0, 0))
    path = _save_debug(debug, "texture", output_dir)

    return {"score": score, "anomaly_px": anomaly_px, "anomaly_pct": round(pct, 2), "debug_path": path}


# ---------------------------------------------------------------------------
# 3. Hole Detection — Internal contours from pest chewing
# ---------------------------------------------------------------------------

def hole_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    img, mask = _plant_mask(image_path)

    # Close small gaps → solid shape, then XOR to find internal holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    holes = cv2.bitwise_and(cv2.bitwise_not(mask), closed)

    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    real_holes = [c for c in contours if cv2.contourArea(c) > 50]
    hole_area = sum(cv2.contourArea(c) for c in real_holes)
    plant_area = cv2.countNonZero(mask)
    pct = (hole_area / plant_area * 100) if plant_area > 0 else 0
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
    text = texture_score(image_path, output_dir=output_dir)
    holes = hole_score(image_path, output_dir)

    composite = round(chlor["score"] * 0.40 + text["score"] * 0.40 + holes["score"] * 0.20)
    needs_attention = composite > 40

    return {
        "anomaly_score": composite,
        "needs_attention": needs_attention,
        "chlorosis": chlor,
        "texture": text,
        "holes": holes,
        "debug_dir": output_dir,
    }
