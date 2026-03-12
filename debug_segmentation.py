"""
Debug script: contour-based counting with area-division.

Strategy:
  1. ExG → threshold → fill noise → external contours
  2. Estimate avg plant area from the mode of mid-range contours
  3. Small contours ≤ 1.5× avg = 1 plant
  4. Large contours = round(area / avg_plant_area)

Usage:  python debug_segmentation.py test_images/plants.png
Output: debug_output/
"""

import sys
import os
import numpy as np
import cv2
from plantcv import plantcv as pcv

pcv.params.debug = "None"

OUTPUT_DIR = "debug_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"  Saved → {path}")


def estimate_single_plant_area(areas):
    """
    Estimate the typical area of ONE plant.
    Use the 75th percentile of all contours — this avoids being
    pulled down by leaf fragments and up by merged clusters.
    """
    return np.percentile(areas, 75)


def debug_pipeline(image_path):
    print(f"\n{'='*60}")
    print(f"  Image: {image_path}")
    print(f"{'='*60}\n")

    img, _, _ = pcv.readimage(filename=image_path)
    save("1_original.png", img)

    # ── Mask ────────────────────────────────────────────────────
    b, g, r = cv2.split(img)
    b, g, r = b.astype(float), g.astype(float), r.astype(float)
    exg = np.clip((2 * g) - r - b, 0, 255).astype(np.uint8)
    save("2_exg_index.png", exg)

    mask = pcv.threshold.binary(gray_img=exg, threshold=20, object_type="light")
    mask = pcv.fill(bin_img=mask, size=200)
    save("3_mask.png", mask)

    # ── Contours ────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_plant = 150
    plant_contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_plant]
    areas = [a for _, a in plant_contours]

    avg_plant = estimate_single_plant_area(areas)
    total_green = sum(areas)
    approx_by_total_area = round(total_green / avg_plant)

    print(f"  Contours ≥ {min_plant}px: {len(plant_contours)}")
    print(f"  75th percentile area (single plant est.): {avg_plant:.0f} px")
    print(f"  Total green area: {total_green:.0f} px")
    print(f"  Quick estimate (total ÷ 75th pct): {approx_by_total_area}")

    # ── Count: per contour ──────────────────────────────────────
    cluster_threshold = avg_plant * 2
    total = 0
    viz = img.copy()

    for cnt, area in plant_contours:
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0

        if area <= cluster_threshold:
            total += 1
            cv2.drawContours(viz, [cnt], -1, (0, 255, 0), 2)
        else:
            n = max(1, round(area / avg_plant))
            total += n
            cv2.drawContours(viz, [cnt], -1, (0, 0, 255), 2)
            cv2.putText(viz, str(n), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    save("4_counted_plants.png", viz)
    print(f"\n  Cluster threshold (2× p75): {cluster_threshold:.0f} px")
    print(f"  ✅ ESTIMATED PLANT COUNT: {total}")
    print(f"     (green = single, red = cluster with sub-count)\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_images/plants.png"
    debug_pipeline(path)
