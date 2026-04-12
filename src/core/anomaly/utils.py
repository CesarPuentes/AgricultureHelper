"""
anomaly/utils.py — Utilidades de Detección de Anomalías
======================================================
Funciones de bajo nivel para algoritmos de detección de anomalías.
Ahora usa utils compartidas del núcleo para máscaras y debugging.
"""

import cv2
import numpy as np
from plantcv import plantcv as pcv

from ..utils import create_lab_mask, draw_label_pill, save_debug_image
from ..config import AnomalyThresholds


def chlorosis_mask(image_path: str):
    """
    Calcula máscaras para tejido verde saludable y estresado (clorosis).
    
    Args:
        image_path: Ruta a la imagen.
    
    Returns:
        tuple: (original_image, green_mask, stress_mask)
    """
    original_image, plant_mask = create_lab_mask(image_path)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Threshold for healthy green and stressed yellow/brown
    green_healthy_mask = cv2.bitwise_and(
        cv2.inRange(hsv_image, AnomalyThresholds.HEALTHY_GREEN_LOWER, AnomalyThresholds.HEALTHY_GREEN_UPPER),
        plant_mask
    )
    stress_yellow_mask = cv2.bitwise_and(
        cv2.inRange(hsv_image, AnomalyThresholds.STRESS_YELLOW_LOWER, AnomalyThresholds.STRESS_YELLOW_UPPER),
        plant_mask
    )

    return original_image, green_healthy_mask, stress_yellow_mask
