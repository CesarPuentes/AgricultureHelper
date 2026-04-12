"""
anomaly/detectors.py — Detectores de Anomalías
=============================================
Algoritmos de visión por computadora para detección de anomalías fisiológicas.
Ahora usa utils compartidas del núcleo.
"""

import numpy as np

from ..utils import save_debug_image, draw_label_pill
from ..config import AnomalyThresholds
from .utils import chlorosis_mask


def chlorosis_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    """
    Calcula el score de clorosis (0-100) basado en ratio de estrés.
    
    Args:
        image_path: Ruta a la imagen.
        output_dir: Directorio para guardar imagen de debug.
    
    Returns:
        dict: Score, conteo de píxeles, y path a imagen de debug.
    """
    original_image, green_mask, stress_mask = chlorosis_mask(image_path)

    green_pixel_count = np.count_nonzero(green_mask)
    stress_pixel_count = np.count_nonzero(stress_mask)
    total_plant_pixels = green_pixel_count + stress_pixel_count
    
    stress_ratio = (stress_pixel_count / total_plant_pixels * 100) if total_plant_pixels > 0 else 0
    calculated_score = min(
        round(stress_ratio * AnomalyThresholds.CHLOROSIS_SCALE_FACTOR),
        AnomalyThresholds.CHLOROSIS_MAX_SCORE
    )

    # Create visualization: dimmed background, color-coded status
    visualization_image = (original_image * 0.3).astype(np.uint8)
    visualization_image[green_mask > 0] = (0, 220, 0)
    visualization_image[stress_mask > 0] = (0, 0, 255)
    
    draw_label_pill(
        visualization_image, 10, 30,
        f"Chlorosis Score: {calculated_score}/100 | Green: {green_pixel_count}px  Stress: {stress_pixel_count}px",
        (0, 0, 0)
    )
    debug_path = save_debug_image(visualization_image, "chlorosis", output_dir)

    return {
        "score": calculated_score, 
        "green_px": green_pixel_count, 
        "stress_px": stress_pixel_count, 
        "debug_path": debug_path
    }
