"""
anomaly/engine.py — Motor de Detección de Anomalías
==================================================
Orquestador que ejecuta todos los detectores y calcula el score compuesto.
"""

import os

from .detectors import chlorosis_score
from ..config import AnomalyThresholds
from ..vision_extractor import calculate_living_canopy


def calculate_anomaly_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    """
    Ejecuta análisis completo de anomalías en una imagen.
    
    Args:
        image_path: Ruta a la imagen de entrada.
        output_dir: Directorio para mapas de debug.
    
    Returns:
        dict: Scores individuales + score compuesto + paths de debug.
    """
    chlor = chlorosis_score(image_path, output_dir)
    
    canopy_res = calculate_living_canopy(
        image_path,
        save_map=True,
        plant_id=f"anomaly_{os.path.basename(image_path).split('.')[0]}",
        output_dir=output_dir
    )
    
    if isinstance(canopy_res, tuple):
        canopy_coverage, canopy_map_info = canopy_res
        canopy_map_path = canopy_map_info["map_path"]
    else:
        canopy_coverage = canopy_res or 0.0
        canopy_map_path = None

    # Composite refleja clorosis (weighted 100%)
    composite = chlor["score"]
    needs_attention = composite > AnomalyThresholds.CHLOROSIS_ATTENTION_THRESHOLD

    return {
        "anomaly_score": composite,
        "needs_attention": needs_attention,
        "canopy_coverage": canopy_coverage,
        "canopy_map": canopy_map_path,
        "chlorosis": chlor,
        "debug_dir": output_dir,
    }
