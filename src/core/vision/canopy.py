"""
vision/canopy.py — Cálculo de Cobertura Verde Viva
==================================================
Módulo dedicado al cálculo del porcentaje de tejido verde vivo
en imágenes de bandejas de plantas.
"""

import logging
from typing import Tuple

import numpy as np

from ..utils.image_utils import create_green_mask
from .diagnostic_map import generate_diagnostic_map

logger = logging.getLogger(__name__)


def calculate_canopy_coverage(image_path: str) -> float | None:
    """
    Calcula el porcentaje de cobertura verde sin generar mapa.
    
    Args:
        image_path: Ruta a la imagen.
    
    Returns:
        float: Porcentaje de cobertura (0-100), o None si hay error.
    """
    try:
        _, mask = create_green_mask(image_path)
        total_pixels = mask.size
        living_pixels = np.count_nonzero(mask)
        coverage = round((living_pixels / total_pixels) * 100, 2)
        return coverage
    except Exception as e:
        logger.error(f"Error calculating canopy coverage: {e}")
        return None


def calculate_living_canopy(
    image_path: str,
    save_map: bool = False,
    plant_id: str = "unknown",
    output_dir: str | None = None,
) -> float | Tuple[float, dict]:
    """
    Calcula el porcentaje de imagen cubierto por tejido verde vivo.
    
    Args:
        image_path: Ruta a la imagen.
        save_map: Si True, genera mapa de diagnóstico visual.
        plant_id: Identificador para el nombre del archivo.
        output_dir: Directorio de salida.
    
    Returns:
        float: Porcentaje de cobertura, o tupla (coverage, map_info).
    """
    coverage = calculate_canopy_coverage(image_path)
    if coverage is None:
        return None
    
    if save_map:
        map_info = generate_diagnostic_map(
            image_path, 
            plant_id=plant_id, 
            output_dir=output_dir
        )
        return coverage, map_info
    
    return coverage
