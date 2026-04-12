"""
vision/diagnostic_map.py — Generación de Mapas de Diagnóstico Visual
=====================================================================
Crea overlays de colores superpuestos sobre imágenes para visualizar
tejido vivo vs muerto con leyenda y métricas.
"""

import logging
from datetime import datetime
from typing import NamedTuple

import cv2
import numpy as np

from ..utils import ensure_output_dir, save_debug_image, draw_label_pill
from ..utils.image_utils import create_green_mask, merge_mask_channels, draw_contours

logger = logging.getLogger(__name__)


class DiagnosticMapResult(NamedTuple):
    """Resultado estructurado de generate_diagnostic_map."""
    map_path: str
    coverage_pct: float
    generated_at: str


def generate_diagnostic_map(
    image_path: str,
    plant_id: str = "unknown",
    output_dir: str | None = None,
    living_color: tuple = (0, 200, 0),
    dead_color: tuple = (0, 0, 180),
    living_alpha: float = 0.4,
    dead_alpha: float = 0.3,
) -> dict:
    """
    Genera un mapa de diagnóstico visual con overlays de colores.
    
    Args:
        image_path: Ruta a la imagen de entrada.
        plant_id: Identificador para el nombre del archivo.
        output_dir: Directorio de salida (usa DEFAULT_DEBUG_DIR si None).
        living_color: Color BGR para tejido vivo.
        dead_color: Color BGR para suelo/no-verde.
        living_alpha: Peso de blend para tejido vivo.
        dead_alpha: Peso de blend para tejido muerto.
    
    Returns:
        dict: map_path, coverage_pct, generated_at
    """
    out_dir = ensure_output_dir(output_dir)
    original, mask = create_green_mask(image_path)
    
    total_pixels = mask.size
    living_pixels = np.count_nonzero(mask)
    coverage_pct = round((living_pixels / total_pixels) * 100, 2)

    overlay = original.copy()
    
    # Crear máscaras de 3 canales para blending
    living_mask_3ch = merge_mask_channels(mask) > 0
    dead_mask_3ch = ~living_mask_3ch
    
    # Capa verde para tejido vivo
    green_layer = np.full_like(original, living_color)
    overlay = np.where(
        living_mask_3ch,
        cv2.addWeighted(original, 1 - living_alpha, green_layer, living_alpha, 0),
        overlay
    )
    
    # Capa roja para suelo/no-verde
    red_layer = np.full_like(original, dead_color)
    overlay = np.where(
        dead_mask_3ch,
        cv2.addWeighted(original, 1 - dead_alpha, red_layer, dead_alpha, 0),
        overlay
    )
    
    # Dibujar contornos
    overlay = draw_contours(overlay, mask, (255, 255, 255), 2)
    
    # Etiqueta principal de cobertura
    draw_label_pill(
        overlay, 10, 10,
        f"Cobertura Verde: {coverage_pct}%",
        bg_color=(0, 0, 0)
    )
    
    # Leyenda
    legend_y = 50
    draw_label_pill(
        overlay, 15, legend_y,
        "Tejido Vivo",
        bg_color=living_color
    )
    draw_label_pill(
        overlay, 15, legend_y + 25,
        "Suelo / No-verde",
        bg_color=dead_color
    )
    
    # Guardar usando util compartida
    map_path = save_debug_image(overlay, f"diagnostic_map_{plant_id}", out_dir)
    
    return {
        "map_path": map_path,
        "coverage_pct": coverage_pct,
        "generated_at": datetime.now().isoformat()
    }
