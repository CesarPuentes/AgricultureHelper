"""
vision/ — Módulo de Visión por Computadora Modular
==================================================
Subpaquete que reorganiza vision_extractor.py en módulos focales:
- canopy.py: Cálculo de cobertura verde viva
- grid.py: Segmentación de cuadrícula y conteo de plantas
- diagnostic_map.py: Generación de mapas visuales de diagnóstico

Para backwards compatibility, vision_extractor.py sigue funcionando
como wrapper que importa de estos módulos.
"""

from .canopy import calculate_living_canopy, calculate_canopy_coverage
from .grid import count_plants, count_leaves, segment_tray_grid
from .diagnostic_map import generate_diagnostic_map

__all__ = [
    "calculate_living_canopy",
    "calculate_canopy_coverage",
    "count_plants",
    "count_leaves",
    "segment_tray_grid",
    "generate_diagnostic_map",
]
