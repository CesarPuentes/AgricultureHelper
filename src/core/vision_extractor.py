"""
vision_extractor.py
-------------------
Wrapper de compatibilidad hacia atrás para el módulo de visión.

** NUEVO EN PHASE 2: ** Importa desde src/core/vision/

Este archivo existe solo para mantener compatibilidad con imports existentes.
Código nuevo debería importar directamente desde:

    from src.core.vision import (
        calculate_living_canopy,
        count_plants,
        count_leaves,
        generate_diagnostic_map,
    )
"""

# Re-export from modular vision/ subpackage
from .vision import (
    calculate_living_canopy,
    count_plants,
    count_leaves,
    generate_diagnostic_map,
    segment_tray_grid,
    calculate_canopy_coverage,
)

__all__ = [
    "calculate_living_canopy",
    "count_plants",
    "count_leaves",
    "generate_diagnostic_map",
    "segment_tray_grid",
    "calculate_canopy_coverage",
]
