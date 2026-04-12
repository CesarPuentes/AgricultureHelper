"""
utils/ — Utilidades Compartidas del Núcleo
==========================================
Paquete que centraliza funciones de I/O y procesamiento de imágenes
usadas por múltiples módulos (vision, anomaly, alerts, llm_api).

Se prioriza:
1. Code reuse (DRY)
2. Consistencia en logging y outputs
3. Configuración centralizada
"""

from .io_utils import (
    ensure_output_dir,
    save_debug_image,
    get_timestamp_filename,
    load_json_if_exists,
    save_json_safe,
)

from .image_utils import (
    create_green_mask,
    create_lab_mask,
    draw_label_pill,
    merge_mask_channels,
)

__all__ = [
    # I/O
    "ensure_output_dir",
    "save_debug_image",
    "get_timestamp_filename",
    "load_json_if_exists",
    "save_json_safe",
    # Image processing
    "create_green_mask",
    "create_lab_mask",
    "draw_label_pill",
    "merge_mask_channels",
]