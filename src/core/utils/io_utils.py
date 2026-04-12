"""
io_utils.py — Utilidades de Entrada/Salida Compartidas
======================================================
Funciones para gestión de directorios, archivos y timestamps.
Unifica la creación de outputs y logging entre módulos.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import DEFAULT_DEBUG_DIR

logger = logging.getLogger(__name__)


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    """
    Asegura que el directorio de salida exista.
    
    Args:
        output_dir: Directorio destino. Si es None, usa DEFAULT_DEBUG_DIR.
    
    Returns:
        Path: Directorio creado (o existente), normalizado como Path.
    
    Example:
        >>> out_path = ensure_output_dir("/tmp/my_output")
        >>> out_path.exists()
        True
    """
    if output_dir is None:
        out_path = DEFAULT_DEBUG_DIR
    else:
        out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def get_timestamp_filename(
    prefix: str,
    extension: str = "png",
    plant_id: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Genera un nombre de archivo con timestamp para evitar colisiones.
    
    Args:
        prefix: Prefijo del archivo (ej: "diagnostic_map").
        extension: Extensión sin punto (ej: "png", "json").
        plant_id: Identificador opcional de planta para incluir en el nombre.
        output_dir: Directorio de salida (se crea si no existe).
    
    Returns:
        Path: Ruta completa al archivo generado.
    
    Example:
        >>> path = get_timestamp_filename("leaf_summary", "png", plant_id="plant_001")
        >>> path.name
        'leaf_summary_plant_001_20240115_143022.png'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if plant_id:
        safe_id = "".join(c if c.isalnum() else "_" for c in plant_id)
        filename = f"{prefix}_{safe_id}_{timestamp}.{extension}"
    else:
        filename = f"{prefix}_{timestamp}.{extension}"
    
    out_dir = ensure_output_dir(output_dir)
    return out_dir / filename


def save_debug_image(
    image,
    name: str,
    output_dir: str | Path | None = None,
    plant_id: str | None = None,
    extension: str = "png",
) -> str:
    """
    Guarda una imagen de debug y retorna la ruta del archivo.
    
    Unifica la lógica de _save_debug de anomaly/utils.py y la
    creación inline de archivos en vision_extractor.py.
    
    Args:
        image: Imagen numpy (OpenCV format BGR).
        name: Nombre base del archivo (sin timestamp).
        output_dir: Directorio de salida.
        plant_id: Identificador opcional para el nombre.
        extension: Extensión de la imagen.
    
    Returns:
        str: Ruta absoluta del archivo guardado.
    """
    import cv2
    
    path = get_timestamp_filename(
        prefix=name,
        extension=extension,
        plant_id=plant_id,
        output_dir=output_dir,
    )
    
    success = cv2.imwrite(str(path), image)
    if not success:
        logger.error(f"Failed to write debug image to {path}")
        raise IOError(f"Could not save image to {path}")
    
    logger.debug(f"Debug image saved: {path}")
    return str(path)


def load_json_if_exists(file_path: str | Path) -> dict | None:
    """
    Carga un archivo JSON si existe y es válido.
    
    Args:
        file_path: Ruta al archivo JSON.
    
    Returns:
        dict: Contenido del JSON, o None si no existe/hay error.
    """
    path = Path(file_path)
    if not path.exists():
        logger.debug(f"JSON file not found: {path}")
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error reading JSON from {path}: {e}")
        return None


def save_json_safe(data: Any, file_path: str | Path, indent: int = 2) -> str:
    """
    Guarda datos como JSON de forma segura (sobrescribe atómicamente).
    
    Args:
        data: Datos serializables a JSON.
        file_path: Ruta de destino.
        indent: Indentación del JSON.
    
    Returns:
        str: Ruta del archivo guardado.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Escritura atómica: escribe a .tmp y renombra
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        tmp_path.replace(path)  # Atomic on POSIX
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise IOError(f"Failed to save JSON to {path}: {e}") from e
    
    logger.debug(f"JSON saved: {path}")
    return str(path)