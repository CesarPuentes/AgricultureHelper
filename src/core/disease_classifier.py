"""
disease_classifier.py — Clasificador de Enfermedades de Plantas (HuggingFace)
------------------------------------------------------------------------------
Módulo opcional que usa un modelo ResNet18 fine-tuned en PlantVillage
para clasificar enfermedades de plantas a partir de fotos de hojas.

Auto-detección: Si `transformers` y `torch` no están instalados,
el módulo se desactiva silenciosamente y el sistema sigue funcional
con puro OpenCV/PlantCV.

El modelo pesa ~45MB y corre perfectamente en CPU (incluso Raspberry Pi).
No requiere GPU.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-detección de dependencias
# ---------------------------------------------------------------------------
_HF_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
    logger.info("HuggingFace transformers disponible. Clasificador de enfermedades activo.")
except ImportError:
    logger.info(
        "HuggingFace transformers no instalado. "
        "Clasificador de enfermedades desactivado. "
        "Para activar: pip install transformers torch torchvision"
    )

# ---------------------------------------------------------------------------
# Singleton del clasificador (lazy loading)
# ---------------------------------------------------------------------------
_classifier_instance = None
_MODEL_NAME = "gianlab/swin-tiny-patch4-window7-224-finetuned-plantdisease"


def _get_classifier():
    """
    Lazy-load del pipeline de clasificación.
    El modelo se descarga y cachea automáticamente la primera vez (~45MB).
    """
    global _classifier_instance
    if _classifier_instance is None:
        logger.info(f"Cargando modelo '{_MODEL_NAME}' (primera ejecución descarga ~45MB)...")
        token = os.getenv("HF_TOKEN")
        _classifier_instance = hf_pipeline(
            "image-classification",
            model=_MODEL_NAME,
            token=token,
            device=-1,  # Forzar CPU
        )
        logger.info("Modelo cargado exitosamente.")
    return _classifier_instance


# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """
    Returns True if HuggingFace dependencies are installed and the
    disease classifier can be used.
    """
    return _HF_AVAILABLE


def classify_disease(image_path: str, top_k: int = 3) -> Optional[list[dict]]:
    """
    Clasifica posibles enfermedades en una imagen de hoja de planta.

    Usa el modelo ResNet18 fine-tuned en el dataset PlantVillage
    (38 clases: enfermedades comunes + hojas sanas de varios cultivos).

    Args:
        image_path: Ruta a la imagen de la hoja.
        top_k: Número de predicciones top a retornar (default 3).

    Returns:
        Lista de dicts con 'label' y 'score', ordenados por confianza.
        Ejemplo: [{"label": "Tomato___Early_blight", "score": 0.92}, ...]
        Retorna None si transformers no está instalado.
    """
    if not _HF_AVAILABLE:
        return None

    try:
        classifier = _get_classifier()
        results = classifier(image_path, top_k=top_k)
        return results
    except Exception as e:
        logger.error(f"Error en clasificación de enfermedad: {e}")
        return None
