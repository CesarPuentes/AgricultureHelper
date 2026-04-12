"""
llm_api/service.py — Servicio de Clasificación con Modelos VLM
==============================================================
Funciones para clasificación de enfermedades usando Gemini Vision.
Refactored en Phase 2 para usar utils compartidas y LLMConfig.
"""

import logging
import shutil
import traceback
from typing import Optional

import google.generativeai as genai
from PIL import Image

from ..config import LLMConfig
from ..utils import ensure_output_dir
from .config import _configure_gemini

logger = logging.getLogger(__name__)


def _configure_model():
    """Configura y retorna el modelo Gemini, o None si no hay API key."""
    if not _configure_gemini():
        return None
    return genai.GenerativeModel(LLMConfig.GEMINI_MODEL)


def _clean_json_response(text: str) -> str:
    """
    Limpia el texto de respuesta para asegurar que es JSON parseable.
    
    Elimina markers de código como ```json y ```
    """
    text = text.strip()
    
    # Manejar ```json ... ```
    if text.startswith("```json"):
        text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
    # Manejar ``` ... ``` genérico
    elif text.startswith("```"):
        text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
    
    return text.strip()


def classify_disease(image_path: str, top_k: int = 3) -> Optional[list[dict]]:
    """
    Clasifica posibles enfermedades en una imagen usando Gemini Vision.
    
    Args:
        image_path: Ruta a la imagen.
        top_k: Número máximo de enfermedades a retornar.
    
    Returns:
        list[dict]: Lista de dicts con 'label' y 'score', o None si hay error.
    """
    model = _configure_model()
    if model is None:
        logger.error(
            "Se intentó llamar a classify_disease pero no hay GEMINI_API_KEY "
            "configurado en el entorno."
        )
        return None

    try:
        img = Image.open(image_path)
        
        prompt = f"""
        Actúa como un experto patólogo de plantas.
        Analiza detenidamente esta imagen fotográfica y evalúa qué enfermedad tiene la planta, o si está sana.
        Recuerda que puede haber varios síntomas superpuestos. Identifica los {top_k} principales.
        
        Devuelve tu respuesta EXACTAMENTE en formato de array JSON, donde cada elemento tiene la etiqueta y el puntaje.
        Ejemplo:
        [
          {{"label": "Mildew (Hongo Blanco)", "score": 0.98}},
          {{"label": "Amarillamiento / Necrosis", "score": 0.40}}
        ]
        """
        
        response = model.generate_content([prompt, img])
        text = _clean_json_response(response.text)
        results = __import__("json").loads(text)
        
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Error en clasificación de enfermedad por Gemini: {e}")
        logger.error(traceback.format_exc())
        return None


def classify_tray_disease(
    image_path: str,
    rows: int = 6,
    cols: int = 4,
    output_dir: str | None = None,
) -> Optional[dict]:
    """
    Evaluación de bandeja entera usando Gemini API.
    
    Args:
        image_path: Ruta a la imagen de la bandeja.
        rows: Filas de la cuadrícula (para calcular total esperado).
        cols: Columnas de la cuadrícula.
        output_dir: Directorio para copiar el mapa (usa DEFAULT_DEBUG_DIR si None).
    
    Returns:
        dict: Resultados con map_path, per_plant, y summary.
    """
    model = _configure_model()
    if model is None:
        logger.error(
            "Se intentó llamar a classify_tray_disease pero no hay GEMINI_API_KEY."
        )
        return None

    try:
        img = Image.open(image_path)
        
        prompt = f"""
        Actúa como un ingeniero y patólogo agrícola experto.
        Te presento una bandeja de cultivo que contiene plantas sembradas en una cuadrícula lógica. 
        Miras toda la bandeja, cuentas a simple vista las plantas totales y me dices exactamente cuántas identificas visiblemente sanas y cuántas están enfermas (ej. manchas blancas de hongo, resecas, muertas, amarilleando severamente).
        
        Tu tarea es devolverme un JSON estructurado con el resumen matemático y una corta explicación.
        FORMATO ESTRICTO:
        {{
            "total": 24,
            "healthy": 18,
            "diseased": 6,
            "explanation": "Hay 6 plantas que muestran manchas blancas de powdery mildew en las hojas centrales del lado derecho."
        }}
        """
        
        response = model.generate_content([prompt, img])
        text = _clean_json_response(response.text)
        data = __import__("json").loads(text)
        
        # Copiar imagen original como mapa usando util compartida
        out_dir = ensure_output_dir(output_dir)
        import time
        map_filename = f"gemini_map_{int(time.time())}.png"
        map_path = out_dir / map_filename
        shutil.copy(image_path, map_path)

        # Construir lista per-plant
        per_plant = []
        for i in range(data.get('healthy', 0)):
            per_plant.append({
                "roi": i, 
                "label": "healthy", 
                "score": 0.99, 
                "status": "healthy"
            })
        for i in range(data.get('diseased', 0)):
            per_plant.append({
                "roi": 99, 
                "label": "diseased", 
                "score": 0.99, 
                "status": "diseased"
            })

        return {
            "map_path": str(map_path),
            "per_plant": per_plant,
            "summary": {
                "total": data.get('total', rows * cols), 
                "healthy": data.get('healthy', 0), 
                "diseased": data.get('diseased', 0),
                "ai_explanation": data.get('explanation', '')
            }
        }

    except Exception as e:
        logger.error(f"Error en clasificación de bandeja por Gemini: {e}")
        return None
