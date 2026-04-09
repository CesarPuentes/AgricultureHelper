import logging
import os
import json
import traceback
import google.generativeai as genai
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración Gemini API
# ---------------------------------------------------------------------------
def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """
    Returns True if GEMINI_API_KEY is configured in the environment.
    """
    return bool(os.getenv("GEMINI_API_KEY"))


def classify_disease(image_path: str, top_k: int = 3) -> Optional[list[dict]]:
    """
    Clasifica posibles enfermedades en una imagen usando Gemini Vision.
    Returns list of dicts with 'label' and 'score', or None.
    """
    if not _configure_gemini():
        logger.error("Se intentó llamar a classify_disease pero no hay GEMINI_API_KEY configurado en el entorno.")
        return None

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
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
        
        # Limpiar el output para asegurar que es JSON parseable
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
        elif text.startswith("```"):
            text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
        results = json.loads(text.strip())
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Error en clasificación de enfermedad por Gemini: {e}")
        logger.error(traceback.format_exc())
        return None


def classify_tray_disease(image_path: str, rows: int = 6, cols: int = 4,
                          output_dir: str = "diagnostic_maps_demo") -> Optional[dict]:
    """
    Evaluación de bandeja entera usando Gemini API.
    A diferencia del método de HuggingFace/PlantCV, enviamos toda la bandeja 
    y le pedimos a la VLM que analice y cuente directamente las plantas enfermas.
    """
    if not _configure_gemini():
         logger.error("Se intentó llamar a classify_tray_disease pero no hay GEMINI_API_KEY.")
         return None

    try:
        model = genai.GenerativeModel('gemini-1.5-flash') 
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
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
            
        data = json.loads(text.strip())
        
        # Replicamos el output requerido por el frontend para no romper la app de FastAPI
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        import time
        map_path = os.path.join(output_dir, f"gemini_map_{int(time.time())}.png")
        shutil.copy(image_path, map_path)

        per_plant = []
        for i in range(data.get('healthy', 0)):
            per_plant.append({"roi": i, "label": "healthy", "score": 0.99, "status": "healthy"})
        for i in range(data.get('diseased', 0)):
            per_plant.append({"roi": 99, "label": "diseased", "score": 0.99, "status": "diseased"})

        return {
            "map_path": map_path,
            "per_plant": per_plant,
            "summary": {
                "total": data.get('total', rows*cols), 
                "healthy": data.get('healthy', 0), 
                "diseased": data.get('diseased', 0),
                "ai_explanation": data.get('explanation', '')
            }
        }

    except Exception as e:
        logger.error(f"Error en clasificación de bandeja por Gemini: {e}")
        return None
