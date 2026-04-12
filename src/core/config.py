"""
config.py — Configuración Centralizada del Núcleo
================================================
Un único lugar para todas las constantes, paths y thresholds
del módulo core. Facilita ajustes sin tocar lógica de negocio.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
# Directorio raíz del proyecto (inferido desde este archivo)
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directorio de salida para mapas de diagnóstico y debug
DEFAULT_DEBUG_DIR = _PROJECT_ROOT / "debug_output"

# Directorio de resultados de análisis (PlantCV outputs)
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "analysis_results"


# ---------------------------------------------------------------------------
# Thresholds de Visión
# ---------------------------------------------------------------------------
class VisionThresholds:
    """Thresholds para algoritmos de visión por computadora."""
    
    # ExG (Excess Green Index) para detección de tejido verde
    EXG_THRESHOLD: int = 20
    EXG_FILL_SIZE: int = 200
    
    # LAB color space para segmentación de bandejas
    LAB_CHANNEL: str = 'a'
    LAB_FILL_SIZE: int = 200
    
    # Grid segmentation (bandejas 6x4 estándar)
    DEFAULT_ROWS: int = 6
    DEFAULT_COLS: int = 4
    
    # Leaf counting
    WATERSHED_DISTANCE: int = 15
    MIN_LEAF_AREA: int = 500


# ---------------------------------------------------------------------------
# Thresholds de Anomalías
# ---------------------------------------------------------------------------
class AnomalyThresholds:
    """Thresholds para detección de anomalías."""
    
    # Clorosis
    CHLOROSIS_SCALE_FACTOR: float = 2.0
    CHLOROSIS_MAX_SCORE: int = 100
    CHLOROSIS_ATTENTION_THRESHOLD: int = 40
    
    # HSV ranges para tejido verde saludable
    HEALTHY_GREEN_LOWER: tuple = (35, 40, 40)
    HEALTHY_GREEN_UPPER: tuple = (85, 255, 255)
    
    # HSV ranges para estrés (amarillo/marrón)
    STRESS_YELLOW_LOWER: tuple = (10, 40, 40)
    STRESS_YELLOW_UPPER: tuple = (35, 255, 255)


# ---------------------------------------------------------------------------
# Thresholds de Alertas
# ---------------------------------------------------------------------------
class AlertThresholds:
    """Thresholds para el sistema de alertas (Capa 0)."""
    
    # Watchdog: minutos máximos de silencio antes de alerta
    MAX_SILENCE_MINUTES: int = 130  # ~2h 10min
    
    # Vision Delta: ventana de comparación y umbral
    VISION_DELTA_HOURS: int = 4
    VISION_DELTA_THRESHOLD_PCT: float = 5.0


# ---------------------------------------------------------------------------
# Configuración de PlantCV
# ---------------------------------------------------------------------------
class PlantCVConfig:
    """Configuración para PlantCV (plantcv.parallel.WorkflowInputs)."""
    
    DEBUG_MODE: str = "None"  # "None", "plot", "print"
    DPI: int = 100
    TEXT_SIZE: float = 10
    TEXT_THICKNESS: int = 20
    RESULT_FILENAME: str = "arabidopsis_results.json"
    SAMPLE_LABEL: str = "genotype"


# ---------------------------------------------------------------------------
# Configuración de LLM API
# ---------------------------------------------------------------------------
class LLMConfig:
    """Configuración para APIs de Large Multimodal Models."""
    
    GEMINI_MODEL: str = "gemini-1.5-flash"
    DEFAULT_TOP_K: int = 3
    IMAGE_QUALITY: str = "auto"


# ---------------------------------------------------------------------------
# Utilidades de configuración
# ---------------------------------------------------------------------------
def get_debug_dir(custom_path: str | Path | None = None) -> Path:
    """Retorna el directorio de debug, creándolo si es necesario."""
    path = Path(custom_path) if custom_path else DEFAULT_DEBUG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_dir(custom_path: str | Path | None = None) -> Path:
    """Retorna el directorio de resultados, creándolo si es necesario."""
    path = Path(custom_path) if custom_path else DEFAULT_RESULTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path