# 🧠 Core Services & MAS Logic

This directory contains the primary "brain" of the AgenteAgricultura system. It is organized into specialized modules and packages to separate concerns between different analysis domains.

---

## 📦 Phase 1: Shared Infrastructure ✅

### 📂 `utils/`
**Utilidades compartidas para maximizar code reuse.**
- **`io_utils.py`**: Gestión centralizada de directorios de salida, timestamps, y archivos JSON.
- **`image_utils.py`**: Máscaras compartidas (ExG green mask, LAB plant mask), helpers de visualización.

### 📄 `config.py`
**Configuración centralizada de thresholds y constantes.**

- **`VisionThresholds`**: ExG threshold, LAB params, grid dimensions, watershed settings.
- **`AnomalyThresholds`**: Chlorosis scale factor, HSV ranges for healthy/stressed tissue.
- **`PlantCVConfig`**: Debug mode, DPI, text size, result filename.
- **`LLMConfig`**: Gemini model name, default top_k.

---

## 📦 Phase 2: Modular Vision & Refactored Modules ✅

### 📂 `vision/` (NEW)
**Subpaquete modular que reorganiza vision_extractor.py en módulos focales.**
- **`canopy.py`**: Cálculo de cobertura verde viva (calculate_living_canopy, calculate_canopy_coverage).
- **`grid.py`**: Segmentación de cuadrícula y conteo (segment_tray_grid, count_plants, count_leaves).
- **`diagnostic_map.py`**: Generación de mapas visuales de diagnóstico.

### 🔄 Refactored Modules

- **`llm_api/service.py`**: Usa LLMConfig + utils de I/O + helpers _configure_model, _clean_json_response.
- **`vision_extractor.py`**: Ahora es wrapper de compatibilidad (re-exporta desde vision/).

---

## 📦 Directory Map

### 📂 `anomaly/`
*The Deterministic Anomaly Engine.*
This package focus on detecting physiological shifts without necessarily naming the pathogen. It is designed to be explainable and lightweight.
- **`engine.py`**: The high-level coordinator that runs all detectors and calculates the composite anomaly score.
- **`detectors.py`**: Individual computer vision algorithms for Chlorosis.
- **`utils.py`**: Low-level image processing utilities. **Uses `utils/`**.

### 📂 `llm_api/`
*The VLM-Based Diagnostic Agent.*
- **`config.py`**: Manages Gemini API initialization.
- **`service.py`**: Disease classification services. **Refactored with LLMConfig**.

### 📂 `vision/` (Phase 2)
*The Modular Vision Layer.*
- **`canopy.py`**: Green coverage calculation.
- **`grid.py`**: Grid segmentation, plant/leaf counting.
- **`diagnostic_map.py`**: Visual overlay generation.

### 🧪 `vision_extractor.py`
*Backward compatibility wrapper.* Re-exports from `vision/` subpackage.

---

## 🛠️ Usage Philosophy
The core logic follows a hierarchical approach:
1. **Tier 1** (`anomaly/`): Explainable Computer Vision metrics for growth tracking.
2. **Tier 1** (`anomaly/`): Explainable Computer Vision metrics for growth tracking.
3. **Tier 2** (`llm_api/`): Complex, AI-driven diagnostics for specific pathogen naming.

---

## 🔄 Migration Status

| Module | Phase 1 Status | Phase 2 Status |
|--------|---------------|----------------|
| `config.py` | ✅ Complete | — |
| `utils/` | ✅ Complete | — |
| `anomaly/utils.py` | ✅ Refactored | — |
| `anomaly/detectors.py` | ✅ Refactored | — |
| `anomaly/engine.py` | ✅ Refactored | — |
| `vision_extractor.py` | ✅ Refactored | ✅ Now wrapper |
| **`vision/` subpackage** | — | ✅ **NEW** |
| **`llm_api/service.py`** | — | ✅ Refactored |

---

## 📐 Refactoring Principles

1. **Code Reuse (DRY)**: Centralized masks, utilities, and config eliminate duplication.
2. **Simplify Without Breaking**: Backward compatibility maintained via wrapper.
3. **Configuration Centralized**: All thresholds in `config.py`, not scattered.
4. **Consistent Logging**: Unified `logger` usage instead of `print()`.
5. **Type Hints**: Improved code readability and IDE support.
6. **Modular Structure**: Vision logic split into focused submodules.

---

## 🚀 Recommended Imports

```python
# New modular imports (recommended)
from src.core.vision import calculate_living_canopy, count_plants, count_leaves
from src.core.anomaly import calculate_anomaly_score
from src.core.llm_api import classify_disease, classify_tray_disease

# Legacy imports (still work for backwards compatibility)
from src.core.vision_extractor import calculate_living_canopy, count_plants
```
