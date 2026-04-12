# 🧠 Core Services & MAS Logic

This directory contains the primary "brain" of the AgenteAgricultura system. It is organized into specialized modules and packages to separate concerns between different analysis domains.

## 📦 Directory Map

### 📂 `anomaly/`
*The Deterministic Anomaly Engine.*
This package focus on detecting physiological shifts without necessarily naming the pathogen. It is designed to be explainable and lightweight.
- **`engine.py`**: The high-level coordinator that runs all detectors and calculates the composite anomaly score.
- **`detectors.py`**: Individual computer vision algorithms for Chlorosis (Active).
- **`utils.py`**: Low-level image processing utilities (masking, labeling, and debug output).

### 📂 `llm_api/`
*The VLM-Based Diagnostic Agent.*
Handles high-level disease classification by leveraging Large Multimodal Models (LMMs).
- **`config.py`**: Manages Gemini API initialization and availability checks.
- **`service.py`**: Provides classification services for both single rosettes and full cultivate trays.

### 🧪 `vision_extractor.py`
*The Structural Vision Layer.*
Handles the foundational structural analysis of the trays.
- **ROI Grid Management**: Partitioning images into 6x4 (or custom) grids.
- **Canopy Calculation**: Estimating living tissue percentage using LAB-color space thresholding.
- **Plant Counting**: Identifying individual plant segments within the grid.

### 🚨 `alerts.py`
*The Tier 0 Deterministic Watchdog.*
Contains the logic for the immediate alert system.
- **Vision Delta**: Flags sudden loss of canopy (e.g., rapid wilting or removal).
- **Sensor Watchdog**: Monitors telemetry silence to identify hardware disconnects.

---

## 🛠️ Usage Philosophy
The core logic follows a hierarchical approach:
1. **Tier 0** (`alerts.py`): Real-time, non-visual/simple-visual checks for immediate action.
2. **Tier 1** (`anomaly/`): Explainable Computer Vision metrics for growth tracking.
3. **Tier 2** (`llm_api/`): Complex, AI-driven diagnostics for specific pathogen naming.
