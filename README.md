# 🌱 AgricultureHelper: Multi-Agent Plant Monitoring System

Welcome to the **AgricultureHelper** repository. This project aims to build a robust, edge-first Multi-Agent System (MAS) to monitor plant health using a combination of environmental sensors and computer vision.

## 🎯 Current Project Focus: Tier 1 MVP

We are currently in the early planning and development stages, focusing entirely on a **Frictionless Tier 1 MVP** designed for ultra-low-cost edge devices (e.g., Raspberry Pi) in offline or remote greenhouse environments.

The architecture is built on a **Blackboard Pattern** and orchestrated via **LangGraph**, ensuring we can later seamlessly plug in heavy Vision-Language Models (VLMs) and Cloud APIs in future scaling phases.

### 🚀 Tier 1 Development Roadmap

*   **Phase 1: The "Dumb" Extractors**
    *   Set up Python OpenCV pipelines to continuously extract an HSV "Green Ratio" as a universal proxy for plant growth/health.
    *   Deploy MQTT brokers to ingest basic IoT soil/temperature sensors into a local SQLite/TimescaleDB.
*   **Phase 2: The Anomaly Brain**
    *   Implement lightweight edge anomaly detection (e.g., `SciKit-Learn IsolationForest`).
    *   Establish the unified `PlantHealthState` Blackboard.
*   **Phase 3: The Investigation Router**
    *   Build the LangGraph Supervisor to route logic (e.g., *Is the green canopy dropping because moisture is low?*).
*   **Phase 4: The Farmer Interface**
    *   Deploy a local SMS or Gradio UI to alert the human farmer "In The Loop" whenever an unknown visual anomaly occurs.

## 🛠️ How to Run

### 1. Requirements
Ensure you have the dependencies installed (preferably in a virtual environment):
```bash
pip install -r requirements.txt
```

### 2. Start the Backend (FastAPI Server)
The server processes the images and runs the vision algorithms.
### 🏃 Cómo ejecutar

#### 1. Backend (FastAPI)
```bash
uvicorn src.api.app:app --reload
```

#### 2. Frontend (Streamlit)
```bash
streamlit run frontend_test.py
```
The interface will open in your browser at `http://localhost:8501`.

---

### 📷 Consideraciones Técnicas: Ground Sample Distance (GSD)
Actualmente, el algoritmo de visión (`src/core/vision_extractor.py`) depende de umbrales estáticos en píxeles (ej. eliminar ruido menor a 200px). Esto significa que el sistema es susceptible a errores si la distancia de la cámara a la bandeja cambia (modificando el GSD). 
*   **A futuro:** Se recomienda implementar un sistema de calibración dinámica (ej. usando un marcador físico de tamaño conocido en la bandeja) para que el algoritmo calcule la relación de *píxeles/cm* y ajuste los parámetros automáticamente sin importar la altura de la cámara.

---

*For a full breakdown of the agent design and theoretical architecture (including scaling to Tier 2 and Tier 3), see the `multiagent_plant_monitoring_proposal.md`.*
