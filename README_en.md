# 🌱 AgricultureHelper: Multi-Agent Plant Monitoring System

Welcome to the **AgricultureHelper** repository. This project aims to build a robust, edge-first Multi-Agent System (MAS) to monitor plant health using a combination of environmental sensors and computer vision.

![alt text](Sample.png)

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

## 🛠️ How to run the project

### 1. Requirements and Virtual Environment
It is recommended to use a virtual environment to install dependencies in isolation:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Services

To test the complete flow with the API and visual interface:

#### Main Service (FastAPI + Web Interface)
The server now processes images, interacts with simulated sensors, and serves the web interface (frontend) directly using Jinja2.
```bash
uvicorn src.api.app:app --reload
```
- The **Base and Analysis API** will be natively available at `http://localhost:8000`.
- The **User Interface (Frontend)** can be opened in your browser by going to `http://localhost:8000/ui`.

### 📸 Vision Pipeline and Diagnostic Maps

The vision system (located in `src/core/vision_extractor.py`) is modularly designed to:
- Calculate living green canopy coverage percentage.
- Count the number of individual plants by analyzing rectangular grid cultivation trays.

**Viewing Diagnostic Maps:**
When testing images from the frontend or with the tests, the pipeline automatically generates a combined image file in PNG format, saving it in the `diagnostic_maps_demo/` folder. These "maps" provide verifiable visual feedback:
- **Semi-transparent Green**: Overlaid layer indicating detected green/living tissue.
- **Semi-transparent Red**: Overlaid layer indicating ignored elements (soil, pot, or dead tissue).
- **White Outlines**: Borders of leaves or plants to check segmentation and counting accuracy.
- Values like Coverage Percentage are printed directly on the image.

#### Test scripts locally via CLI
```bash
# Run the unit test and generation file
python test_api.py

# Run the unit test suite
pytest test_api.py
```

---

### 📷 Technical Considerations: Ground Sample Distance (GSD)
Currently, the vision algorithm (`src/core/vision_extractor.py`) relies on static pixel thresholds (e.g., removing noise smaller than 200px). This means the system is susceptible to errors if the camera-to-tray distance changes (modifying the GSD).

*   **Future recommendation:** It is recommended to implement a dynamic calibration system (e.g., using a physical marker of known size on the tray) so the algorithm can calculate the *pixels/cm* ratio and adjust parameters automatically regardless of camera height.

---

*For a full breakdown of the agent design and theoretical architecture (including scaling to Tier 2 and Tier 3), see the `multiagent_plant_monitoring_proposal.md`.*
