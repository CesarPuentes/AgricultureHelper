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

---
*For a full breakdown of the agent design and theoretical architecture (including scaling to Tier 2 and Tier 3), see the `multiagent_plant_monitoring_proposal.md`.*
