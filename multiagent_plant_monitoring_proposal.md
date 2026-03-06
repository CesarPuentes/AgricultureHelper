# 🌱 Multi-Agent Plant Monitoring Service — Architecture Proposal

> *A short expert proposal on strategies and architectures for a multiagent service that monitors plants from visual, sensor, farmer interaction, and system integration fronts.*

---

## 1. Problem Statement

Modern agriculture demands holistic plant health monitoring that integrates **multiple modalities**:

| Front | Data Source | Challenge |
|:------|:-----------|:----------|
| **Visual** | Cameras, drones, satellite imagery | Detecting disease, pests, nutrient deficiency from pixels |
| **Sensor data** | IoT sensors (soil moisture, pH, temperature, humidity, light) | Continuous real-time streams, anomaly detection |
| **Farmer interaction** | Natural language (chat, voice), manual observations | Translating domain expertise into actionable inputs |
| **System integration** | Weather APIs, market prices, irrigation controllers, ERP systems | Coordinating actions across external platforms |

No single agent can handle all of these effectively. A **multi-agent system (MAS)** is the natural architecture.

---

## 2. Recommended Architecture: Hierarchical MAS with Specialized Agents

### 2.1 High-Level Design

```mermaid
graph TD
    subgraph "Orchestrator Layer"
        O["🧠 Supervisor Agent<br/>(Orchestrator / Router)"]
    end

    subgraph "Specialist Agents"
        V["👁️ Vision Agent<br/>Image analysis, disease detection"]
        S["📡 Sensor Agent<br/>IoT data processing, anomaly alerts"]
        F["🧑‍🌾 Farmer Agent<br/>NL interface, recommendations"]
        I["🔗 Integration Agent<br/>Weather, irrigation, ERP"]
    end

    subgraph "Shared Infrastructure"
        M["🗄️ Blackboard / Shared State<br/>(Plant Health Profile)"]
        K["📚 Knowledge Base<br/>(RAG: crop guides, pest DB)"]
        A["⚡ Action Bus<br/>(MCP / A2A Protocol)"]
    end

    O --> V
    O --> S
    O --> F
    O --> I

    V --> M
    S --> M
    F --> M
    I --> M

    V --> K
    F --> K

    I --> A
    S --> A
```

### 2.2 Why Hierarchical + Blackboard?

Drawing from the **Unified Theory of Agents** taxonomy:

| Dimension | Choice | Rationale |
|:----------|:-------|:----------|
| **Organization** | **Hierarchical** | A Supervisor Agent routes tasks to specialists, preventing chaos from concurrent analysis |
| **Coordination** | **Blackboard** (shared `PlantHealthState`) | All agents contribute to a unified plant profile — the Sensor Agent writes soil data, the Vision Agent writes disease scores, and the Farmer Agent reads all of it |
| **Social behavior** | **Cooperative** | All agents share one goal: maximize plant health |
| **Agent diversity** | **Heterogeneous** | Each agent uses different models and tools suited to its modality |
| **Architecture** | **Hybrid (BDI)** | Agents plan (deliberative) but react to real-time sensor alerts (reactive) |

---

## 3. Agent Specifications

### 3.1 👁️ Vision Agent

**Purpose**: Provide a frictionless, universal baseline for plant health by tracking "greenness" over time, independent of plant species. 

| Aspect | Detail |
|:-------|:-------|
| **Model** | **Local**: VLM (e.g., LLaVA, Moondream2, Qwen-VL) invoked only on anomalies.<br/>**Cloud fallback**: API-based Multimodal LLM |
| **Tools** | **HSV Green Pixel Extractor** (Simple classical CV thresholding via OpenCV/PlantCV), camera capture API |
| **Training / Data**| **Zero-Shot / No Training Required** (The beauty of tracking pure color is that green is green, regardless if it's mint or peppers). |
| **RAG source** | Pest/disease knowledge base with reference images (only used by the VLM for diagnosis). |
| **Output** | Structured diagnosis: `{disease: "none", green_pixel_ratio: 0.65, health_trend: "stable", confidence: 0.92}` |

**A Frictionless MVP Approach (The "Greenness Proxy")**:
The absolute simplest and most reliable way to monitor any plant species without retraining neural networks or generating synthetic datasets (UPGen) is to treat computer vision purely as a **"Dumb Green Pixel Counter"**. 

1. **The Fast Brain (Universal Health Tracker):** Every hour, a tiny Python script (using OpenCV or PlantCV) extracts the image, converts it to HSV color space, and counts the percentage of pixels that fall within the "healthy green" spectrum.
2. **Species-Agnostic Proxy:** This simple number acts as a universal proxy for both growth (more green pixels = the canopy is expanding) and health (fewer green pixels = leaves are yellowing or dropping). If you swap a mint field for peppers, the math doesn't change: a drop in green pixels mathematically equates to a drop in canopy health.
3. **The Slow Brain (On-Demand Agent):** If a time-series algorithm (like Scikit-Learn `IsolationForest`) detects that the "green ratio" has stalled or dropped unexpectedly, it acts as an alarm. *Only then* does the system wake up the VLM (LLaVA/Moondream2) to look at the image and deduce *why* the green is disappearing (e.g., pests, necrosis, wilting).

### 3.2 📡 Sensor Agent

**Purpose**: Process IoT sensor streams, detect anomalies, and trigger alerts.

| Aspect | Detail |
|:-------|:-------|
| **Model** | Small Language Model (SLM, e.g., Phi-3, Qwen 2.5 7B) — lightweight for edge deployment |
| **Tools** | Time-series database reader, anomaly detection algorithm, threshold configurator |
| **Architecture** | **Reactive** — primarily stimulus-response (threshold alerts), with deliberative periodic summaries |
| **Output** | Anomaly alerts: `{sensor: "soil_moisture_plot_3", value: 12%, threshold: 25%, status: "critical"}` |

**Strategy**: Deploy at the **edge** (on-device / on-farm gateway) to ensure low latency and operation during connectivity outages. Use an **iteration cap** and **compaction** to summarize long sensor logs before passing them to the Supervisor.

### 3.3 🧑‍🌾 Farmer Agent

**Purpose**: Natural language interface for the farmer — receives observations, answers questions, delivers recommendations.

| Aspect | Detail |
|:-------|:-------|
| **Model** | **Local**: Llama-3-8B-Instruct, Mistral-7B.<br/>**Cloud**: Capable conversational LLM (e.g., Gemini Pro, GPT-4o) |
| **Tools** | Text-to-speech / speech-to-text (for voice in the field), chat UI, notification system |
| **Memory** | **Episodic** (remembers past conversations: "Last week you mentioned yellowing in plot 5") + **Semantic** (crop knowledge) |
| **RAG source** | Local crop management guides, farm-specific historical data |
| **Output** | Plain-language advice, alerts framed in farmer's context |

**Strategy**: Implement the **Auditor Pattern** — recommendations generated by the Farmer Agent are verified against the Blackboard (sensor data + vision analysis) before being delivered. This prevents hallucinated advice.

### 3.4 🔗 Integration Agent

**Purpose**: Bridge between the MAS and external systems.

| Aspect | Detail |
|:-------|:-------|
| **Model** | **Local**: Hermes-2-Pro, Llama-3-Groq-Tool-Use.<br/>**Cloud**: Tool-calling LLM API |
| **Tools** | Weather API, irrigation controller API, market price API, ERP connector, notification service (SMS/email) |
| **Protocol** | **MCP** (Model Context Protocol) for standardized tool access; **A2A** for communicating with agents from partner platforms |
| **Output** | Executed actions: irrigation adjustments, weather-informed alerts, inventory orders |

**Strategy**: Apply **Human-in-the-Loop (HITL)** for any destructive action (e.g., starting irrigation, placing orders). The agent proposes; the farmer approves via the Farmer Agent's chat interface.

---

## 4. Shared Infrastructure

### 4.1 The Blackboard: Plant Health Profile

All agents contribute to a single, shared state object — the **Plant Health Profile**:

```python
class PlantHealthState(TypedDict):
    plot_id: str
    crop_type: str
    # Vision Agent writes:
    visual_diagnosis: List[Dict]       # disease, severity, confidence
    last_image_timestamp: str
    # Sensor Agent writes:
    soil_moisture: float
    soil_ph: float
    temperature: float
    humidity: float
    anomalies: List[Dict]
    # Farmer Agent writes:
    farmer_observations: List[str]     # "Leaves look yellow in NE corner"
    # Integration Agent writes:
    weather_forecast: Dict
    irrigation_status: str
    # Supervisor writes:
    overall_health_score: float        # 0.0–1.0, synthesized from all inputs
    recommended_actions: List[str]
```

### 4.2 Knowledge Base (RAG)

A **Hybrid RAG** system serving all agents:

- **Vector DB** (ChromaDB or Vertex AI Vector Search): Embedded crop guides, pest identification databases, treatment protocols
- **BM25 index**: For exact-match lookups (chemical names, product codes, sensor model numbers)
- **Graph RAG** (optional, for advanced setups): Knowledge graph of crop-disease-treatment relationships for multi-hop queries like *"What treatment works for early blight on tomatoes in humid climates?"*

---

## 5. Best Strategies — Ranked by Impact

| # | Strategy | From Theory Level | Impact | Effort |
|:--|:---------|:------------------|:-------|:-------|
| 1 | **Auditor Pattern on recommendations** | Level 7 | 🔴 Critical — prevents harmful advice to farmers | Medium |
| 2 | **Edge deployment for Sensor Agent** | Next Frontier (Edge AI) | 🔴 Critical — ensures real-time response, works offline | Medium |
| 3 | **Multimodal Vision via RAG** | Levels 4, 6, 8 | 🟠 High — image + knowledge = accurate diagnosis | Medium |
| 4 | **Blackboard shared state** | Level 7 | 🟠 High — unifies all modalities into one profile | Low |
| 5 | **HITL for destructive actions** | Level 0 | 🟠 High — safety for irrigation, purchases | Low |
| 6 | **MCP for tool interoperability** | Next Frontier | 🟡 Medium — future-proofs integrations | Low |
| 7 | **Episodic memory for Farmer Agent** | Prologue (MemGPT) | 🟡 Medium — personalized, context-aware advice | Medium |
| 8 | **Context engineering / compaction** | Next Frontier | 🟡 Medium — keeps long-running agents effective | Medium |

---

## 6. Technology Stack Recommendation

| Layer | Technology | Rationale |
|:------|:----------|:----------|
| **Orchestration** | **LangGraph** (StateGraph) | Proven for hierarchical MAS with conditional routing; Python-native |
| **Vision model** | **Local**: LLaVA, Moondream2, Qwen-VL<br/>**Cloud**: Gemini Pro Vision / GPT-4o | Local VLMs offer offline independence; cloud APIs offer higher reasoning. |
| **Computer Vision Tools**| **PlantCV** & **UPGen** | PlantCV for precise botanical metrics (size, color); UPGen for synthetic data generation to train offline local models. |
| **Sensor processing** | **Qwen 2.5 7B** or **Phi-3-Mini** on edge device | Small, efficient, runs on local CPU gateway entirely offline. |
| **Conversational model** | **Local**: Llama-3 (8B), Mistral-7B<br/>**Cloud**: Gemini Pro / GPT-4o | Local models ensure the farmer can chat without internet connectivity. |
| **Vector DB** | **ChromaDB** (local) or **Vertex AI Vector Search** (cloud) | Depends on scale and deployment model |
| **IoT connectivity** | **MQTT** → time-series DB (InfluxDB / TimescaleDB) | Industry-standard for sensor streams |
| **External tool protocol** | **MCP** | Standardized tool schemas, vendor-agnostic |
| **Agent-to-agent comms** | **A2A Protocol** | For future interop with external agricultural platforms |
| **Interface** | **Gradio** or **WhatsApp API** | Farmer-friendly; voice-capable |

---

## 7. Deployment Architecture

```mermaid
graph LR
    subgraph "Farm Edge (Gateway Device)"
        SE["📡 Sensor Agent<br/>(SLM on CPU)"]
        MQTT["MQTT Broker"]
        CAM["📷 Camera"]
    end

    subgraph "Cloud (GCP / AWS / On-Prem)"
        SUP["🧠 Supervisor Agent"]
        VA["👁️ Vision Agent"]
        FA["🧑‍🌾 Farmer Agent"]
        IA["🔗 Integration Agent"]
        DB["🗄️ Vector DB + State"]
    end

    subgraph "Farmer Interface"
        CHAT["💬 Chat / Voice App"]
        DASH["📊 Dashboard"]
    end

    MQTT --> SE
    CAM --> VA
    SE --> SUP
    VA --> SUP
    FA --> SUP
    IA --> SUP
    SUP --> DB
    FA --> CHAT
    SUP --> DASH
```

**Key design decisions**:
- **Sensor Agent runs at the edge** — processes streams locally, only sends anomalies and summaries to the cloud
- **Vision Agent runs in the cloud** — requires GPU for multimodal inference
- **Farmer Agent is cloud-based** but accessible through mobile (WhatsApp, Gradio, or custom app)
- **All agents write to the Blackboard** in the cloud; the Supervisor synthesizes and triggers actions

---

## 8. MVP Roadmap (Phased)

| Phase | Scope | Duration |
|:------|:------|:---------|
| **Phase 1** | **Frictionless MVP**: "Dumb Green Pixel Counter" (OpenCV/PlantCV) for continuous health proxy + Farmer Agent (chat) + Basic VLM Diagnosis on Anomalies | 4–6 weeks |
| **Phase 2** | Sensor Agent (IoT anomalies) + Supervisor routing + Blackboard integration | 4–6 weeks |
| **Phase 3** | Integration Agent (weather + irrigation control with HITL) | 3–4 weeks |
| **Phase 4** | Advanced Vision Ops (UPGen crop-specific CNNs for precise yield/leaf counting) | 4–6 weeks |

---

## 9. Conclusion

The best architecture for a multi-front plant monitoring service is a **Hierarchical, Cooperative, Heterogeneous MAS** coordinated via a **Blackboard pattern** implemented as a LangGraph `StateGraph`. This mirrors the proven patterns from the Unified Theory (Levels 7–8) while incorporating cutting-edge strategies from the frontier (Edge AI, MCP, Context Engineering).

The key insight is that **each monitoring front maps naturally to a specialized agent** — and the Blackboard pattern ensures that no agent operates in isolation. The whole is greater than the sum of its parts: a Vision Agent that knows the soil is dry (from the Sensor Agent) and the farmer saw yellowing yesterday (from the Farmer Agent) will produce far more accurate diagnoses than any single-modality system.

> *"An agent is an LLM in a `while` loop. A multi-agent system is multiple loops sharing a blackboard."*
