For English, go to [this link](README_en.md)

# 🌱 AgricultureHelper: Sistema de Monitoreo de Plantas Multi-Agente

Bienvenido al repositorio de **AgricultureHelper**. Este proyecto tiene como objetivo construir un Sistema Multi-Agente (MAS) robusto, priorizando el procesamiento local (edge), para monitorear la salud de las plantas utilizando una combinación de sensores ambientales y visión artificial.

![alt text](Sample.png)

## 🎯 Enfoque Actual del Proyecto: MVP Nivel 1

Actualmente nos encontramos en las etapas iniciales de planificación y desarrollo, enfocándonos completamente en un **MVP Nivel 1 sin fricciones**, diseñado para dispositivos de bajo costo (ej. Raspberry Pi) en entornos de invernaderos remotos o sin conexión a internet.

La arquitectura se basa en un **Patrón de Pizarra (Blackboard)** y está orquestada a través de **LangGraph**, lo que garantiza que más adelante podamos conectar sin problemas modelos pesados de Visión-Lenguaje (VLMs) y APIs en la nube en futuras fases de escalamiento.

### 🚀 Hoja de Ruta del Desarrollo Nivel 1

*   **Fase 1: Extractores "Simples"**
    *   Configurar pipelines de OpenCV en Python para extraer continuamente una "Relación de Verde" HSV como indicador universal del crecimiento/salud de las plantas.
    *   Desplegar brokers MQTT para ingestar sensores básicos de IoT (suelo/temperatura) en una base de datos local SQLite/TimescaleDB.
*   **Fase 2: El Cerebro de Anomalías**
    *   Implementar detección de anomalías ligera en el dispositivo (ej. `SciKit-Learn IsolationForest`).
    *   Establecer la Pizarra unificada `PlantHealthState`.
*   **Fase 3: El Enrutador de Investigación**
    *   Construir el supervisor de LangGraph para enrutar la lógica (ej. *¿Se está reduciendo la cobertura verde porque la humedad es baja?*).
*   **Fase 4: Interfaz para el Agricultor**
    *   Desplegar una interfaz local (SMS o Gradio) para alertar al humano "en el circuito" cada vez que ocurra una anomalía visual desconocida.

## 🛠️ Cómo ejecutar el proyecto

### 1. Requisitos y Entorno Virtual
Se recomienda utilizar un entorno virtual para instalar las dependencias de manera aislada:
```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
# .venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Levantar los Servicios

Para probar el flujo completo con la API y la interfaz visual:

#### Servicio Principal (FastAPI + Interfaz Web)
El servidor ahora procesa las imágenes, interactúa con los sensores simulados y sirve la interfaz web (frontend) directamente usando Jinja2.
```bash
uvicorn src.api.app:app --reload
```
- La **API base y de análisis** estará disponible de forma nativa en `http://localhost:8000`.
- La **Interfaz de Usuario (Frontend)** se puede abrir en tu navegador ingresando a `http://localhost:8000/ui`.

### 📸 Pipeline de Visión y Mapas Diagnósticos

El sistema de visión (ubicado en `src/core/vision_extractor.py`) está diseñado de forma modular para:
- Calcular el porcentaje de cobertura verde viva (canopy).
- Contar el número de plantas individuales analizando bandejas de cultivo en cuadrícula.

**Viendo los Mapas de Diagnóstico:**
Al probar las imágenes desde el frontend o con los tests, el pipeline genera automáticamente un archivo de imagen combinado en formato PNG guardándolo en la carpeta `diagnostic_maps_demo/`. Estos "mapas" proveen *feedback visual* verificable:
- **Verde Semitransparente**: Capa superpuesta indicando el tejido verde/vivo detectado.
- **Rojo Semitransparente**: Capa superpuesta indicando lo ignorado (suelo, maceta, o tejido muerto).
- **Contornos Blancas**: Borde de las hojas o plantas para comprobar la exactitud de segmentación y conteo.
- Valores como el Porcentaje de Cobertura quedan impresos sobre la propia imagen.

#### Probar los scripts localmente en CLI
```bash
# Correr el archivo de pruebas y generación unitaria
python test_api.py

# Correr la suite de pruebas unitarias
pytest test_api.py
```

---

### 📷 Consideraciones Técnicas: Ground Sample Distance (GSD)
Actualmente, el algoritmo de visión (`src/core/vision_extractor.py`) depende de umbrales estáticos en píxeles (ej. eliminar ruido menor a 200px). Esto significa que el sistema es susceptible a errores si la distancia de la cámara a la bandeja cambia (modificando el GSD).

*   **A futuro:** Se recomienda implementar un sistema de calibración dinámica (ej. usando un marcador físico de tamaño conocido en la bandeja) para que el algoritmo calcule la relación de *píxeles/cm* y ajuste los parámetros automáticamente sin importar la altura de la cámara.

---

*Para un desglose completo del diseño del agente y la arquitectura teórica (incluyendo el escalamiento a Nivel 2 y Nivel 3), consulte `multiagent_plant_monitoring_proposal.md`.*
