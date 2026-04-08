from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
import os
import shutil
import numpy as np
from datetime import datetime, timedelta

from src.models.models import PlantHealthState, VisionData, AlertResult, DiagnosticMapResult
from src.core.vision_extractor import calculate_living_canopy, count_plants, generate_diagnostic_map
from src.core.alerts import run_all_checks
from src.core import disease_classifier
from src.database.manager import init_db, save_readings, get_recent_readings

os.makedirs("src/static/uploads", exist_ok=True)
init_db()

app = FastAPI(
    title="AgricultureHelper Tier 1 Agent",
    description="FastAPI shell for the 'Fast Brain' Multi-Agent System.",
    version="0.3.0"
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")


class AnalyzeRequest(BaseModel):
    image_path: str
    plant_id: str = "unknown_plant"
    rows: int = Field(6, description="Number of rows in the tray")
    cols: int = Field(4, description="Number of columns in the tray")
    generate_map: bool = Field(False, description="Generate a diagnostic overlay map")


def _validate_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found at {path}")


@app.post("/api/analyze", response_model=PlantHealthState)
async def analyze_plant_image(request: AnalyzeRequest):
    """Trigger the Vision Agent to analyze an image and return the Blackboard state."""
    _validate_image(request.image_path)

    coverage = calculate_living_canopy(request.image_path)
    if coverage is None:
        raise HTTPException(status_code=500, detail="Vision extraction failed.")

    plant_count = count_plants(request.image_path, rows=request.rows, cols=request.cols)
    
    vision_data = VisionData(living_coverage_pct=coverage, plant_count=plant_count)

    needs_review = False
    reason = None
    if coverage < 5.0:
        needs_review = True
        reason = "Living canopy coverage critically low (< 5%). Potential necrosis or missing plant."

    return PlantHealthState(
        plant_id=request.plant_id,
        vision=vision_data,
        requires_farmer_review=needs_review,
        anomaly_reason=reason,
    )


# ---------------------------------------------------------------------------
# Capa 0: Alertas
# ---------------------------------------------------------------------------
@app.get("/api/alerts/{plant_id}")
async def get_alerts(plant_id: str):
    """
    Ejecuta los checks de Capa 0 (Watchdog + Delta de Visión) para un sensor.
    Retorna lista de alertas activas.
    """
    alerts = run_all_checks(plant_id)
    return {
        "plant_id": plant_id,
        "alert_count": len(alerts),
        "alerts": [a.model_dump() for a in alerts],
    }


# ---------------------------------------------------------------------------
# Mapa de Diagnóstico Visual
# ---------------------------------------------------------------------------
class DiagnosticMapRequest(BaseModel):
    image_path: str
    plant_id: str = "unknown_plant"


@app.post("/api/diagnostic-map")
async def create_diagnostic_map(request: DiagnosticMapRequest):
    """Genera un mapa de diagnóstico visual con overlay verde/rojo."""
    _validate_image(request.image_path)

    result = generate_diagnostic_map(request.image_path, plant_id=request.plant_id)
    return {
        "plant_id": request.plant_id,
        **result,
    }


# ---------------------------------------------------------------------------
# Clasificador de Enfermedades (HuggingFace, opcional)
# ---------------------------------------------------------------------------
class ClassifyDiseaseRequest(BaseModel):
    image_path: str
    top_k: int = Field(3, description="Number of top predictions to return")


@app.get("/api/classifier/status")
async def classifier_status():
    """Check if the HuggingFace disease classifier is available."""
    return {
        "available": disease_classifier.is_available(),
        "model": "onuralp/resnet18-plantvillage" if disease_classifier.is_available() else None,
        "note": "Install transformers + torch to enable" if not disease_classifier.is_available() else "Ready",
    }


@app.post("/api/classify-disease")
async def classify_disease(request: ClassifyDiseaseRequest):
    """
    Clasifica posibles enfermedades usando ResNet18-PlantVillage.
    Requiere transformers + torch instalados.
    """
    if not disease_classifier.is_available():
        raise HTTPException(
            status_code=503,
            detail="Clasificador no disponible. Instalar: pip install transformers torch torchvision"
        )

    _validate_image(request.image_path)

    results = disease_classifier.classify_disease(request.image_path, top_k=request.top_k)
    if results is None:
        raise HTTPException(status_code=500, detail="Error en clasificación.")

    return {
        "image_path": request.image_path,
        "predictions": results,
    }


# ---------------------------------------------------------------------------
# UI Routes (Jinja2 Templates)
# ---------------------------------------------------------------------------
@app.get("/ui", response_class=HTMLResponse)
async def dashboard_ui(request: Request):
    """Render the main dashboard UI."""
    return templates.TemplateResponse("index.html", {"request": request, "active_tab": "vision"})

@app.post("/ui/analyze", response_class=HTMLResponse)
async def ui_analyze_image(
    request: Request,
    file: UploadFile = File(...),
    plant_id: str = Form("test_plant_01"),
    rows: int = Form(6),
    cols: int = Form(4)
):
    """Handle image upload and vision analysis from the UI."""
    if not file.filename:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "active_tab": "vision",
            "vision_error": "No file selected."
        })
    
    upload_path = os.path.join("src", "static", "uploads", file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        coverage = calculate_living_canopy(upload_path)
        if coverage is None:
            raise ValueError("Vision extraction failed.")
            
        plant_count = count_plants(upload_path, rows=rows, cols=cols)
        vision_data = VisionData(living_coverage_pct=coverage, plant_count=plant_count)
        
        needs_review = coverage < 5.0
        reason = "Living canopy coverage critically low (< 5%). Potential necrosis or missing plant." if needs_review else None
        
        health_state = PlantHealthState(
            plant_id=plant_id,
            vision=vision_data,
            requires_farmer_review=needs_review,
            anomaly_reason=reason
        )
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "active_tab": "vision",
            "vision_data": health_state,
            "uploaded_filename": file.filename
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "active_tab": "vision",
            "vision_error": f"Analysis failed: {str(e)}"
        })

@app.post("/ui/sensor", response_class=HTMLResponse)
async def ui_sensor_simulation(request: Request, plant_id: str = Form(...)):
    """Generate and save 10 days of sensor history, then show it."""
    try:
        n = 10
        start_time = datetime.now() - timedelta(days=n-1)
        timestamps = [(start_time + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]
        
        humidity_array = np.random.normal(55, 2, n).round(1)
        soil_moisture_array = np.random.normal(30, 3, n).round(1)
        heat_array = np.random.normal(22, 0.5, n).round(1)
        light_array = np.random.randint(5000, 6500, size=n)
        
        canopy_results = np.linspace(1.0, 35.5, n).round(1).tolist()
        count_results = [0, 2, 5, 8, 10, 12, 14, 14, 14, 14]
        
        readings_to_save = []
        for i in range(n):
            readings_to_save.append({
                "plant_id": plant_id,
                "timestamp": timestamps[i],
                "air_humidity_rh": humidity_array[i],
                "soil_moisture_pct": soil_moisture_array[i],
                "temperature_c": heat_array[i],
                "light_lux": float(light_array[i]),
                "living_coverage_pct": canopy_results[i],
                "plant_count": int(count_results[i])
            })
            
        save_readings(readings_to_save)
        recent = get_recent_readings(plant_id)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "active_tab": "sensors",
            "sensor_success": True,
            "sensor_success_msg": f"Successfully saved {n} records to 'agriculture.db' for plant '{plant_id}'!",
            "recent_readings": recent,
            "plant_id": plant_id
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "active_tab": "sensors",
            "sensor_error": f"Database error: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
