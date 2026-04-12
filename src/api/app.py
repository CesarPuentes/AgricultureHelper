from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import uuid
from pydantic import BaseModel, Field
from typing import Optional
import os
import shutil
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from src.models.models import PlantHealthState, VisionData, DiagnosticMapResult
from src.core.vision import calculate_living_canopy, count_plants, generate_diagnostic_map, count_leaves
from src.core.llm_api import service as disease_classifier
from src.core.anomaly.engine import calculate_anomaly_score
from src.database.manager import init_db, save_readings, get_recent_readings
from src.core.analysis.plant_state import evaluate_reading

os.makedirs("src/static/uploads", exist_ok=True)
os.makedirs("diagnostic_maps_demo", exist_ok=True)
init_db()

app = FastAPI(
    title="AgricultureHelper Tier 1 Agent",
    description="FastAPI shell for the 'Fast Brain' Multi-Agent System.",
    version="0.3.0"
)

app.mount("/static/test_images", StaticFiles(directory="test_images"), name="test_images")
app.mount("/static/maps", StaticFiles(directory="diagnostic_maps_demo"), name="maps")
app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# Simple in-memory results cache for PRG pattern
_UI_RESULTS_CACHE = {}

DEMO_SETS = {
    "counting": {"conteo_4x6": ["GeminiConteo3.jpg", "GeminiConteo4.jpg", "GeminiConteo5.jpg"]},
    "disease": {"disease_1": ["Gemini_plant_disease1.png"]},
    "anomaly": {
        "chlorosis_set": ["Gemini_clorosis.png", "Gemini_clorosis2.png"],
        "disease_set": ["Gemini_plant_disease1.png", "Gemini_plant_disease2.png"],
        "healthy_set": ["Gemini_sana1.png", "Gemini_sana2.png"],
    },
}


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
    
    leaf_count_res = count_leaves(
        request.image_path, 
        rows=request.rows, 
        cols=request.cols, 
        save_map=request.generate_map,
        plant_id=request.plant_id,
        output_dir="diagnostic_maps_demo"
    )
    
    if request.generate_map:
        leaf_count, leaf_map_path = leaf_count_res
        leaf_map_url = f"/static/maps/{os.path.basename(leaf_map_path)}"
    else:
        leaf_count = leaf_count_res
        leaf_map_url = None
    
    vision_data = VisionData(
        living_coverage_pct=coverage, 
        plant_count=plant_count, 
        leaf_count=leaf_count,
        leaf_map_url=leaf_map_url
    )

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
async def dashboard_ui(request: Request, results_id: Optional[str] = None, active_tab: str = "vision"):
    """Render the main dashboard UI, optionally with cached results."""
    # Pull results from cache if present (matches PRG pattern)
    cached_data = _UI_RESULTS_CACHE.pop(results_id, {}) if results_id else {}
    
    ctx = {
        "request": request, 
        "active_tab": cached_data.get("active_tab", active_tab)
    }
    # Merge additional context from cache (results, errors, etc.)
    ctx.update(cached_data.get("data", {}))
    
    return templates.TemplateResponse("index.html", ctx)

@app.post("/ui/analyze", response_class=HTMLResponse)
async def ui_analyze_image(
    request: Request, file: UploadFile = File(...),
    analysis_type: str = Form("counting"), plant_id: str = Form("test_plant_01"),
    rows: int = Form(6), cols: int = Form(4)):
    if not file.filename:
        rid = str(uuid.uuid4())
        _UI_RESULTS_CACHE[rid] = {
            "active_tab": "vision",
            "data": {"vision_error": "No file selected."}
        }
        return RedirectResponse(url=f"/ui?results_id={rid}", status_code=303)
    upload_path = os.path.join("src", "static", "uploads", file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        r = {"original_url": f"/static/uploads/{file.filename}",
             "map_url": None, "leaf_map_url": None, "coverage_pct": None, "plant_count": None, "leaf_count": None, 
             "predictions": None, "anomaly": None}
        if analysis_type == "disease":
            if disease_classifier.is_available():
                r["predictions"] = disease_classifier.classify_disease(upload_path)
        elif analysis_type == "anomaly":
            anomaly = calculate_anomaly_score(upload_path, output_dir="diagnostic_maps_demo")
            # Rewrite debug paths to serveable URLs
            for key in ("chlorosis",):
                if anomaly.get(key, {}).get("debug_path"):
                    anomaly[key]["debug_url"] = f"/static/maps/{os.path.basename(anomaly[key]['debug_path'])}"
            if anomaly.get("canopy_map"):
                anomaly["canopy_url"] = f"/static/maps/{os.path.basename(anomaly['canopy_map'])}"
            r["anomaly"] = anomaly
        else:
            map_info = generate_diagnostic_map(upload_path, plant_id=plant_id, output_dir="diagnostic_maps_demo")
            r["map_url"] = f"/static/maps/{os.path.basename(map_info['map_path'])}"
            r["coverage_pct"] = map_info["coverage_pct"]
            if analysis_type == "counting":
                r["plant_count"] = count_plants(upload_path, rows=rows, cols=cols)
                l_count, l_map = count_leaves(upload_path, rows=rows, cols=cols, save_map=True, plant_id=plant_id, output_dir="diagnostic_maps_demo")
                r["leaf_count"] = l_count
                r["leaf_map_url"] = f"/static/maps/{os.path.basename(l_map)}"
        rid = str(uuid.uuid4())
        _UI_RESULTS_CACHE[rid] = {
            "active_tab": "vision",
            "data": {
                "results": [r] if "r" in locals() and r.get("original_url") else [],
                "result_title": f"Upload: {analysis_type.replace('_', ' ').title()}"
            }
        }
        return RedirectResponse(url=f"/ui?results_id={rid}", status_code=303)
    except Exception as e:
        rid = str(uuid.uuid4())
        _UI_RESULTS_CACHE[rid] = {
            "active_tab": "vision",
            "data": {"vision_error": f"Analysis failed: {str(e)}"}
        }
        return RedirectResponse(url=f"/ui?results_id={rid}", status_code=303)


@app.post("/ui/demo", response_class=HTMLResponse)
async def ui_run_demo(request: Request, demo_type: str = Form(...), image_set: str = Form(...)):
    images = DEMO_SETS.get(demo_type, {}).get(image_set)
    if not images:
        rid = str(uuid.uuid4())
        _UI_RESULTS_CACHE[rid] = {
            "active_tab": "vision",
            "data": {"vision_error": "Invalid demo."}
        }
        return RedirectResponse(url=f"/ui?results_id={rid}", status_code=303)
    results = []
    for name in images:
        path = os.path.join("test_images", name)
        r = {"original_url": f"/static/test_images/{name}",
             "map_url": None, "leaf_map_url": None, "coverage_pct": None, "plant_count": None, "leaf_count": None, "predictions": None}
        r["anomaly"] = None
        if demo_type == "disease":
            if disease_classifier.is_available():
                r["predictions"] = disease_classifier.classify_disease(path)
        elif demo_type == "anomaly":
            anomaly = calculate_anomaly_score(path, output_dir="diagnostic_maps_demo")
            for key in ("chlorosis",):
                if anomaly.get(key, {}).get("debug_path"):
                    anomaly[key]["debug_url"] = f"/static/maps/{os.path.basename(anomaly[key]['debug_path'])}"
            if anomaly.get("canopy_map"):
                anomaly["canopy_url"] = f"/static/maps/{os.path.basename(anomaly['canopy_map'])}"
            r["anomaly"] = anomaly
        else:
            map_info = generate_diagnostic_map(path, plant_id=f"demo_{os.path.splitext(name)[0]}", output_dir="diagnostic_maps_demo")
            r["map_url"] = f"/static/maps/{os.path.basename(map_info['map_path'])}"
            r["coverage_pct"] = map_info["coverage_pct"]
            if demo_type == "counting":
                r["plant_count"] = count_plants(path, rows=6, cols=4)
                l_count, l_map = count_leaves(path, rows=6, cols=4, save_map=True, plant_id=f"demo_{name}", output_dir="diagnostic_maps_demo")
                r["leaf_count"] = l_count
                r["leaf_map_url"] = f"/static/maps/{os.path.basename(l_map)}"
        results.append(r)
            
    rid = str(uuid.uuid4())
    _UI_RESULTS_CACHE[rid] = {
        "active_tab": "vision",
        "data": {
            "results": results, 
            "result_title": f"Demo: {demo_type.title()} — {image_set.replace('_', ' ').title()}"
        }
    }
    return RedirectResponse(url=f"/ui?results_id={rid}", status_code=303)

@app.post("/ui/sensor", response_class=HTMLResponse)
async def ui_sensor_simulation(request: Request, plant_id: str = Form(...), scenario: str = Form("normal")):
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

        if scenario == "heat_spike":
            heat_array[6] = 29.0
            heat_array[7] = 30.5
        elif scenario == "plant_drop":
            count_results[8] = 12
            count_results[9] = 10
        elif scenario == "drought":
            soil_moisture_array[7] = 15.0
            soil_moisture_array[8] = 10.0
            humidity_array[7] = 30.0
            humidity_array[8] = 25.0
        
        readings_to_save = []
        for i in range(n):
            reading = {
                "plant_id": plant_id,
                "timestamp": timestamps[i],
                "air_humidity_rh": humidity_array[i],
                "soil_moisture_pct": soil_moisture_array[i],
                "temperature_c": heat_array[i],
                "light_lux": float(light_array[i]),
                "living_coverage_pct": canopy_results[i],
                "plant_count": int(count_results[i]),
                "leaf_count": int(count_results[i] * 4 + np.random.randint(-2, 3)) if count_results[i] > 0 else 0
            }
            prev = readings_to_save[-1] if readings_to_save else None
            reading["status"] = evaluate_reading(reading, crop="arabidopsis", prev_reading=prev)
            readings_to_save.append(reading)
            
        save_readings(readings_to_save)
        recent = get_recent_readings(plant_id)
        
        rid = str(uuid.uuid4())
        _UI_RESULTS_CACHE[rid] = {
            "active_tab": "sensors",
            "data": {
                "sensor_success": True,
                "sensor_success_msg": f"Successfully saved {n} records to 'agriculture.db' for plant '{plant_id}'!",
                "recent_readings": recent,
                "plant_id": plant_id
            }
        }
        return RedirectResponse(url=f"/ui?results_id={rid}", status_code=303)
    except Exception as e:
        rid = str(uuid.uuid4())
        _UI_RESULTS_CACHE[rid] = {
            "active_tab": "sensors",
            "data": {"sensor_error": f"Database error: {str(e)}"}
        }
        return RedirectResponse(url=f"/ui?results_id={rid}", status_code=303)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
