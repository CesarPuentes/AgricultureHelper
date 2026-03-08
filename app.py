from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from models import PlantHealthState, VisionData
from vision_extractor import calculate_living_canopy

app = FastAPI(
    title="AgricultureHelper Tier 1 Agent",
    description="FastAPI shell for the 'Fast Brain' Multi-Agent System.",
    version="0.1.0"
)

class AnalyzeRequest(BaseModel):
    image_path: str
    plant_id: str = "unknown_plant"

@app.post("/api/analyze", response_model=PlantHealthState)
async def analyze_plant_image(request: AnalyzeRequest):
    """
    Trigger the Vision Agent to analyze an image and return the Blackboard state.
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {request.image_path}")

    # 1. Vision Agent: Extract Data
    coverage = calculate_living_canopy(request.image_path)
    
    if coverage is None:
        raise HTTPException(status_code=500, detail="Vision extraction failed.")

    # 2. Write to Blackboard
    vision_data = VisionData(living_coverage_pct=coverage)
    
    # 3. Future LangGraph Routing (Placeholder)
    # If coverage < 5%, we might set requires_farmer_review = True
    needs_review = False
    reason = None
    if coverage < 5.0:
        needs_review = True
        reason = "Living canopy coverage critically low (< 5%). Potential necrosis or missing plant."

    # Return the populated state
    state = PlantHealthState(
        plant_id=request.plant_id,
        vision=vision_data,
        requires_farmer_review=needs_review,
        anomaly_reason=reason
    )
    
    return state

if __name__ == "__main__":
    import uvicorn
    # Allow running directly via `python app.py` for simplicity
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
