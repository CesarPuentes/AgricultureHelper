from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from models import PlantHealthState, VisionData, PlantCountData, PlantSizeData
from vision_extractor import calculate_living_canopy, count_plants, measure_plant_sizes

app = FastAPI(
    title="AgricultureHelper Tier 1 Agent",
    description="FastAPI shell for the 'Fast Brain' Multi-Agent System.",
    version="0.2.0"
)


class AnalyzeRequest(BaseModel):
    image_path: str
    plant_id: str = "unknown_plant"


class CountRequest(BaseModel):
    image_path: str
    plant_id: str = "unknown_plant"


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

    vision_data = VisionData(living_coverage_pct=coverage)

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


@app.post("/api/count", response_model=PlantHealthState)
async def count_and_measure(request: CountRequest):
    """
    Count individual plants via watershed segmentation and measure each one's size.
    Also computes living canopy coverage.
    """
    _validate_image(request.image_path)

    # 1. Living canopy coverage
    coverage = calculate_living_canopy(request.image_path)
    if coverage is None:
        raise HTTPException(status_code=500, detail="Vision extraction failed.")

    # 2. Plant count
    estimated = count_plants(request.image_path)
    if estimated is None:
        raise HTTPException(status_code=500, detail="Plant counting failed.")

    # 3. Per-plant sizes
    raw_sizes = measure_plant_sizes(request.image_path) or []
    plants = [PlantSizeData(**p) for p in raw_sizes]

    # 4. Assemble vision data
    plant_count_data = PlantCountData(estimated_count=estimated, plants=plants)
    vision_data = VisionData(living_coverage_pct=coverage, plant_count=plant_count_data)

    # 5. Anomaly detection
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
