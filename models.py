from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PlantSizeData(BaseModel):
    """Size and shape measurements for a single plant segment."""
    label: str = Field(..., description="Segment identifier, e.g. 'plant_1'.")
    area: float = Field(..., description="Object area in pixels.")
    height: float = Field(..., description="Object bounding height in pixels.")
    width: float = Field(..., description="Object bounding width in pixels.")
    perimeter: float = Field(..., description="Object perimeter in pixels.")
    solidity: float = Field(..., description="Ratio of area to convex hull area (0-1).")


class PlantCountData(BaseModel):
    """Watershed-based plant count and per-plant size measurements."""
    estimated_count: int = Field(..., description="Estimated number of individual plants.")
    plants: List[PlantSizeData] = Field(
        default_factory=list,
        description="Per-plant size measurements (may be empty if only counting)."
    )


class VisionData(BaseModel):
    living_coverage_pct: float = Field(
        ...,
        description="Percentage of the image area covered by living green tissue (ExG index)."
    )
    plant_count: Optional[PlantCountData] = Field(
        None,
        description="Plant count and size data from watershed segmentation."
    )


class SensorData(BaseModel):
    temperature_c: Optional[float] = Field(None, description="Current ambient temperature in Celsius.")
    soil_moisture_pct: Optional[float] = Field(None, description="Soil moisture percentage (0-100%).")


class PlantHealthState(BaseModel):
    """
    Central 'Blackboard' model for the MAS. Every agent reads from
    and writes to this state during a LangGraph routing cycle.
    """
    plant_id: str = Field(..., description="Unique identifier for the plant/pot being monitored.")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this state was recorded.")

    # Agent Outputs
    vision: Optional[VisionData] = Field(None, description="Data extracted by the Vision Agent pipeline.")
    sensors: Optional[SensorData] = Field(None, description="Data gathered by the physical Sensor Agent.")

    # LangGraph Routing Flags
    requires_farmer_review: bool = Field(
        False,
        description="Flag set by the Supervisor Agent if anomalies are detected."
    )
    anomaly_reason: Optional[str] = Field(
        None,
        description="Human-readable reason for why the farmer needs to intervene."
    )
