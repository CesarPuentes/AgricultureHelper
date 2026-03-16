from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class VisionData(BaseModel):
    living_coverage_pct: float = Field(
        ...,
        description="Percentage of the image area covered by living green tissue (ExG index)."
    )
    plant_count: Optional[int] = Field(
        None,
        description="Plant count from watershed segmentation."
    )

class SensorReading(BaseModel):
    """Representa una única lectura en un momento específico."""
    timestamp: datetime = Field(..., description="Timestamp of the specific reading.")
    temperature_c: Optional[float] = Field(None, description="Ambient temperature in Celsius.")
    light_lux: Optional[float] = Field(None, description="Light intensity in Lux.")
    soil_moisture_pct: Optional[float] = Field(None, description="Soil moisture percentage (0-100%).")
    air_humidity_rh: Optional[float] = Field(None, description="Relative air humidity (0-100%).")

class PlantSensorHistory(BaseModel):
    """Agrupa los 10 días de historia para una planta específica."""
    plant_id: str = Field(..., example="test_plant_01")
    readings: List[SensorReading] = Field(..., description="List of 10 consecutive daily readings.")

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
