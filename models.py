from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class VisionData(BaseModel):
    living_coverage_pct: float = Field(
        ..., 
        description="Percentage of the image area covered by living green tissue (calculated via ExG index)."
    )

class SensorData(BaseModel):
    temperature_c: Optional[float] = Field(None, description="Current ambient temperature in Celsius.")
    soil_moisture_pct: Optional[float] = Field(None, description="Soil moisture percentage (0-100%).")

class PlantHealthState(BaseModel):
    """
    The central 'Blackboard' model for the MAS. Every agent reads from 
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
