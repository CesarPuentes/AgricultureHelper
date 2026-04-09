import os
from .detectors import chlorosis_score, texture_score, hole_score
from src.core.vision_extractor import calculate_living_canopy

def calculate_anomaly_score(image_path: str, output_dir: str = "diagnostic_maps_demo") -> dict:
    """
    Run chlorosis + texture + holes on one image.
    Returns individual scores + weighted composite (0–100).
    """
    chlor = chlorosis_score(image_path, output_dir)
    text = texture_score(image_path, output_dir=output_dir)
    holes = hole_score(image_path, output_dir)
    
    canopy_res = calculate_living_canopy(image_path, save_map=True, plant_id=f"anomaly_{os.path.basename(image_path).split('.')[0]}", output_dir=output_dir)
    if isinstance(canopy_res, tuple):
        canopy_coverage, canopy_map_info = canopy_res
        canopy_map_path = canopy_map_info["map_path"]
    else:
        canopy_coverage = canopy_res or 0.0
        canopy_map_path = None

    # Composite now only reflects chlorosis (weighted 100%)
    composite = chlor["score"]
    needs_attention = composite > 40

    return {
        "anomaly_score": composite,
        "needs_attention": needs_attention,
        "canopy_coverage": canopy_coverage,
        "canopy_map": canopy_map_path,
        "chlorosis": chlor,
        "texture": text,
        "holes": holes,
        "debug_dir": output_dir,
    }
