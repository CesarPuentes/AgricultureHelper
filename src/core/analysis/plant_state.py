CROP_BASELINES = {
    "arabidopsis": {
        "temp_c": {"min": 18, "max": 24, "target": 22},
        "humidity_rh": {"min": 50, "max": 65, "target": 55},
        "soil_moisture_pct": {"min": 20, "max": 40, "target": 28},
        "light_lux": {"min": 4000, "max": 8000, "target": 6000},
        "notes": "Sensible al calor; más de 28°C induce floración por estrés."
    },
    "lettuce": {
        "temp_c": {"min": 15, "max": 22, "target": 18},
        "humidity_rh": {"min": 50, "max": 80, "target": 65},
        "soil_moisture_pct": {"min": 30, "max": 60, "target": 45},
        "light_lux": {"min": 3500, "max": 7000, "target": 5000},
        "notes": "Prefiere noches frescas y humedad alta para evitar hojas amargas."
    },
    "tomato": {
        "temp_c": {"min": 20, "max": 30, "target": 25},
        "humidity_rh": {"min": 50, "max": 70, "target": 60},
        "soil_moisture_pct": {"min": 40, "max": 70, "target": 55},
        "light_lux": {"min": 10000, "max": 30000, "target": 20000},
        "notes": "Alta demanda de luz; la humedad >80% puede impedir la polinización."
    },
    "wheat": {
        "temp_c": {"min": 10, "max": 25, "target": 18},
        "humidity_rh": {"min": 40, "max": 60, "target": 50},
        "soil_moisture_pct": {"min": 20, "max": 50, "target": 35},
        "light_lux": {"min": 15000, "max": 40000, "target": 25000},
        "notes": "Resistente, pero el calor extremo durante el llenado del grano reduce el rendimiento."
    }
}

def evaluate_reading(reading: dict, crop: str = "arabidopsis") -> str:
    """Evaluate if reading is within nominal crop limits."""
    baseline = CROP_BASELINES.get(crop)
    if not baseline:
        return "normal"
        
    temp = reading.get("temperature_c")
    hum = reading.get("air_humidity_rh")
    moist = reading.get("soil_moisture_pct")
    lux = reading.get("light_lux")
    
    if temp is not None and (temp < baseline["temp_c"]["min"] or temp > baseline["temp_c"]["max"]):
        return "alert"
    if hum is not None and (hum < baseline["humidity_rh"]["min"] or hum > baseline["humidity_rh"]["max"]):
        return "alert"
    if moist is not None and (moist < baseline["soil_moisture_pct"]["min"] or moist > baseline["soil_moisture_pct"]["max"]):
        return "alert"
    if lux is not None and (lux < baseline["light_lux"]["min"] or lux > baseline["light_lux"]["max"]):
        return "alert"
        
    return "normal"