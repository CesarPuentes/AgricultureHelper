from fastapi.testclient import TestClient
from src.api.app import app
import os
import sqlite3
from datetime import datetime, timedelta

from src.core.vision_extractor import calculate_living_canopy, count_plants, generate_diagnostic_map
from src.core.alerts import check_watchdog, check_vision_delta, run_all_checks
from src.core import disease_classifier

client = TestClient(app)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

HEALTHY_IMAGE = os.path.join(TEST_IMAGES_DIR, "Gemini_sana1.png")

# ---------------------------------------------------------------------------
# Helpers — Test DB
# ---------------------------------------------------------------------------
_TEST_DB = os.path.join(BASE_DIR, "_test_alerts.db")


def _setup_test_db():
    """Create a temporary test database with sensor_readings table."""
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)
    conn = sqlite3.connect(_TEST_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            air_humidity_rh FLOAT,
            soil_moisture_pct FLOAT,
            temperature_c FLOAT,
            light_lux FLOAT,
            living_coverage_pct FLOAT,
            plant_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    return _TEST_DB


def _insert_reading(db_path, plant_id, timestamp, coverage_pct=50.0):
    """Insert a test reading into the DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sensor_readings 
        (plant_id, timestamp, air_humidity_rh, soil_moisture_pct, 
         temperature_c, light_lux, living_coverage_pct, plant_count)
        VALUES (?, ?, 60.0, 40.0, 22.0, 500.0, ?, 5)
    ''', (plant_id, timestamp.strftime("%Y-%m-%d %H:%M:%S"), coverage_pct))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Unit tests — vision functions (original)
# ---------------------------------------------------------------------------
def test_living_canopy():
    result = calculate_living_canopy(HEALTHY_IMAGE)
    assert result is not None
    assert 0 < result <= 100


# ---------------------------------------------------------------------------
# Unit tests — diagnostic map
# ---------------------------------------------------------------------------
def test_generate_diagnostic_map():
    """Verify diagnostic map generates a PNG file with coverage data."""
    result = generate_diagnostic_map(HEALTHY_IMAGE, plant_id="test_map")
    assert result is not None
    assert "map_path" in result
    assert "coverage_pct" in result
    assert os.path.exists(result["map_path"])
    assert result["coverage_pct"] > 0
    # Cleanup
    os.remove(result["map_path"])


def test_living_canopy_with_map():
    """Verify calculate_living_canopy returns tuple when save_map=True."""
    result = calculate_living_canopy(HEALTHY_IMAGE, save_map=True, plant_id="test_canopy_map")
    assert isinstance(result, tuple)
    coverage, map_info = result
    assert coverage > 0
    assert os.path.exists(map_info["map_path"])
    os.remove(map_info["map_path"])


# ---------------------------------------------------------------------------
# Unit tests — watchdog alerts
# ---------------------------------------------------------------------------
def test_check_watchdog_no_alert():
    """Recent reading → no alert."""
    import src.database.manager as mgr
    original_path = mgr.DB_PATH
    try:
        mgr.DB_PATH = _setup_test_db()
        now = datetime.now()
        _insert_reading(mgr.DB_PATH, "plant_ok", now - timedelta(minutes=30))
        result = check_watchdog("plant_ok", now=now)
        assert result is None
    finally:
        mgr.DB_PATH = original_path
        if os.path.exists(_TEST_DB):
            os.remove(_TEST_DB)


def test_check_watchdog_alert():
    """Old reading (>2h10m) → watchdog alert."""
    import src.database.manager as mgr
    original_path = mgr.DB_PATH
    try:
        mgr.DB_PATH = _setup_test_db()
        now = datetime.now()
        _insert_reading(mgr.DB_PATH, "plant_dead", now - timedelta(hours=3))
        result = check_watchdog("plant_dead", now=now)
        assert result is not None
        assert result.alert_type == "watchdog"
        assert result.severity == "critical"
        assert "desconectado" in result.message
    finally:
        mgr.DB_PATH = original_path
        if os.path.exists(_TEST_DB):
            os.remove(_TEST_DB)


# ---------------------------------------------------------------------------
# Unit tests — vision delta alerts
# ---------------------------------------------------------------------------
def test_check_vision_delta_no_alert():
    """Small delta (<5%) → no alert."""
    import src.database.manager as mgr
    original_path = mgr.DB_PATH
    try:
        mgr.DB_PATH = _setup_test_db()
        now = datetime.now()
        _insert_reading(mgr.DB_PATH, "plant_stable", now - timedelta(hours=4), coverage_pct=50.0)
        _insert_reading(mgr.DB_PATH, "plant_stable", now, coverage_pct=48.0)
        result = check_vision_delta("plant_stable")
        assert result is None
    finally:
        mgr.DB_PATH = original_path
        if os.path.exists(_TEST_DB):
            os.remove(_TEST_DB)


def test_check_vision_delta_alert():
    """Large delta (>5%) → vision_delta alert."""
    import src.database.manager as mgr
    original_path = mgr.DB_PATH
    try:
        mgr.DB_PATH = _setup_test_db()
        now = datetime.now()
        _insert_reading(mgr.DB_PATH, "plant_wilting", now - timedelta(hours=4), coverage_pct=45.0)
        _insert_reading(mgr.DB_PATH, "plant_wilting", now, coverage_pct=35.0)
        result = check_vision_delta("plant_wilting")
        assert result is not None
        assert result.alert_type == "vision_delta"
        assert result.severity == "critical"
        assert result.details["delta_pct"] == 10.0
    finally:
        mgr.DB_PATH = original_path
        if os.path.exists(_TEST_DB):
            os.remove(_TEST_DB)


# ---------------------------------------------------------------------------
# Unit tests — disease classifier graceful degradation
# ---------------------------------------------------------------------------
def test_disease_classifier_availability():
    """Classifier should report its availability without crashing."""
    # This test passes regardless of whether transformers is installed
    result = disease_classifier.is_available()
    assert isinstance(result, bool)


def test_disease_classifier_no_crash():
    """If transformers not installed, classify_disease returns None gracefully."""
    if not disease_classifier.is_available():
        result = disease_classifier.classify_disease(HEALTHY_IMAGE)
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests — API endpoints
# ---------------------------------------------------------------------------
def test_analyze_endpoint():
    response = client.post("/api/analyze", json={
        "image_path": HEALTHY_IMAGE,
        "plant_id": "test_plant_001",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["plant_id"] == "test_plant_001"
    assert data["vision"]["living_coverage_pct"] > 0


def test_alerts_endpoint():
    """Alerts endpoint returns valid response structure."""
    response = client.get("/api/alerts/nonexistent_plant")
    assert response.status_code == 200
    data = response.json()
    assert "alert_count" in data
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_diagnostic_map_endpoint():
    """Diagnostic map endpoint returns valid result."""
    response = client.post("/api/diagnostic-map", json={
        "image_path": HEALTHY_IMAGE,
        "plant_id": "test_map_api",
    })
    assert response.status_code == 200
    data = response.json()
    assert "map_path" in data
    assert "coverage_pct" in data
    # Cleanup
    if os.path.exists(data["map_path"]):
        os.remove(data["map_path"])


def test_classifier_status_endpoint():
    """Classifier status endpoint returns availability info."""
    response = client.get("/api/classifier/status")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert isinstance(data["available"], bool)


# ---------------------------------------------------------------------------
# CLI quick-run
# ---------------------------------------------------------------------------

# Manual testing

if __name__ == "__main__":
    print(f"Working dir: {BASE_DIR}\n")

    # Canopy coverage
    for name in ["Gemini_sana1.png", "Gemini_clorosis.png"]:
        path = os.path.join(TEST_IMAGES_DIR, name)
        if os.path.exists(path):
            cov = calculate_living_canopy(path)
            print(f"[{name}] Living Canopy Coverage: {cov}%")

    # Diagnostic Map
    print("\n" + "-" * 40)
    print("Test: Generación de Mapa de Diagnóstico")
    print("-" * 40)
    result = generate_diagnostic_map(HEALTHY_IMAGE, plant_id="cli_test")
    print(f"Mapa generado: {result['map_path']}")
    print(f"Cobertura: {result['coverage_pct']}%")

    # Plant Counting Test
    print("\n" + "-" * 40)
    print("Test: Conteo de plantas (Bandeja)")
    print("-" * 40)

    # CONTEO
    
    # Intenta usar la imagen por defecto del script, sino hace fallback
    rows = 6
    columns = 4
    
    lista_4_6 = ["GeminiConteo3.jpg", "GeminiConteo4.jpg", "GeminiConteo5.jpg"]

    for test_img in lista_4_6:
        test_img = f"./test_images/{test_img}"
        if os.path.exists(test_img):
            print(f"Analizando imagen: {test_img}")
            try: 
                result = count_plants(test_img, rows, columns)
                print(f"Resultado -> Plantas detectadas: {result}")
            except Exception as e:
                print(f"Error al ejecutar count_plants: {e}")
        else:
            print(f"Imagen para conteo ({test_img}) no encontrada. Saltando este test.")