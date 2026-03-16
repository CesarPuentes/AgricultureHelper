from fastapi.testclient import TestClient
from app import app
import os

from vision_extractor import calculate_living_canopy, count_plants

client = TestClient(app)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

MULTI_PLANT = os.path.join(TEST_IMAGES_DIR, "plants.png")
HEALTHY_IMAGE = os.path.join(TEST_IMAGES_DIR, "Gemini_sana1.png")


# ---------------------------------------------------------------------------
# Unit tests — vision functions
# ---------------------------------------------------------------------------
def test_living_canopy():
    result = calculate_living_canopy(HEALTHY_IMAGE)
    assert result is not None
    assert 0 < result <= 100


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


# ---------------------------------------------------------------------------
# CLI quick-run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Working dir: {BASE_DIR}\n")

    # Canopy coverage
    for name in ["Gemini_sana1.png", "Gemini_clorosis.png"]:
        path = os.path.join(TEST_IMAGES_DIR, name)
        if os.path.exists(path):
            cov = calculate_living_canopy(path)
            print(f"[{name}] Living Canopy Coverage: {cov}%")

    # Plant Counting Test
    print("\n" + "-" * 40)
    print("Test: Conteo de plantas (Bandeja)")
    print("-" * 40)
    
    # Intenta usar la imagen por defecto del script, sino hace fallback
    test_img = "./test_images/GeminiConteo5.jpg"
    if not os.path.exists(test_img):
        test_img = MULTI_PLANT

    if os.path.exists(test_img):
        print(f"Analizando imagen: {test_img}")
        try:
            result = count_plants(image_path=test_img)
            print(f"Resultado -> Plantas detectadas: {result}")
        except Exception as e:
            print(f"Error al ejecutar count_plants: {e}")
    else:
        print(f"Imagen para conteo ({test_img}) no encontrada. Saltando este test.")