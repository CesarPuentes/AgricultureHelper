from fastapi.testclient import TestClient
from app import app
import os

from vision_extractor import calculate_living_canopy, count_plants, measure_plant_sizes

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


def test_count_plants_healthy():
    count = count_plants(HEALTHY_IMAGE)
    assert count is not None
    assert count >= 1


def test_count_plants_multi():
    count = count_plants(MULTI_PLANT)
    assert count is not None
    assert count >= 1


def test_measure_plant_sizes():
    sizes = measure_plant_sizes(MULTI_PLANT)
    assert sizes is not None
    assert len(sizes) >= 1
    for plant in sizes:
        assert plant["area"] >= 0
        assert plant["height"] >= 0
        assert plant["width"] >= 0


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


def test_count_endpoint():
    response = client.post("/api/count", json={
        "image_path": MULTI_PLANT,
        "plant_id": "test_tray_001",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["vision"]["plant_count"]["estimated_count"] >= 1
    assert isinstance(data["vision"]["plant_count"]["plants"], list)


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

    # Plant count & sizes
    for name in ["plant.png", "plants.png"]:
        path = os.path.join(TEST_IMAGES_DIR, name)
        if not os.path.exists(path):
            continue
        n = count_plants(path)
        print(f"\n[{name}] Estimated plant count: {n}")
        sizes = measure_plant_sizes(path)
        if sizes:
            for p in sizes:
                print(f"  {p['label']}: area={p['area']}, h={p['height']}, w={p['width']}, "
                      f"perimeter={p['perimeter']:.1f}, solidity={p['solidity']:.3f}")