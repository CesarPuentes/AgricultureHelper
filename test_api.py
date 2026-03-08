from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_analyze_plant_image():
    # Use one of the existing test images
    test_image = "Gemini_sana1.png"
    
    # Ensure the image exists
    assert os.path.exists(test_image), f"Test image {test_image} not found!"

    response = client.post(
        "/api/analyze",
        json={
            "image_path": test_image,
            "plant_id": "test_plant_001"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check the Blackboard structure
    assert data["plant_id"] == "test_plant_001"
    assert "timestamp" in data
    assert "vision" in data
    assert data["vision"]["living_coverage_pct"] > 0
    
    print("Test passed successfully!")
    print("Generated Blackboard State:")
    print(data)

if __name__ == "__main__":
    test_analyze_plant_image()
