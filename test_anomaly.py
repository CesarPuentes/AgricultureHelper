"""Quick smoke test for anomaly_detector on test images."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.anomaly_detector import calculate_anomaly_score

def test(path, label):
    print(f"\n{'='*50}")
    print(f"  {label}: {path}")
    print(f"{'='*50}")
    result = calculate_anomaly_score(path)
    print(f"  Anomaly Score: {result['anomaly_score']}/100  {'⚠️  NEEDS ATTENTION' if result['needs_attention'] else '✅ OK'}")
    print(f"  ├─ Chlorosis:  {result['chlorosis']['score']}/100  (green={result['chlorosis']['green_px']}px, stress={result['chlorosis']['stress_px']}px)")
    print(f"  ├─ Texture:    {result['texture']['score']}/100  (anomaly={result['texture']['anomaly_px']}px, {result['texture']['anomaly_pct']}%)")
    print(f"  └─ Holes:      {result['holes']['score']}/100  ({result['holes']['hole_count']} holes, {result['holes']['hole_area_pct']}% area)")
    print(f"  Debug images saved to: {result['debug_dir']}/")

if __name__ == "__main__":
    test("test_images/Gemini_sana1.png", "Planta Sana")
    test("test_images/Gemini_plant_disease1.png", "Planta Enferma")
    test("test_images/Gemini_clorosis.png", "Planta con Clorosis")
