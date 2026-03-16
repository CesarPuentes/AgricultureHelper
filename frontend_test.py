import streamlit as st
import requests
from PIL import Image
import os
import numpy as np
from datetime import datetime, timedelta

# cantidad de muestras
n = 10
# Definimos el punto de inicio (hace 10 días desde hoy, por ejemplo)
start_time = datetime.now() - timedelta(days=n-1)

# CAMBIO MÍNIMO: Usamos days=i en lugar de hours=i
timestamps = [(start_time + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]

# 1. Humedad del AIRE (Relativa)
humidity_array = np.random.normal(55, 2, n).round(1)

# 2. Humedad del SUELO (Sustrato)
soil_moisture_array = np.random.normal(30, 3, n).round(1)

# 3. Temperatura del Aire
heat_array = np.random.normal(22, 0.5, n).round(1)

# 4. Luz (Lux)
light_array = np.random.randint(5000, 6500, size=n)

st.set_page_config(page_title="Vision Tester", page_icon="🌱", layout="centered")

st.title("🌱 AgricultureHelper Vision Tester")
st.markdown("Test the FastAPI vision extraction backend using local images.")

uploaded_file = st.file_uploader("Upload a Plant Image", type=["jpg", "jpeg", "png"])
plant_id = st.text_input("Plant ID:", value="test_plant_01")

# Actualizamos el label para reflejar que son días distintos
time_sensor = st.text_input("Timestamps (10 Consecutive Days):", value=", ".join(timestamps))

# Sensores
air_hum_sensor = st.text_input("Air Humidity (RH%):", value=", ".join(map(str, humidity_array)))
soil_hum_sensor = st.text_input("Soil Moisture (%):", value=", ".join(map(str, soil_moisture_array)))
heat_sensor = st.text_input("Air Temperature (°C):", value=", ".join(map(str, heat_array)))
light_sensor = st.text_input("Light Intensity (Lux):", value=", ".join(map(str, light_array)))
image_path = None

if uploaded_file is not None:
    # Save the file temporarily
    os.makedirs("test_images", exist_ok=True)
    temp_path = os.path.join("test_images", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    image_path = temp_path
    
    # Show the image
    try:
        img = Image.open(temp_path)
        st.image(img, caption=f"Preview: {uploaded_file.name}", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image preview: {e}")
else:
    st.info("Please upload an image to begin.")

if st.button("Analyze Plant", type="primary"):
    if not image_path or not os.path.exists(image_path):
        st.error("Please upload an image first.")
    else:
        with st.spinner("Analyzing plant..."):
            try:
                # Call the FastAPI backend
                response = requests.post(
                    "http://localhost:8000/api/analyze", 
                    json={"image_path": image_path, "plant_id": plant_id}
                )
                
                if response.status_code == 200:
                    st.success("Analysis Complete!")
                    data = response.json()
                    
                    # Display results nicely
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Living Canopy Coverage", f"{data['vision']['living_coverage_pct']:.2f}%")
                    with col2:
                        count = data['vision'].get('plant_count', 'N/A')
                        st.metric("Plant Count", str(count))
                        
                    if data['requires_farmer_review']:
                        st.warning(f"Alert: {data['anomaly_reason']}")
                        
                    with st.expander("Show raw JSON response"):
                        st.json(data)
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Is the FastAPI server running? (Run `uvicorn app:app --reload`)")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
