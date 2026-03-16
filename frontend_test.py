import streamlit as st
import requests
from PIL import Image
import os
import numpy as np

# cantidad de muestras
n = 10


# 1. Humedad: Prefieren ambientes moderadamente húmedos (50-60%)
# Usamos una distribución normal (media=55, desviación=2) para que sea más natural
humidity_array = np.random.normal(55, 2, n).round(1)

# 2. Temperatura: Clave para evitar floración prematura. 
# Lo ideal son 22°C constantes. Simulamos pequeñas fluctuaciones del aire acondicionado.
heat_array = np.random.normal(22, 0.5, n).round(1)

# 3. Luz (PAR): En investigación se mide en µmol/m²/s, pero si usas Lux,
# para Arabidopsis lo normal son unos 5,000 - 8,000 Lux (intensidad moderada).
light_array = np.random.randint(5000, 6500, size=n)

st.set_page_config(page_title="Vision Tester", page_icon="🌱", layout="centered")

st.title("🌱 AgricultureHelper Vision Tester")
st.markdown("Test the FastAPI vision extraction backend using local images.")
# Upload an image file
uploaded_file = st.file_uploader("Upload a Plant Image", type=["jpg", "jpeg", "png"])
plant_id = st.text_input("Plant ID:", value="test_plant_01")
humidity_sensor = st.text_input("Humidity simulated values:", value=", ".join(map(str, humidity_array)))
heat_sensor = st.text_input("Heat simulated values:", value=", ".join(map(str, heat_array)))
light_sensor = st.text_input("Light simulated values:", value=", ".join(map(str, light_array)))

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
