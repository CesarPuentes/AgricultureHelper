import streamlit as st
import requests
from PIL import Image
import os
import numpy as np
from datetime import datetime, timedelta
from src.database.manager import init_db, save_readings, get_recent_readings

# Initialize DB
init_db()

# cantidad de muestras
n = 10

# Initialize session state for vision results if not present (Fictional Data)
if "canopy_results" not in st.session_state:
    st.session_state.canopy_results = np.linspace(1.0, 35.5, n).round(1).tolist()
if "count_results" not in st.session_state:
    st.session_state.count_results = [0, 2, 5, 8, 10, 12, 14, 14, 14, 14]

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

st.set_page_config(page_title="AgricultureHelper Lab", page_icon="🌱", layout="wide")

st.title("🌱 AgricultureHelper: Testing Lab")
st.markdown("Experiment with vision processing and simulated sensor telemetry.")

tab_vision, tab_sensors = st.tabs(["📸 Vision Lab", "📟 Sensor Lab"])

with tab_vision:
    st.header("Computer Vision Analysis")
    col_up, col_prev = st.columns([1, 1])
    
    with col_up:
        # Selection for pre-loaded samples
        samples = ["None (Upload Own)", "GeminiConteo1.jpg", "GeminiConteo2.jpg", "GeminiConteo3.jpg", "GeminiConteo4.jpg", "GeminiConteo5.jpg"]
        selected_sample = st.selectbox("Select a Sample Image:", options=samples)
        
        uploaded_file = st.file_uploader("OR Upload a New Image", type=["jpg", "jpeg", "png"])
        
        c_pid, c_rows, c_cols = st.columns(3)
        v_plant_id = c_pid.text_input("Plant ID:", value="test_plant_01")
        tray_rows = c_rows.number_input("Tray Rows:", min_value=1, value=6)
        tray_cols = c_cols.number_input("Tray Cols:", min_value=1, value=4)
        
        analyze_btn = st.button("Analyze Image", type="primary")

    image_path = None
    
    # Priority 1: Manual Upload
    if uploaded_file is not None:
        os.makedirs("test_images", exist_ok=True)
        temp_path = os.path.join("test_images", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = temp_path
        with col_prev:
            st.image(Image.open(temp_path), caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
    
    # Priority 2: Selected Sample
    elif selected_sample != "None (Upload Own)":
        sample_path = os.path.join("test_images", selected_sample)
        if os.path.exists(sample_path):
            image_path = sample_path
            with col_prev:
                st.image(Image.open(sample_path), caption=f"Sample: {selected_sample}", use_container_width=True)
        else:
            st.error(f"Sample file {selected_sample} not found in test_images/")
            
    else:
        with col_prev:
            st.info("Select a sample or upload an image to see the preview.")

    if analyze_btn:
        if not image_path:
            st.error("Please select or upload an image first.")
        else:
            with st.spinner("Processing vision pipeline..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/api/analyze", 
                        json={
                            "image_path": image_path, 
                            "plant_id": v_plant_id,
                            "rows": tray_rows,
                            "cols": tray_cols
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Analysis Complete!")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Canopy Coverage", f"{data['vision']['living_coverage_pct']:.2f}%")
                        c2.metric("Plant Count", data['vision'].get('plant_count', 'N/A'))
                        status = "Review Required ⚠️" if data['requires_farmer_review'] else "Healthy ✅"
                        c3.metric("System Status", status)
                        
                        if data['requires_farmer_review']:
                            st.warning(f"**Anomaly detected:** {data['anomaly_reason']}")
                        
                        with st.expander("Raw API Response"):
                            st.json(data)
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

with tab_sensors:
    st.header("Sensor Telemetry Simulation")
    st.info("Viewing a 10-day history window with fictional vision and sensor data.")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Simulated Data")
        time_sensor = st.text_area("Timestamps (10 Days):", value="\n".join(timestamps), height=150)
        
        # Vision metrics inputs
        canopy_input = st.text_area("Canopy Coverage (%):", value=", ".join(map(str, st.session_state.canopy_results)))
        count_input = st.text_area("Plant Count:", value=", ".join(map(str, st.session_state.count_results)))
        
        # Update session state with edited values
        try:
            st.session_state.canopy_results = [float(x.strip()) for x in canopy_input.split(",")]
            st.session_state.count_results = [int(float(x.strip())) for x in count_input.split(",")]
        except Exception:
            pass # Ignore temporary errors while typing
            
        # Sensor metrics inputs
        air_hum_sensor = st.text_area("Air Humidity (RH%):", value=", ".join(map(str, humidity_array)))
        soil_hum_sensor = st.text_area("Soil Moisture (%):", value=", ".join(map(str, soil_moisture_array)))
        heat_sensor = st.text_area("Air Temperature (°C):", value=", ".join(map(str, heat_array)))
        light_sensor = st.text_area("Light Intensity (Lux):", value=", ".join(map(str, light_array)))

    with col_r:
        st.subheader("Multi-Variable Correlation")
        # Chart data including vision metrics
        chart_df = {
            "Day": list(range(1, n+1)),
            "Soil Moisture (%)": soil_moisture_array,
            "Air Temp (°C)": heat_array,
            "Canopy Coverage (%)": st.session_state.canopy_results,
            "Plant Count": st.session_state.count_results
        }
        st.line_chart(chart_df, x="Day", y=["Soil Moisture (%)", "Air Temp (°C)", "Canopy Coverage (%)", "Plant Count"])
        
        st.subheader("Summary Metrics")
        stat_c1, stat_c2, stat_c3 = st.columns(3)
        stat_c1.metric("Avg Soil Moisture", f"{np.mean(soil_moisture_array):.1f}%")
        stat_c2.metric("Avg Temp", f"{np.mean(heat_array):.1f}°C")
        
        avg_canopy = np.mean(st.session_state.canopy_results)
        delta_canopy = st.session_state.canopy_results[-1] - st.session_state.canopy_results[0]
        stat_c3.metric("Avg Canopy Coverage", f"{avg_canopy:.1f}%", delta=f"{delta_canopy:.1f}%")

    st.divider()
    if st.button("💾 Save history to Local Database", type="primary", use_container_width=True):
        try:
            readings_to_save = []
            for i in range(n):
                reading = {
                    "plant_id": v_plant_id,
                    "timestamp": timestamps[i],
                    "air_humidity_rh": humidity_array[i],
                    "soil_moisture_pct": soil_moisture_array[i],
                    "temperature_c": heat_array[i],
                    "light_lux": float(light_array[i]),
                    "living_coverage_pct": st.session_state.canopy_results[i],
                    "plant_count": int(st.session_state.count_results[i])
                }
                readings_to_save.append(reading)
            
            save_readings(readings_to_save)
            st.success(f"Successfully saved {n} records to 'agriculture.db' for plant '{v_plant_id}'!")
            
            with st.expander("View Saved Data"):
                recent = get_recent_readings(v_plant_id)
                st.table(recent)
        except Exception as e:
            st.error(f"Error saving to database: {e}")


