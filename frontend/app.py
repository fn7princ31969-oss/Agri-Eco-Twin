# streamlit_advanced_app.py
import streamlit as st
import requests
import pandas as pd
import altair as alt
import time

# ========================
# App Config
# ========================
st.set_page_config(
    page_title="Agri-Eco-Twin Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Background & CSS
# ========================
st.markdown("""
<style>
.stApp {
    background-image: url('https://images.unsplash.com/photo-1576765607928-23a4d810e9f7?auto=format&fit=crop&w=1400&q=80');
    background-size: cover;
    background-attachment: fixed;
}
.transparent-box {
    background-color: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.big-font { font-size:32px !important; font-weight:bold;}
.stButton>button {
    background-color: #4CAF50;
    color:white;
    font-size:16px;
    height:40px;
    width:100%;
}
</style>
""", unsafe_allow_html=True)

# ========================
# Sidebar
# ========================
st.sidebar.title("Agri-Eco-Twin 🌱")
app_mode = st.sidebar.selectbox(
    "Choose Section",
    [
        "Crop Prediction",
        "Yield Prediction",
        "Water Prediction",
        "Revenue Prediction",
        "Summary Dashboard",
        "Graphs Page",
        "Smart Farm Analysis"   # ✅ ADDED
    ]
)

API_URL = "http://127.0.0.1:8001"

price_df = pd.read_csv("market_price.csv")

# ========================
# Helper Functions
# ========================
def show_loading(msg="Processing..."):
    with st.spinner(msg):
        time.sleep(1)

def plot_bar(df, x, y, color_range=["#2ca02c","#ff7f0e"]):
    chart = alt.Chart(df).mark_bar().encode(
        x=x,
        y=y,
        color=alt.Color(x, scale=alt.Scale(range=color_range)),
        tooltip=[x,y]
    )
    st.altair_chart(chart, use_container_width=True)

def plot_line(df, x, y):
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=x,
        y=y,
        tooltip=[x,y]
    )
    st.altair_chart(chart, use_container_width=True)

def plot_area(df, x, y):
    chart = alt.Chart(df).mark_area(opacity=0.5).encode(
        x=x,
        y=y,
        tooltip=[x,y]
    )
    st.altair_chart(chart, use_container_width=True)

# ========================
# EXISTING PAGES (UNCHANGED)
# ========================
if app_mode == "Crop Prediction":
    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">🌾 Crop Recommendation</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 90)
        P = st.number_input("Phosphorus (P)", 0, 200, 40)
        K = st.number_input("Potassium (K)", 0, 200, 40)
        temperature = st.number_input("Temperature (°C)", 0, 50, 25)
        humidity = st.number_input("Humidity (%)", 0, 100, 70)
    with col2:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0, 500, 200)
        aspect = st.selectbox("Aspect", ["North", "South", "East", "West"])
        soil_texture = st.selectbox("Soil Texture", ["Clay", "Loamy", "Sandy", "Silt"])

    if st.button("Predict Crop"):
        payload = {"N": N,"P":P,"K":K,"temperature":temperature,
                   "humidity":humidity,"ph":ph,"rainfall":rainfall,
                   "aspect":aspect,"soil_texture":soil_texture}
        resp = requests.post(f"{API_URL}/predict_crop", json=payload)
        if resp.status_code == 200:
            st.success(f"🌱 Recommended Crop: {resp.json().get('recommended_crop')}")
        else:
            st.error("Prediction failed.")
    st.markdown('</div>', unsafe_allow_html=True)

# (All your other pages remain same here…)

# ========================
# SMART FARM ANALYSIS (UPDATED)
# ========================
elif app_mode == "Smart Farm Analysis":

    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">🚀 Smart Farm AI Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 90)
        P = st.number_input("Phosphorus (P)", 0, 200, 40)
        K = st.number_input("Potassium (K)", 0, 200, 40)
        temperature = st.number_input("Temperature (°C)", 0, 50, 28)
        humidity = st.number_input("Humidity (%)", 0, 100, 70)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

    with col2:
        rainfall = st.number_input("Rainfall (mm)", 0, 5000, 200)
        aspect = st.selectbox("Aspect", ["North", "South", "East", "West"])
        soil_texture = st.selectbox("Soil Texture", ["Clay", "Loamy", "Sandy", "Silt"])
        area = st.number_input("Area (ha)", 0.1, 1000.0, 5.0)
        fertilizer = st.number_input("Fertilizer (kg)", 0, 5000, 500)
        pesticide = st.number_input("Pesticide (kg)", 0, 1000, 200)
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        weather_condition = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])

    if st.button("Run Smart Analysis"):

        payload = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "aspect": aspect,
            "soil_texture": soil_texture,
            "area": area,
            "fertilizer": fertilizer,
            "pesticide": pesticide,
            "region": region,
            "weather_condition": weather_condition
        }

        resp = requests.post(f"{API_URL}/predict_all", json=payload)

        if resp.status_code == 200:
            data = resp.json()

            st.success("✅ Smart Analysis Completed!")

            # 🔹 Recommended Crop
            st.subheader("🌾 Crop Recommendation")
            st.success(f"{data['recommended_crop']}")

            # 🔹 Metrics Row 1
            colA, colB, colC, colD = st.columns(4)
            colA.metric("🌱 Yield", f"{data['predicted_yield']:.2f}")
            colB.metric("💧 Water", f"{data['predicted_water_requirement']}")
            colC.metric("💰 Price/Unit", f"₹{data['price_per_unit']:,.2f}")
            colD.metric("📈 Revenue", f"₹{data['predicted_revenue']:,.2f}")

            # 🔹 Metrics Row 2
            colE, colF = st.columns(2)
            colE.metric("💵 Total Cost", f"₹{data['total_cost']:,.2f}")
            colF.metric("🟢 Profit", f"₹{data['predicted_profit']:,.2f}")

            # 🔹 Sustainability
            st.subheader("🌍 Sustainability Score")
            score = data["sustainability_score"]
            st.metric("Score", f"{score:.2f}/100")

            if score >= 80:
                grade = "A 🌱 Excellent"
            elif score >= 60:
                grade = "B 👍 Good"
            elif score >= 40:
                grade = "C ⚠ Moderate"
            else:
                grade = "D ❌ Poor"

            st.info(f"Sustainability Grade: {grade}")

            # 🔹 Visualization
            df = pd.DataFrame({
                "Metric":["Yield","Revenue","Profit","Water"],
                "Value":[
                    data["predicted_yield"],
                    data["predicted_revenue"],
                    data["predicted_profit"],
                    data["predicted_water_requirement"]
                ]
            })

            plot_bar(df,"Metric","Value",
                     ["#2ca02c","#ff7f0e","#1f77b4","#17becf"])

        else:
            st.error("❌ Prediction failed. Backend error.")

    st.markdown('</div>', unsafe_allow_html=True)