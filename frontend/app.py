 
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
app_mode = st.sidebar.selectbox("Choose Section", 
                                ["Crop Prediction", "Yield Prediction", "Water Prediction", 
                                 "Revenue Prediction", "Summary Dashboard", "Graphs Page","Smart Farm Analysis"])

API_URL = "http://127.0.0.1:8001"

# ========================
# Load Market Prices CSV
# ========================
price_df = pd.read_csv("market_price.csv")  # must contain 'Crop' and 'Base_Price'

# ========================
# Crop Options
# ========================
YIELD_CROPS = [
    'Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Black pepper',        
    'Cardamom', 'Cashewnut', 'Castor seed', 'Coconut', 'Coriander',
    'Cotton(lint)', 'Cowpea(Lobia)', 'Dry chillies', 'Garlic', 'Ginger', 'Gram', 
    'Groundnut', 'Guar seed', 'Horse-gram', 'Jowar', 'Jute', 'Khesari', 'Linseed',
    'Maize', 'Masoor', 'Mesta', 'Moong(Green Gram)', 'Moth', 'Niger seed',       
    'Oilseeds total', 'Onion', 'Other Rabi pulses', 'Other Cereals',
    'Other Kharif pulses', 'Other Summer Pulses', 'Peas & beans (Pulses)',
    'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Safflower', 'Sannhamp',
    'Sesamum', 'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower',
    'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric', 'Urad', 'Wheat',
    'other oilseeds'
]

WATER_CROPS = [
    'BANANA', 'BEAN', 'CABBAGE', 'CITRUS', 'COTTON', 'MAIZE', 'MELON', 'MUSTARD',
    'ONION', 'POTATO', 'RICE', 'SOYABEAN', 'SUGARCANE', 'TOMATO', 'WHEAT'
]

# ========================
# Helper Functions
# ========================
def show_loading(msg="Processing..."):
    with st.spinner(msg):
        time.sleep(1)

def plot_bar(df, x, y, color_range=["#2ca02c","#ff7f0e"]):
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=x,
        y=y,
        color=alt.Color(x, scale=alt.Scale(range=color_range)),
        tooltip=[x,y]
    )
    st.altair_chart(chart, use_container_width=True)

def plot_line(df, x, y, color="#1f77b4"):
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=x,
        y=y,
        color=alt.value(color),
        tooltip=[x,y]
    )
    st.altair_chart(chart, use_container_width=True)

def plot_area(df, x, y, color="#ff7f0e"):
    chart = alt.Chart(df).mark_area(opacity=0.5, interpolate='monotone').encode(
        x=x, y=y,
        color=alt.value(color),
        tooltip=[x,y]
    )
    st.altair_chart(chart, use_container_width=True)

# ========================
# Crop Prediction
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
        show_loading("Predicting...")
        payload = {"N": N,"P":P,"K":K,"temperature":temperature,"humidity":humidity,
                   "ph":ph,"rainfall":rainfall,"aspect":aspect,"soil_texture":soil_texture}
        resp = requests.post(f"{API_URL}/predict_crop", json=payload)
        if resp.status_code == 200:
            crop_name = resp.json().get("recommended_crop")
            st.success(f"🌱 Recommended Crop: **{crop_name}**")
        else:
            st.error("Prediction failed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Yield Prediction
# ========================
elif app_mode == "Yield Prediction":
    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">📈 Yield Prediction</p>', unsafe_allow_html=True)

    crop = st.selectbox("Crop Name", YIELD_CROPS)
    area = st.number_input("Area (hectares)", 0.1, 1000.0, 2.0)
    rainfall = st.number_input("Annual Rainfall (mm)", 0, 5000, 200)
    fertilizer = st.number_input("Fertilizer (kg)", 0, 5000, 1000)
    pesticide = st.number_input("Pesticide (kg)", 0, 1000, 50)

    if st.button("Predict Yield"):
        show_loading("Predicting yield...")
        payload = {"crop": crop,"area":area,"annual_rainfall":rainfall,"fertilizer":fertilizer,"pesticide":pesticide}
        resp = requests.post(f"{API_URL}/predict_yield", json=payload)
        if resp.status_code==200:
            yld = resp.json().get("predicted_yield")
            st.success(f"🌾 Predicted Yield: **{yld:.2f} tons**")

            # Different graph for yield
            df = pd.DataFrame({"Area":[area],"Yield":[yld]})
            plot_area(df,"Area","Yield","#2ca02c")
        else:
            st.error("Prediction failed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Water Prediction
# ========================
elif app_mode == "Water Prediction":
    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">💧 Water Requirement Prediction</p>', unsafe_allow_html=True)

    crop = st.selectbox("Crop Type", WATER_CROPS)
    soil = st.selectbox("Soil Type",["DRY","HUMID","WET"])
    region = st.selectbox("Region",["DESERT","HUMID","SEMI ARID","SEMI HUMID"])
    temp = st.number_input("Temperature (°C)",0,50,28)
    weather = st.selectbox("Weather Condition",["SUNNY","RAINY","NORMAL","WINDY"])

    if st.button("Predict Water Requirement"):
        show_loading("Calculating water...")
        payload = {"CROP TYPE":crop,"SOIL TYPE":soil,"REGION":region,"TEMPERATURE":temp,"WEATHER CONDITION":weather}
        resp = requests.post(f"{API_URL}/predict_water", json=payload)
        if resp.status_code==200:
            water = resp.json().get("predicted_water_requirement")
            st.success(f"💧 Water Requirement: **{water:.2f} liters**")

            df = pd.DataFrame({"Region":[region],"Water Requirement":[water]})
            plot_line(df,"Region","Water Requirement","#1f77b4")
        else:
            st.error("Prediction failed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Revenue Prediction
# ========================
elif app_mode=="Revenue Prediction":
    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">💰 Revenue Prediction</p>', unsafe_allow_html=True)

    crop = st.selectbox("Crop Name", YIELD_CROPS)
    area = st.number_input("Area (ha)",0.1,1000.0,2.0)
    rainfall = st.number_input("Annual Rainfall",0,5000,200)
    fertilizer = st.number_input("Fertilizer",0,5000,1000)
    pesticide = st.number_input("Pesticide",0,1000,50)

    if st.button("Predict Revenue"):
        show_loading("Predicting revenue...")
        payload = {"CROP":crop,"AREA":area,"ANNUAL_RAINFALL":rainfall,"FERTILIZER":fertilizer,"PESTICIDE":pesticide}
        resp = requests.post(f"{API_URL}/predict_revenue", json=payload)
        if resp.status_code==200:
            data = resp.json()
            yld = data.get("predicted_yield")

            # Get price from CSV
            price_row = price_df[price_df['Crop'] == crop]
            price = float(price_row['Base_Price'].values[0]) if not price_row.empty else 0
            revenue = yld * price

            st.success(f"🌾 Yield: {yld:.2f} tons")
            st.success(f"💵 Price/unit: {price:.2f} INR")
            st.success(f"💰 Revenue: {revenue:.2f} INR")

            df = pd.DataFrame({"Metric":["Yield","Revenue"],"Value":[yld,revenue]})
            plot_bar(df,"Metric","Value",color_range=["#2ca02c","#ff7f0e"])
        else:
            st.error("Prediction failed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Summary Dashboard
# ========================
elif app_mode=="Summary Dashboard":
    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">📊 Summary of All Predictions</p>', unsafe_allow_html=True)

    crop = st.selectbox("Crop Name for Summary", YIELD_CROPS)
    area = st.number_input("Area",0.1,1000.0,2.0,key="sum_area")
    rainfall = st.number_input("Rainfall",0,5000,200,key="sum_rainfall")
    fertilizer = st.number_input("Fertilizer",0,5000,1000,key="sum_fert")
    pesticide = st.number_input("Pesticide",0,1000,50,key="sum_pest")

    if st.button("Get Summary"):
        show_loading("Fetching all predictions...")
        y_resp = requests.post(f"{API_URL}/predict_yield",json={"crop":crop,"area":area,"annual_rainfall":rainfall,"fertilizer":fertilizer,"pesticide":pesticide})
        r_resp = requests.post(f"{API_URL}/predict_revenue",json={"CROP":crop,"AREA":area,"ANNUAL_RAINFALL":rainfall,"FERTILIZER":fertilizer,"PESTICIDE":pesticide})
        w_resp = requests.post(f"{API_URL}/predict_water",json={"CROP TYPE":crop,"SOIL TYPE":"Loamy","REGION":"North","TEMPERATURE":28,"WEATHER CONDITION":"Sunny"})

        if y_resp.status_code==200 and r_resp.status_code==200 and w_resp.status_code==200:
            yld = y_resp.json().get("predicted_yield")

            # Price from CSV
            price_row = price_df[price_df['Crop'] == crop]
            price = float(price_row['Base_Price'].values[0]) if not price_row.empty else 0
            rev = yld * price

            water = w_resp.json().get("predicted_water_requirement")

            summary_df = pd.DataFrame({
                "Metric":["Yield (tons)","Price/unit (INR)","Revenue (INR)","Water Requirement (liters)"],
                "Value":[yld,price,rev,water]
            })

            st.table(summary_df)
            plot_bar(summary_df,"Metric","Value",color_range=["#2ca02c","#1f77b4","#ff7f0e","#17becf"])
        else:
            st.error("Failed to fetch summary.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Graphs Page (All stacked graphs)
# ========================
elif app_mode=="Graphs Page":
    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">📊 All Model Graphs</p>', unsafe_allow_html=True)

    crop = st.selectbox("Crop Name for Graphs", YIELD_CROPS)
    area = st.number_input("Area",0.1,1000.0,2.0,key="graph_area")
    rainfall = st.number_input("Rainfall",0,5000,200,key="graph_rainfall")
    fertilizer = st.number_input("Fertilizer",0,5000,1000,key="graph_fert")
    pesticide = st.number_input("Pesticide",0,1000,50,key="graph_pest")

    if st.button("Show Graphs"):
        show_loading("Fetching predictions...")
        y_resp = requests.post(f"{API_URL}/predict_yield",json={"crop":crop,"area":area,"annual_rainfall":rainfall,"fertilizer":fertilizer,"pesticide":pesticide})
        w_resp = requests.post(f"{API_URL}/predict_water",json={"CROP TYPE":crop,"SOIL TYPE":"Loamy","REGION":"North","TEMPERATURE":28,"WEATHER CONDITION":"Sunny"})

        if y_resp.status_code==200 and w_resp.status_code==200:
            yld = y_resp.json().get("predicted_yield")
            water = w_resp.json().get("predicted_water_requirement")

            # Price from CSV
            price_row = price_df[price_df['Crop'] == crop]
            price = float(price_row['Base_Price'].values[0]) if not price_row.empty else 0
            rev = yld * price

            st.markdown("### Yield Graph")
            df_yld = pd.DataFrame({"Area":[area],"Yield":[yld]})
            plot_area(df_yld,"Area","Yield","#2ca02c")

            st.markdown("### Revenue Graph")
            df_rev = pd.DataFrame({"Metric":["Revenue"],"Value":[rev]})
            plot_bar(df_rev,"Metric","Value",color_range=["#ff7f0e"])

            st.markdown("### Water Requirement Graph")
            df_water = pd.DataFrame({"Region":["North"],"Water Requirement":[water]})
            plot_line(df_water,"Region","Water Requirement","#1f77b4")
        else:
            st.error("Failed to fetch graphs.")
    st.markdown('</div>', unsafe_allow_html=True) 

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