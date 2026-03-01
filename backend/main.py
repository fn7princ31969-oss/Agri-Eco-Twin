import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Agri-Eco-Twin", layout="wide")
st.title("🌾 Agri-Eco-Twin Smart Prediction System")

# ==========================
# LOAD FILES
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Market Price
market_prices_df = pd.read_csv("market_price.csv")
crop_prices = {crop.upper(): price for crop, price in zip(market_prices_df["Crop"], market_prices_df["Base_Price"])}

crop_name_mapping = {
    "SOYBEANS": "Soyabean",
    "SOYBEAN": "Soyabean"
}

# Load Models
@st.cache_resource
def load_models():
    models = {}
    models["crop_model"] = joblib.load("ml_models/crop_model.pkl")
    models["aspect_encoder"] = joblib.load("ml_models/aspect_encoder.pkl")
    models["texture_encoder"] = joblib.load("ml_models/texture_encoder.pkl")
    models["crop_label_encoder"] = joblib.load("ml_models/crop_label_encoder.pkl")
    models["yield_model"] = joblib.load("ml_models/yield_model.pkl")
    models["yield_crop_encoder"] = joblib.load("ml_models/yield_crop_encoder.pkl")
    models["water_model"] = joblib.load("ml_models/water_model.pkl")
    models["water_crop_encoder"] = joblib.load("ml_models/water_crop_encoder.pkl")
    models["water_soil_encoder"] = joblib.load("ml_models/water_soil_encoder.pkl")
    models["water_region_encoder"] = joblib.load("ml_models/water_region_encoder.pkl")
    models["water_weather_encoder"] = joblib.load("ml_models/water_weather_encoder.pkl")
    return models

models = load_models()

def safe_encode(encoder, value):
    value = value.upper()
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0

# ==========================
# INPUT SECTION
# ==========================

st.header("🌱 Soil & Environment")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)")
    P = st.number_input("Phosphorus (P)")
    K = st.number_input("Potassium (K)")

with col2:
    temperature = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    ph = st.number_input("pH")

with col3:
    rainfall = st.number_input("Rainfall")
    aspect = st.text_input("Aspect")
    soil_texture = st.text_input("Soil Texture")

st.header("🌾 Farming Inputs")

col4, col5, col6 = st.columns(3)

with col4:
    area = st.number_input("Area (hectare)")

with col5:
    fertilizer = st.number_input("Fertilizer")

with col6:
    pesticide = st.number_input("Pesticide")

st.header("💧 Climate Inputs")

col7, col8 = st.columns(2)

with col7:
    region = st.text_input("Region")

with col8:
    weather_condition = st.text_input("Weather Condition")

# ==========================
# PREDICTION BUTTON
# ==========================

if st.button("🚀 Run Smart Prediction"):

    try:
        # ======================
        # 1️⃣ CROP PREDICTION
        # ======================
        aspect_enc = models["aspect_encoder"].transform([aspect])[0]
        soil_enc = models["texture_encoder"].transform([soil_texture])[0]

        crop_features = np.array([[ 
            N, P, K,
            temperature, humidity,
            ph, rainfall,
            aspect_enc, soil_enc
        ]])

        crop_prediction = models["crop_model"].predict(crop_features)[0]
        crop_name = models["crop_label_encoder"].inverse_transform([crop_prediction])[0]

        if crop_name.upper() in crop_name_mapping:
            crop_name = crop_name_mapping[crop_name.upper()]

        # ======================
        # 2️⃣ YIELD PREDICTION
        # ======================
        crop_encoded_yield = models["yield_crop_encoder"].transform([crop_name])[0]

        yield_features = np.array([[ 
            crop_encoded_yield,
            area,
            rainfall,
            fertilizer,
            pesticide
        ]])

        predicted_yield = models["yield_model"].predict(yield_features)[0]

        # ======================
        # 3️⃣ WATER PREDICTION
        # ======================
        crop_encoded_water = safe_encode(models["water_crop_encoder"], crop_name)
        soil_encoded = safe_encode(models["water_soil_encoder"], soil_texture)
        region_encoded = safe_encode(models["water_region_encoder"], region)
        weather_encoded = safe_encode(models["water_weather_encoder"], weather_condition)

        water_features = np.array([[ 
            crop_encoded_water,
            soil_encoded,
            region_encoded,
            temperature,
            weather_encoded
        ]])

        predicted_water = models["water_model"].predict(water_features)[0]

        # ======================
        # 4️⃣ REVENUE
        # ======================
        price = crop_prices.get(crop_name.upper(), 0)
        revenue = predicted_yield * price

        # ======================
        # 5️⃣ PROFIT
        # ======================
        water_cost_per_unit = 5
        water_cost = predicted_water * water_cost_per_unit
        total_cost = fertilizer + pesticide + water_cost
        profit = revenue - total_cost

        # ======================
        # 6️⃣ SUSTAINABILITY
        # ======================
        sustainability_score = 100 \
            - (predicted_water * 0.05) \
            - (fertilizer * 0.02) \
            - (pesticide * 0.03)

        sustainability_score = max(0, min(100, sustainability_score))

        # ======================
        # OUTPUT
        # ======================

        st.success("✅ Prediction Completed")

        colA, colB, colC = st.columns(3)

        colA.metric("🌱 Recommended Crop", crop_name)
        colA.metric("🌾 Predicted Yield", round(float(predicted_yield),2))

        colB.metric("💧 Water Requirement", round(float(predicted_water),2))
        colB.metric("💰 Revenue", round(float(revenue),2))

        colC.metric("📊 Profit", round(float(profit),2))
        colC.metric("🌍 Sustainability Score", round(float(sustainability_score),2))

    except Exception as e:
        st.error(f"Error: {e}")