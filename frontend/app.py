import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Agri-Eco-Twin", layout="wide")

st.title("🌾 Agri-Eco-Twin Smart Agriculture Dashboard")

# =============================
# LOAD MODELS
# =============================
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

# =============================
# LOAD MARKET PRICE
# =============================
market_prices_df = pd.read_csv("market_price.csv")
crop_prices = {crop.upper(): price for crop, price in zip(market_prices_df["Crop"], market_prices_df["Base_Price"])}

# =============================
# INPUT SECTION
# =============================
st.header("🌱 Soil & Environmental Inputs")

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
    fertilizer = st.number_input("Fertilizer Used")

with col6:
    pesticide = st.number_input("Pesticide Used")

st.header("💧 Water & Climate")

col7, col8 = st.columns(2)

with col7:
    region = st.text_input("Region")

with col8:
    weather_condition = st.text_input("Weather Condition")

# =============================
# PREDICTION BUTTON
# =============================
if st.button("🚀 Run Smart Prediction"):

    try:
        # ---------------------
        # 1️⃣ Crop Prediction
        # ---------------------
        aspect_enc = models["aspect_encoder"].transform([aspect])[0]
        soil_enc = models["texture_encoder"].transform([soil_texture])[0]

        crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall, aspect_enc, soil_enc]])

        crop_pred = models["crop_model"].predict(crop_features)[0]
        crop_name = models["crop_label_encoder"].inverse_transform([crop_pred])[0]

        # ---------------------
        # 2️⃣ Yield Prediction
        # ---------------------
        crop_encoded_yield = models["yield_crop_encoder"].transform([crop_name])[0]

        yield_features = np.array([[crop_encoded_yield, area, rainfall, fertilizer, pesticide]])
        predicted_yield = models["yield_model"].predict(yield_features)[0]

        # ---------------------
        # 3️⃣ Water Prediction
        # ---------------------
        crop_encoded_water = models["water_crop_encoder"].transform([crop_name])[0]
        soil_encoded = models["water_soil_encoder"].transform([soil_texture])[0]
        region_encoded = models["water_region_encoder"].transform([region])[0]
        weather_encoded = models["water_weather_encoder"].transform([weather_condition])[0]

        water_features = np.array([[crop_encoded_water, soil_encoded, region_encoded, temperature, weather_encoded]])
        predicted_water = models["water_model"].predict(water_features)[0]

        # ---------------------
        # 4️⃣ Revenue
        # ---------------------
        price = crop_prices.get(crop_name.upper(), 0)
        revenue = predicted_yield * price

        # ---------------------
        # 5️⃣ Profit
        # ---------------------
        water_cost_per_unit = 5
        water_cost = predicted_water * water_cost_per_unit
        total_cost = fertilizer + pesticide + water_cost
        profit = revenue - total_cost

        # ---------------------
        # 6️⃣ Sustainability
        # ---------------------
        sustainability_score = 100 \
            - (predicted_water * 0.05) \
            - (fertilizer * 0.02) \
            - (pesticide * 0.03)

        sustainability_score = max(0, min(100, sustainability_score))

        # =====================
        # OUTPUT SECTION
        # =====================
        st.success("✅ Prediction Complete")

        colA, colB, colC = st.columns(3)

        colA.metric("🌱 Recommended Crop", crop_name)
        colA.metric("🌾 Predicted Yield", round(float(predicted_yield),2))

        colB.metric("💧 Water Requirement", round(float(predicted_water),2))
        colB.metric("💰 Revenue", round(float(revenue),2))

        colC.metric("📊 Profit", round(float(profit),2))
        colC.metric("🌍 Sustainability Score", round(float(sustainability_score),2))

    except Exception as e:
        st.error(f"Error: {e}")