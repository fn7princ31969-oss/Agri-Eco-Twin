from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel,Field


app = FastAPI()

import pandas as pd
import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

aspect_encoder = joblib.load(os.path.join(MODEL_DIR, "aspect_encoder.pkl"))
crop_model = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
yield_model = joblib.load(os.path.join(MODEL_DIR, "yield_model.pkl"))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
market_prices_path = os.path.join(BASE_DIR, "..", "market_price.csv")
market_prices_df = pd.read_csv(market_prices_path)

# Crop name normalization mapping
crop_name_mapping = {
    "SOYBEANS": "Soyabean",
    "SOYBEAN": "Soyabean"
}


# Load market prices
crop_prices = dict(zip(market_prices_df["Crop"], market_prices_df["Base_Price"]))

# Create crop_prices dict with uppercase keys
crop_prices = {crop.upper(): price for crop, price in zip(market_prices_df["Crop"], market_prices_df["Base_Price"])}


# ==========================
# LOAD ALL MODELS
# ==========================

# Crop Model
crop_model = joblib.load("ml_models/crop_model.pkl")
aspect_encoder = joblib.load("backend/ml_models/filename.pkl")
texture_encoder = joblib.load("ml_models/texture_encoder.pkl")
crop_label_encoder = joblib.load("ml_models/crop_label_encoder.pkl")

print("Crop model expects:", crop_model.n_features_in_)

# Yield Model
# yield_model = joblib.load("ml_models/yield_model.pkl")
# yield_crop_encoder = joblib.load("ml_models/yield_crop_encoder.pkl")



# ---- Load Models and Encoders ----
# Water Model
# ---- Water Model ----
water_model = joblib.load("ml_models/water_model.pkl")
water_crop_encoder = joblib.load("ml_models/water_crop_encoder.pkl")
water_soil_encoder = joblib.load("ml_models/water_soil_encoder.pkl")
water_region_encoder = joblib.load("ml_models/water_region_encoder.pkl")
water_weather_encoder = joblib.load("ml_models/water_weather_encoder.pkl")

class WaterRequest(BaseModel):
    crop: str = Field(..., alias="CROP TYPE")
    soil_type: str = Field(..., alias="SOIL TYPE")
    region: str = Field(..., alias="REGION")
    temperature: float = Field(..., alias="TEMPERATURE")
    weather_condition: str = Field(..., alias="WEATHER CONDITION")

    model_config = {
        "validate_by_name": True
    }

def safe_encode(encoder, value, name):
    value = value.upper()
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        # Map to a default index (e.g., 0)
        print(f"Warning: {name} '{value}' was unseen. Using default index 0")
        return 0

# ==========================
# CROP PREDICTION
# ==========================

@app.post("/predict_crop")
def predict_crop(data: dict):

    try:
        print("Incoming data:", data)

        aspect = aspect_encoder.transform([data["aspect"]])[0]
        soil_texture = texture_encoder.transform([data["soil_texture"]])[0]

        features = np.array([[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"]),
            aspect,
            soil_texture
        ]])

        print("Feature shape:", features.shape)
        print("Features:", features)

        prediction = crop_model.predict(features)[0]
        crop_name = crop_label_encoder.inverse_transform([prediction])[0]

        return {"recommended_crop": crop_name}

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}

# ==========================
# YIELD PREDICTION
# ==========================



yield_model = joblib.load("ml_models/yield_model.pkl")
yield_crop_encoder = joblib.load("ml_models/yield_crop_encoder.pkl")

class YieldRequest(BaseModel):
    crop: str
    area: float
    annual_rainfall: float
    fertilizer: float
    pesticide: float

@app.post("/predict_yield")
def predict_yield(data: YieldRequest):
    try:
        crop_encoded = yield_crop_encoder.transform([data.crop])[0]

        features = np.array([[
            crop_encoded,
            data.area,
            data.annual_rainfall,
            data.fertilizer,
            data.pesticide
        ]])

        prediction = yield_model.predict(features)

        return {"predicted_yield": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
    

class RevenueRequest(BaseModel):
    crop: str = Field(..., alias="CROP")
    area: float = Field(..., alias="AREA")
    annual_rainfall: float = Field(..., alias="ANNUAL_RAINFALL")
    fertilizer: float = Field(..., alias="FERTILIZER")
    pesticide: float = Field(..., alias="PESTICIDE")

    model_config = {
        "validate_by_name": True
    }
      
    
@app.post("/predict_revenue")
def predict_revenue(data: RevenueRequest):
    try:
        # Encode crop for yield model
        crop_encoded = yield_crop_encoder.transform([data.crop])[0]

        # Predict yield
        features = np.array([[crop_encoded, data.area, data.annual_rainfall, data.fertilizer, data.pesticide]])
        predicted_yield = yield_model.predict(features)[0]

        # Get market price from CSV (case-insensitive)
        price = crop_prices.get(data.crop.upper(), 0)

        # Calculate revenue
        revenue = predicted_yield * price

        return {
            "predicted_yield": float(predicted_yield),
            "price_per_unit": float(price),
            "predicted_revenue": float(revenue)
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================
# WATER PREDICTION
# ==========================
@app.post("/predict_water")
def predict_water(data: WaterRequest):
    try:
        crop_encoded = safe_encode(water_crop_encoder, data.crop, "Crop")
        soil_encoded = safe_encode(water_soil_encoder, data.soil_type, "Soil type")
        region_encoded = safe_encode(water_region_encoder, data.region, "Region")
        weather_encoded = safe_encode(water_weather_encoder, data.weather_condition, "Weather condition")

        features = np.array([[crop_encoded, soil_encoded, region_encoded, data.temperature, weather_encoded]])
        prediction = water_model.predict(features)[0]

        return {"predicted_water_requirement": round(float(prediction), 2)}

    except Exception as e:
        return {"error": str(e)}
    
# ==========================
# UNIFIED SMART PREDICTION
# ==========================

class FullRequest(BaseModel):
    # Crop inputs
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    aspect: str
    soil_texture: str

    # Yield inputs
    area: float
    fertilizer: float
    pesticide: float

    # Water inputs
    region: str
    weather_condition: str


@app.post("/predict_all")
def predict_all(data: FullRequest):
    try:
        # ======================
        # 1️⃣ CROP PREDICTION
        # ======================
        aspect = aspect_encoder.transform([data.aspect])[0]
        soil_texture = texture_encoder.transform([data.soil_texture])[0]

        crop_features = np.array([[ 
            data.N, data.P, data.K,
            data.temperature, data.humidity,
            data.ph, data.rainfall,
            aspect, soil_texture
        ]])

        crop_prediction = crop_model.predict(crop_features)[0]
        crop_name = crop_label_encoder.inverse_transform([crop_prediction])[0]

        if crop_name.upper() in crop_name_mapping:
            crop_name = crop_name_mapping[crop_name.upper()]

        # ======================
        # 2️⃣ YIELD PREDICTION
        # ======================
        crop_encoded_yield = yield_crop_encoder.transform([crop_name])[0]

        yield_features = np.array([[ 
            crop_encoded_yield,
            data.area,
            data.rainfall,
            data.fertilizer,
            data.pesticide
        ]])

        predicted_yield = yield_model.predict(yield_features)[0]

        # ======================
        # 3️⃣ WATER PREDICTION
        # ======================
        crop_encoded_water = safe_encode(water_crop_encoder, crop_name, "Crop")
        soil_encoded = safe_encode(water_soil_encoder, data.soil_texture, "Soil")
        region_encoded = safe_encode(water_region_encoder, data.region, "Region")
        weather_encoded = safe_encode(water_weather_encoder, data.weather_condition, "Weather")

        water_features = np.array([[ 
            crop_encoded_water,
            soil_encoded,
            region_encoded,
            data.temperature,
            weather_encoded
        ]])

        predicted_water = water_model.predict(water_features)[0]

        # ======================
        # 4️⃣ REVENUE
        # ======================
        price = crop_prices.get(crop_name.upper(), 0)
        revenue = predicted_yield * price

        # ======================
        # 5️⃣ PROFIT CALCULATION
        # ======================
        water_cost_per_unit = 5
        water_cost = predicted_water * water_cost_per_unit
        total_cost = data.fertilizer + data.pesticide + water_cost
        profit = revenue - total_cost

        # ======================
        # 6️⃣ SUSTAINABILITY SCORE
        # ======================
        sustainability_score = 100 \
            - (predicted_water * 0.05) \
            - (data.fertilizer * 0.02) \
            - (data.pesticide * 0.03)

        # Limit score between 0 and 100
        sustainability_score = max(0, min(100, sustainability_score))

        return {
            "recommended_crop": crop_name,
            "predicted_yield": round(float(predicted_yield), 2),
            "predicted_water_requirement": round(float(predicted_water), 2),
            "price_per_unit": float(price),
            "predicted_revenue": round(float(revenue), 2),
            "total_cost": round(float(total_cost), 2),
            "predicted_profit": round(float(profit), 2),
            "sustainability_score": round(float(sustainability_score), 2)
        }

    except Exception as e:
        return {"error": str(e)}