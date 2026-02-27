import joblib

model = joblib.load("ml_models/water_weather_encoder.pkl")
print("Water weather encoder expects:", model.n_features_in_)