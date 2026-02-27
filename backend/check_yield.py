import joblib

model = joblib.load("ml_models/yield_model.pkl")
print("Yield model expects:", model.n_features_in_)