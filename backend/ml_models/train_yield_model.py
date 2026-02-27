import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("../data/crop_yield.csv")

# Encode Crop column
le_crop = LabelEncoder()
df["Crop"] = le_crop.fit_transform(df["Crop"])

# Select important features
X = df[["Crop", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]]
y = df["Yield"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Yield Model R2 Score:", score)

# Save model
joblib.dump(model, "yield_model.pkl")
joblib.dump(le_crop, "yield_crop_encoder.pkl")

print("Yield model trained and saved successfully!")