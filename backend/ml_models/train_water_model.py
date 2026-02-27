import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("../data/DATASET - Sheet1.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert temperature range to average number
def convert_temp(temp_range):
    parts = temp_range.split("-")
    return (int(parts[0]) + int(parts[1])) / 2

df["TEMPERATURE"] = df["TEMPERATURE"].apply(convert_temp)

# Encode categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_region = LabelEncoder()
le_weather = LabelEncoder()

df["CROP TYPE"] = le_crop.fit_transform(df["CROP TYPE"])
df["SOIL TYPE"] = le_soil.fit_transform(df["SOIL TYPE"])
df["REGION"] = le_region.fit_transform(df["REGION"])
df["WEATHER CONDITION"] = le_weather.fit_transform(df["WEATHER CONDITION"])

# Features & Target
X = df[["CROP TYPE", "SOIL TYPE", "REGION", "TEMPERATURE", "WEATHER CONDITION"]]
y = df["WATER REQUIREMENT"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeRegressor(max_depth=10)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Water Model R2 Score:", score)

# Save model & encoders
joblib.dump(model, "water_model.pkl")
joblib.dump(le_crop, "water_crop_encoder.pkl")
joblib.dump(le_soil, "water_soil_encoder.pkl")
joblib.dump(le_region, "water_region_encoder.pkl")
joblib.dump(le_weather, "water_weather_encoder.pkl")

print("Water model trained successfully!")