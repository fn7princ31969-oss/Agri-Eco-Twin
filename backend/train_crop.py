import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/Crop recommendation dataset.csv")

# Encode categorical columns
le_aspect = LabelEncoder()
le_texture = LabelEncoder()
le_label = LabelEncoder()

df["aspect"] = le_aspect.fit_transform(df["aspect"])
df["soil_texture"] = le_texture.fit_transform(df["soil_texture"])
df["label"] = le_label.fit_transform(df["label"])

# Features & Target
# Features & Target
X = df[[
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "aspect",
    "soil_texture"
]]

y = df["label"]

print("Total features used:", len(X.columns))
print("Columns:", X.columns)
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Crop Model Accuracy:", accuracy)

# Save model INSIDE ml_models folder
joblib.dump(model, "ml_models/crop_model.pkl")
joblib.dump(le_aspect, "ml_models/aspect_encoder.pkl")
joblib.dump(le_texture, "ml_models/texture_encoder.pkl")
joblib.dump(le_label, "ml_models/crop_label_encoder.pkl")

print("✅ Crop model trained and saved successfully!")