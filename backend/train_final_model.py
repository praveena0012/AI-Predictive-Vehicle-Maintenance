from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------
# 1. File Paths
# -----------------------------------

BASE = Path(__file__).resolve().parent

DATA = BASE / "bus_maintenance_data.csv"

# Save separately first.
# Do NOT overwrite the old demo model yet.
MODEL_FILE = BASE / "final_bus_model.joblib"


# -----------------------------------
# 2. Load Prepared Bus Dataset
# -----------------------------------

df = pd.read_csv(DATA)

print("Dataset loaded successfully.")
print("Dataset size:", df.shape)


# -----------------------------------
# 3. Select Features and Target
# -----------------------------------

features = [
    "Mileage",
    "Maintenance_History",
    "Reported_Issues",
    "Vehicle_Age",
    "Fuel_Type",
    "Transmission_Type",
    "Engine_Size",
    "Odometer_Reading",
    "Owner_Type",
    "Service_History",
    "Accident_History",
    "Fuel_Efficiency",
    "Tire_Condition",
    "Brake_Condition",
    "Battery_Status"
]

target = "Need_Maintenance"

X = df[features].copy()
y = df[target]


# -----------------------------------
# 4. Numeric Features
# -----------------------------------

numeric_features = [
    "Mileage",
    "Reported_Issues",
    "Vehicle_Age",
    "Engine_Size",
    "Odometer_Reading",
    "Service_History",
    "Accident_History",
    "Fuel_Efficiency"
]


# -----------------------------------
# 5. Categorical Features
# -----------------------------------

categorical_features = [
    "Maintenance_History",
    "Fuel_Type",
    "Transmission_Type",
    "Owner_Type",
    "Tire_Condition",
    "Brake_Condition",
    "Battery_Status"
]


# -----------------------------------
# 6. Numeric Preprocessing
# -----------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# -----------------------------------
# 7. Categorical Preprocessing
# -----------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# -----------------------------------
# 8. Combine Preprocessing
# -----------------------------------

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# -----------------------------------
# 9. Final Random Forest
# -----------------------------------

final_model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])


# -----------------------------------
# 10. Train on Prepared Dataset
# -----------------------------------

print("\nTraining final Random Forest model...")

final_model.fit(X, y)

print("Training completed successfully.")


# -----------------------------------
# 11. Save Model
# -----------------------------------

model_package = {
    "model": final_model,
    "features": features,
    "target": target,
    "model_name": "Random Forest",
    "dataset_records": len(df)
}

joblib.dump(
    model_package,
    MODEL_FILE
)


print("\nFinal model saved successfully:")
print(MODEL_FILE)

print("\nModel information:")
print("Algorithm: Random Forest")
print("Training records:", len(df))
print("Number of input features:", len(features))
print("Target:", target)