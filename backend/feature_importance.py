from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

BASE = Path(__file__).resolve().parent
DATA = BASE / "bus_maintenance_data.csv"

df = pd.read_csv(DATA)

print("Dataset size:", df.shape)


# -----------------------------------
# 2. Select Features
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
# 3. Separate Feature Types
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
# 4. Preprocessing
# -----------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# -----------------------------------
# 5. Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 6. Random Forest
# -----------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# -----------------------------------
# 7. Get Feature Names
# -----------------------------------

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model.named_steps[
    "classifier"
].feature_importances_


# -----------------------------------
# 8. Create Importance Table
# -----------------------------------

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print("========================================")

print(
    importance_df.head(15).to_string(index=False)
)


# -----------------------------------
# 9. Save Results
# -----------------------------------

importance_df.to_csv(
    BASE / "feature_importance_results.csv",
    index=False
)


# -----------------------------------
# 10. Create Graph
# -----------------------------------

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    BASE / "feature_importance.png",
    dpi=300
)

plt.close()


print("\nResults saved:")
print(BASE / "feature_importance_results.csv")

print("\nGraph saved:")
print(BASE / "feature_importance.png")
