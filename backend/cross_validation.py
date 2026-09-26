from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_validate


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
# 3. Feature Types
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
# 5. Models
# -----------------------------------

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
}


# -----------------------------------
# 6. 5-Fold Stratified Cross Validation
# -----------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

results = []


# -----------------------------------
# 7. Evaluate Each Model
# -----------------------------------

for model_name, classifier in models.items():

    print("\n====================================")
    print(model_name)
    print("====================================")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    accuracy = scores["test_accuracy"].mean()
    precision = scores["test_precision"].mean()
    recall = scores["test_recall"].mean()
    f1 = scores["test_f1"].mean()

    print(f"Mean Accuracy : {accuracy:.4f}")
    print(f"Mean Precision: {precision:.4f}")
    print(f"Mean Recall   : {recall:.4f}")
    print(f"Mean F1 Score : {f1:.4f}")

    print("\nAccuracy for each fold:")
    print(scores["test_accuracy"])

    results.append({
        "Model": model_name,
        "Mean Accuracy": accuracy,
        "Mean Precision": precision,
        "Mean Recall": recall,
        "Mean F1 Score": f1
    })


# -----------------------------------
# 8. Final Comparison
# -----------------------------------

results_df = pd.DataFrame(results)

print("\n\nCROSS-VALIDATION COMPARISON")
print("====================================")

print(results_df.to_string(index=False))


# Save results
results_df.to_csv(
    BASE / "cross_validation_results.csv",
    index=False
)

print("\nResults saved:")
print(BASE / "cross_validation_results.csv")