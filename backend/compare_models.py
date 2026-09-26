from sklearn.inspection import permutation_importance
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -----------------------------------
# 1. Load Bus Dataset
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
# 3. Numeric and Categorical Features
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

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------
# 6. Models
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
# 7. Train and Evaluate
# -----------------------------------

results = []

for model_name, classifier in models.items():

    print("\n====================================")
    print(model_name)
    print("====================================")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# -----------------------------------
# 8. Comparison
# -----------------------------------

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("====================================")
print(results_df.to_string(index=False))