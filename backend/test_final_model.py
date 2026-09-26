from pathlib import Path
import joblib
import pandas as pd

# -----------------------------------
# 1. Load Final Model
# -----------------------------------

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "final_bus_model.joblib"

package = joblib.load(MODEL_FILE)

model = package["model"]
features = package["features"]

print("Model loaded successfully.")
print("Model:", package["model_name"])
print("Expected features:", len(features))


# -----------------------------------
# 2. Test Bus Record
# -----------------------------------

test_bus = {
    "Mileage": 70000,
    "Maintenance_History": "Poor",
    "Reported_Issues": 4,
    "Vehicle_Age": 8,
    "Fuel_Type": "Diesel",
    "Transmission_Type": "Manual",
    "Engine_Size": 2000,
    "Odometer_Reading": 120000,
    "Owner_Type": "First",
    "Service_History": 8,
    "Accident_History": 2,
    "Fuel_Efficiency": 12.5,
    "Tire_Condition": "Worn Out",
    "Brake_Condition": "Worn Out",
    "Battery_Status": "Weak"
}

input_data = pd.DataFrame(
    [test_bus],
    columns=features
)


# -----------------------------------
# 3. Make Prediction
# -----------------------------------

prediction = model.predict(input_data)[0]

print("\nPrediction result:", prediction)

if prediction == 1:
    print("Result: Maintenance Required")
else:
    print("Result: No Maintenance Required")


# -----------------------------------
# 4. Prediction Probability
# -----------------------------------

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(input_data)[0]

    print("\nPrediction probabilities:")

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):
        print(
            f"Class {class_name}: "
            f"{probability * 100:.2f}%"
        )