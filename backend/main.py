from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import engine, get_db
from models import Base, Bus, Inspection, MaintenanceAlert,MaintenanceRecord


# --------------------------------------------------
# 1. Database Setup
# --------------------------------------------------

Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# 2. Application Setup
# --------------------------------------------------

app = FastAPI(
    title="BusPredict API",
    description="AI-Based Predictive Maintenance API for Public Buses",
    version="2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# 3. Load Final Random Forest Model
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "final_bus_model.joblib"

package = joblib.load(MODEL_FILE)

model = package["model"]
features = package["features"]

print("Final BusPredict model loaded successfully.")
print("Algorithm:", package["model_name"])
print("Expected features:", len(features))


# --------------------------------------------------
# 4. Smart Inspection Input Schema
# --------------------------------------------------

class BusInspection(BaseModel):

    Mileage: int = Field(ge=0)
    Maintenance_History: str
    Reported_Issues: int = Field(ge=0)
    Vehicle_Age: int = Field(ge=0, le=100)

    Fuel_Type: str
    Transmission_Type: str
    Engine_Size: int = Field(gt=0)
    Odometer_Reading: int = Field(ge=0)

    Owner_Type: str
    Service_History: int = Field(ge=0)
    Accident_History: int = Field(ge=0)
    Fuel_Efficiency: float = Field(gt=0)

    Tire_Condition: str
    Brake_Condition: str
    Battery_Status: str


# --------------------------------------------------
# 5. Bus Input Schema
# --------------------------------------------------

class BusCreate(BaseModel):

    bus_number: str = Field(min_length=1)
    registration_number: str = Field(min_length=1)
    bus_model: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    vehicle_age: int = Field(ge=0, le=100)
    mileage: float = Field(ge=0)
    fuel_type: str = Field(min_length=1)
    transmission_type: str = Field(min_length=1)
    status: str = "Active"


# --------------------------------------------------
# 6. Inspection Save Schema
# --------------------------------------------------

class InspectionCreate(BusInspection):

    bus_id: int
# --------------------------------------------------
# 7. Maintenance Record Input Schema
# --------------------------------------------------

class MaintenanceRecordCreate(BaseModel):

    bus_id: int = Field(gt=0)
    alert_id: int | None = Field(default=None, gt=0)

    maintenance_type: str = Field(min_length=1)
    work_performed: str = Field(min_length=1)
    technician_name: str = Field(min_length=1)
    cost: float = Field(ge=0)

    status: str = "Completed"
    notes: str | None = None

# --------------------------------------------------
# 7. Health Endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": package["model_name"],
        "training_records": package["dataset_records"],
        "features": len(features)
    }


# --------------------------------------------------
# 8. Create Bus
# --------------------------------------------------

@app.post("/buses")
def create_bus(
    bus: BusCreate,
    db: Session = Depends(get_db)
):

    existing_bus = db.query(Bus).filter(
        Bus.bus_number == bus.bus_number
    ).first()

    if existing_bus:

        raise HTTPException(
            status_code=400,
            detail="Bus number already exists"
        )


    existing_registration = db.query(Bus).filter(
        Bus.registration_number ==
        bus.registration_number
    ).first()

    if existing_registration:

        raise HTTPException(
            status_code=400,
            detail="Registration number already exists"
        )


    new_bus = Bus(

        bus_number=bus.bus_number,

        registration_number=
            bus.registration_number,

        bus_model=bus.bus_model,

        year=bus.year,

        vehicle_age=bus.vehicle_age,

        mileage=bus.mileage,

        fuel_type=bus.fuel_type,

        transmission_type=
            bus.transmission_type,

        status=bus.status
    )


    db.add(new_bus)

    db.commit()

    db.refresh(new_bus)


    return new_bus


# --------------------------------------------------
# 9. Get All Buses
# --------------------------------------------------

@app.get("/buses")
def get_buses(
    db: Session = Depends(get_db)
):

    buses = db.query(Bus).all()

    return buses


# --------------------------------------------------
# 10. Delete Bus
# --------------------------------------------------

@app.delete("/buses/{bus_id}")
def delete_bus(
    bus_id: int,
    db: Session = Depends(get_db)
):

    bus = db.query(Bus).filter(
        Bus.id == bus_id
    ).first()


    if not bus:

        raise HTTPException(
            status_code=404,
            detail="Bus not found"
        )


    # Protect historical inspection / maintenance data.
    # A bus can only be deleted when it has no related history.
    has_inspections = db.query(Inspection).filter(
        Inspection.bus_id == bus_id
    ).first()

    has_alerts = db.query(MaintenanceAlert).filter(
        MaintenanceAlert.bus_id == bus_id
    ).first()

    has_maintenance_records = db.query(MaintenanceRecord).filter(
        MaintenanceRecord.bus_id == bus_id
    ).first()

    if has_inspections or has_alerts or has_maintenance_records:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot delete this bus because it has "
                "inspection, maintenance alert, or maintenance record history."
            )
        )

    db.delete(bus)
    db.commit()

    return {
        "message": "Bus deleted successfully"
    }


# --------------------------------------------------
# 11. AI Prediction Helper
# --------------------------------------------------

def run_prediction(data):

    # Convert input to dictionary
    input_dict = {

        "Mileage":
            data.Mileage,

        "Maintenance_History":
            data.Maintenance_History,

        "Reported_Issues":
            data.Reported_Issues,

        "Vehicle_Age":
            data.Vehicle_Age,

        "Fuel_Type":
            data.Fuel_Type,

        "Transmission_Type":
            data.Transmission_Type,

        "Engine_Size":
            data.Engine_Size,

        "Odometer_Reading":
            data.Odometer_Reading,

        "Owner_Type":
            data.Owner_Type,

        "Service_History":
            data.Service_History,

        "Accident_History":
            data.Accident_History,

        "Fuel_Efficiency":
            data.Fuel_Efficiency,

        "Tire_Condition":
            data.Tire_Condition,

        "Brake_Condition":
            data.Brake_Condition,

        "Battery_Status":
            data.Battery_Status
    }


    # Create DataFrame in exact training order
    input_data = pd.DataFrame(
        [input_dict],
        columns=features
    )


    # Run Random Forest
    prediction = int(
        model.predict(input_data)[0]
    )


    probabilities = model.predict_proba(
        input_data
    )[0]


    class_positions = {

        int(label): index

        for index, label
        in enumerate(model.classes_)
    }


    maintenance_probability = float(
        probabilities[
            class_positions[1]
        ]
    )


    maintenance_percentage = round(
        maintenance_probability * 100,
        2
    )


    # --------------------------------------------------
    # Decision-Support Logic
    # --------------------------------------------------

    if prediction == 0:

        maintenance_status = (
            "No Maintenance Required"
        )

        risk_level = "Low"

        recommendation = (
            "Continue normal operation and "
            "scheduled maintenance inspections."
        )


    else:

        maintenance_status = (
            "Maintenance Required"
        )


        if maintenance_percentage >= 80:

            risk_level = "High"

            recommendation = (
                "Arrange a technician inspection "
                "as soon as possible and review "
                "reported vehicle issues."
            )


        else:

            risk_level = "Medium"

            recommendation = (
                "Schedule a maintenance inspection "
                "and monitor the vehicle condition."
            )


    return {

        "maintenance_prediction":
            prediction,

        "maintenance_status":
            maintenance_status,

        "risk_level":
            risk_level,

        "maintenance_probability":
            maintenance_percentage,

        "recommendation":
            recommendation,

        "model":
            package["model_name"]
    }


# --------------------------------------------------
# 12. Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(
    data: BusInspection
):

    result = run_prediction(data)


    return {

        **result,

        "disclaimer": (
            "Decision support only. "
            "The prediction should be "
            "verified by a trained "
            "vehicle technician."
        )
    }


# --------------------------------------------------
# 13. Run Inspection + Save to Database
# --------------------------------------------------

@app.post("/inspections")
def create_inspection(
    data: InspectionCreate,
    db: Session = Depends(get_db)
):

    # --------------------------------------------------
    # Check Bus
    # --------------------------------------------------

    bus = db.query(Bus).filter(
        Bus.id == data.bus_id
    ).first()


    if not bus:

        raise HTTPException(
            status_code=404,
            detail="Bus not found"
        )


    # --------------------------------------------------
    # Run AI Model
    # --------------------------------------------------

    result = run_prediction(data)


    # --------------------------------------------------
    # Create Inspection Record
    # --------------------------------------------------

    inspection = Inspection(

        bus_id=data.bus_id,

        mileage=data.Mileage,

        maintenance_history=
            data.Maintenance_History,

        reported_issues=
            data.Reported_Issues,

        vehicle_age=
            data.Vehicle_Age,

        fuel_type=
            data.Fuel_Type,

        transmission_type=
            data.Transmission_Type,

        engine_size=
            data.Engine_Size,

        odometer_reading=
            data.Odometer_Reading,

        owner_type=
            data.Owner_Type,

        service_history=
            data.Service_History,

        accident_history=
            data.Accident_History,

        fuel_efficiency=
            data.Fuel_Efficiency,

        tire_condition=
            data.Tire_Condition,

        brake_condition=
            data.Brake_Condition,

        battery_status=
            data.Battery_Status,

        maintenance_prediction=
            result[
                "maintenance_prediction"
            ],

        maintenance_status=
            result[
                "maintenance_status"
            ],

        risk_level=
            result[
                "risk_level"
            ],

        maintenance_probability=
            result[
                "maintenance_probability"
            ],

        recommendation=
            result[
                "recommendation"
            ],

        model_name=
            result[
                "model"
            ]
    )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    db.add(inspection)

    db.commit()

    db.refresh(inspection)
    # --------------------------------------------------
# Create Maintenance Alert for Medium / High Risk
# --------------------------------------------------
    # --------------------------------------------------
    # Create Maintenance Alert for Medium / High Risk
    # --------------------------------------------------

    if result["risk_level"] in ["Medium", "High"]:

        if result["risk_level"] == "High":
            alert_message = (
                "High maintenance risk detected. "
                "Immediate technician inspection is recommended."
            )
        else:
            alert_message = (
                "Maintenance attention is recommended. "
                "Schedule a technician inspection."
            )

        alert = MaintenanceAlert(
            bus_id=bus.id,
            inspection_id=inspection.id,
            risk_level=result["risk_level"],
            maintenance_probability=result["maintenance_probability"],
            message=alert_message,
            status="Open"
        )

        db.add(alert)
        db.commit()
        db.refresh(alert)


    # --------------------------------------------------
    # Return Saved Result
    # --------------------------------------------------

    return {
        "inspection_id": inspection.id,
        "bus_id": bus.id,
        "bus_number": bus.bus_number,
        "registration_number": bus.registration_number,
        "inspection_date": inspection.inspection_date,
        "maintenance_prediction": inspection.maintenance_prediction,
        "maintenance_status": inspection.maintenance_status,
        "risk_level": inspection.risk_level,
        "maintenance_probability": inspection.maintenance_probability,
        "recommendation": inspection.recommendation,
        "model": inspection.model_name,
        "message": "Inspection saved successfully",
        "disclaimer": (
            "Decision support only. "
            "The prediction should be verified by a trained "
            "vehicle technician."
        )
    }


# --------------------------------------------------
# 14. Get Inspection History
# --------------------------------------------------

@app.get("/inspections")
def get_inspections(
    db: Session = Depends(get_db)
):

    inspections = (

        db.query(
            Inspection,
            Bus
        )

        .join(
            Bus,
            Inspection.bus_id ==
            Bus.id
        )

        .order_by(
            Inspection.inspection_date.desc()
        )

        .all()
    )


    history = []


    for inspection, bus in inspections:

        history.append({

            "inspection_id":
                inspection.id,

            "bus_id":
                bus.id,

            "bus_number":
                bus.bus_number,

            "registration_number":
                bus.registration_number,

            "bus_model":
                bus.bus_model,

            "inspection_date":
                inspection.inspection_date,

            "maintenance_status":
                inspection.maintenance_status,

            "risk_level":
                inspection.risk_level,

            "maintenance_probability":
                inspection.maintenance_probability,

            "recommendation":
                inspection.recommendation,

            "model":
                inspection.model_name

        })


    return history
# --------------------------------------------------
# 15. Get Maintenance Alerts
# --------------------------------------------------

@app.get("/alerts")
def get_alerts(
    db: Session = Depends(get_db)
):

    alerts = (

        db.query(
            MaintenanceAlert,
            Bus
        )

        .join(
            Bus,
            MaintenanceAlert.bus_id == Bus.id
        )

        .order_by(
            MaintenanceAlert.created_at.desc()
        )

        .all()
    )


    result = []


    for alert, bus in alerts:

        result.append({

            "alert_id":
                alert.id,

            "bus_id":
                bus.id,

            "bus_number":
                bus.bus_number,

            "registration_number":
                bus.registration_number,

            "inspection_id":
                alert.inspection_id,

            "created_at":
                alert.created_at,

            "risk_level":
                alert.risk_level,

            "maintenance_probability":
                alert.maintenance_probability,

            "message":
                alert.message,

            "status":
                alert.status

        })


    return result
# --------------------------------------------------
# 16. Resolve Maintenance Alert
# --------------------------------------------------

@app.patch("/alerts/{alert_id}/resolve")
def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db)
):

    alert = db.query(MaintenanceAlert).filter(
        MaintenanceAlert.id == alert_id
    ).first()

    if not alert:
        raise HTTPException(
            status_code=404,
            detail="Maintenance alert not found"
        )

    if alert.status == "Resolved":
        return {
            "message": "Alert is already resolved",
            "alert_id": alert.id,
            "status": alert.status
        }

    alert.status = "Resolved"

    db.commit()
    db.refresh(alert)

    return {
        "message": "Maintenance alert resolved successfully",
        "alert_id": alert.id,
        "status": alert.status
    }
# --------------------------------------------------
# 17. Create Maintenance Record
# --------------------------------------------------

@app.post("/maintenance-records")
def create_maintenance_record(
    data: MaintenanceRecordCreate,
    db: Session = Depends(get_db)
):

    # --------------------------------------------------
    # Check Bus
    # --------------------------------------------------

    bus = db.query(Bus).filter(
        Bus.id == data.bus_id
    ).first()

    if not bus:
        raise HTTPException(
            status_code=404,
            detail="Bus not found"
        )


    # --------------------------------------------------
    # Check Alert (if provided)
    # --------------------------------------------------

    alert = None

    if data.alert_id is not None:

        alert = db.query(MaintenanceAlert).filter(
            MaintenanceAlert.id == data.alert_id
        ).first()

        if not alert:
            raise HTTPException(
                status_code=404,
                detail="Maintenance alert not found"
            )

        # Make sure alert belongs to selected bus
        if alert.bus_id != data.bus_id:
            raise HTTPException(
                status_code=400,
                detail="The selected alert does not belong to this bus"
            )


    # --------------------------------------------------
    # Create Maintenance Record
    # --------------------------------------------------

    record = MaintenanceRecord(
        bus_id=data.bus_id,
        alert_id=data.alert_id,
        maintenance_type=data.maintenance_type,
        work_performed=data.work_performed,
        technician_name=data.technician_name,
        cost=data.cost,
        status=data.status,
        notes=data.notes
    )

    db.add(record)
    db.commit()
    db.refresh(record)


    # --------------------------------------------------
    # Resolve Related Alert
    # --------------------------------------------------

    if alert and data.status == "Completed":

        alert.status = "Resolved"

        db.commit()
        db.refresh(alert)


    # --------------------------------------------------
    # Return Result
    # --------------------------------------------------

    return {
        "maintenance_record_id": record.id,
        "bus_id": bus.id,
        "bus_number": bus.bus_number,
        "registration_number": bus.registration_number,
        "alert_id": record.alert_id,
        "maintenance_date": record.maintenance_date,
        "maintenance_type": record.maintenance_type,
        "work_performed": record.work_performed,
        "technician_name": record.technician_name,
        "cost": record.cost,
        "status": record.status,
        "notes": record.notes,
        "message": "Maintenance record saved successfully"
    }
# --------------------------------------------------
# 18. Get Maintenance Records
# --------------------------------------------------

@app.get("/maintenance-records")
def get_maintenance_records(
    db: Session = Depends(get_db)
):

    records = (
        db.query(
            MaintenanceRecord,
            Bus
        )
        .join(
            Bus,
            MaintenanceRecord.bus_id == Bus.id
        )
        .order_by(
            MaintenanceRecord.maintenance_date.desc()
        )
        .all()
    )

    result = []

    for record, bus in records:

        result.append({
            "maintenance_record_id": record.id,
            "bus_id": bus.id,
            "bus_number": bus.bus_number,
            "registration_number": bus.registration_number,
            "alert_id": record.alert_id,
            "maintenance_date": record.maintenance_date,
            "maintenance_type": record.maintenance_type,
            "work_performed": record.work_performed,
            "technician_name": record.technician_name,
            "cost": record.cost,
            "status": record.status,
            "notes": record.notes
        })

    return result
