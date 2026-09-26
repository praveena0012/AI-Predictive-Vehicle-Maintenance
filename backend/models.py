from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey
)

from database import Base


# ==================================================
# BUS TABLE
# ==================================================

class Bus(Base):

    __tablename__ = "buses"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    bus_number = Column(
        String,
        unique=True,
        nullable=False
    )

    registration_number = Column(
        String,
        unique=True,
        nullable=False
    )

    bus_model = Column(
        String,
        nullable=False
    )

    year = Column(
        Integer,
        nullable=False
    )

    vehicle_age = Column(
        Integer,
        nullable=False
    )

    mileage = Column(
        Float,
        nullable=False
    )

    fuel_type = Column(
        String,
        nullable=False
    )

    transmission_type = Column(
        String,
        nullable=False
    )

    status = Column(
        String,
        default="Active"
    )


# ==================================================
# INSPECTION TABLE
# ==================================================

class Inspection(Base):

    __tablename__ = "inspections"

    # ------------------------------
    # BASIC INFORMATION
    # ------------------------------

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    bus_id = Column(
        Integer,
        ForeignKey("buses.id"),
        nullable=False
    )

    inspection_date = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )


    # ------------------------------
    # ML INPUT DATA
    # ------------------------------

    mileage = Column(
        Integer,
        nullable=False
    )

    maintenance_history = Column(
        String,
        nullable=False
    )

    reported_issues = Column(
        Integer,
        nullable=False
    )

    vehicle_age = Column(
        Integer,
        nullable=False
    )

    fuel_type = Column(
        String,
        nullable=False
    )

    transmission_type = Column(
        String,
        nullable=False
    )

    engine_size = Column(
        Integer,
        nullable=False
    )

    odometer_reading = Column(
        Integer,
        nullable=False
    )

    owner_type = Column(
        String,
        nullable=False
    )

    service_history = Column(
        Integer,
        nullable=False
    )

    accident_history = Column(
        Integer,
        nullable=False
    )

    fuel_efficiency = Column(
        Float,
        nullable=False
    )

    tire_condition = Column(
        String,
        nullable=False
    )

    brake_condition = Column(
        String,
        nullable=False
    )

    battery_status = Column(
        String,
        nullable=False
    )


    # ------------------------------
    # AI RESULT
    # ------------------------------

    maintenance_prediction = Column(
        Integer,
        nullable=False
    )

    maintenance_status = Column(
        String,
        nullable=False
    )

    risk_level = Column(
        String,
        nullable=False
    )

    maintenance_probability = Column(
        Float,
        nullable=False
    )

    recommendation = Column(
        String,
        nullable=False
    )

    model_name = Column(
        String,
        nullable=False
    )
    # ==================================================
# MAINTENANCE ALERT TABLE
# ==================================================

class MaintenanceAlert(Base):

    __tablename__ = "maintenance_alert"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    bus_id = Column(
        Integer,
        ForeignKey("buses.id"),
        nullable=False
    )

    inspection_id = Column(
        Integer,
        ForeignKey("inspections.id"),
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    risk_level = Column(
        String,
        nullable=False
    )

    maintenance_probability = Column(
        Float,
        nullable=False
    )

    message = Column(
        String,
        nullable=False
    )

    status = Column(
        String,
        default="Open",
        nullable=False
    )
    # ==================================================
# MAINTENANCE RECORD TABLE
# ==================================================

class MaintenanceRecord(Base):

    __tablename__ = "maintenance_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    bus_id = Column(
        Integer,
        ForeignKey("buses.id"),
        nullable=False
    )

    alert_id = Column(
        Integer,
        ForeignKey("maintenance_alert.id"),
        nullable=True
    )

    maintenance_date = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    maintenance_type = Column(
        String,
        nullable=False
    )

    work_performed = Column(
        String,
        nullable=False
    )

    technician_name = Column(
        String,
        nullable=False
    )

    cost = Column(
        Float,
        nullable=False
    )

    status = Column(
        String,
        default="Completed",
        nullable=False
    )

    notes = Column(
        String,
        nullable=True
    )