from pathlib import Path
import pandas as pd

# File locations
BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "vehicle_maintenance_data.csv"
OUTPUT_FILE = BASE / "bus_maintenance_data.csv"

# Load original dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset size:", df.shape)

# Show vehicle types
print("\nVehicle types:")
print(df["Vehicle_Model"].value_counts())

# Filter only Bus records
bus_df = df[df["Vehicle_Model"] == "Bus"].copy()

print("\nBus dataset size:", bus_df.shape)

# Remove duplicate records
before = len(bus_df)
bus_df = bus_df.drop_duplicates()
after = len(bus_df)

print("Duplicates removed:", before - after)

# Show missing values
print("\nMissing values:")
print(bus_df.isnull().sum())

# Show target distribution
print("\nMaintenance target:")
print(bus_df["Need_Maintenance"].value_counts())

# Save prepared bus dataset
bus_df.to_csv(OUTPUT_FILE, index=False)

print("\nBus dataset saved successfully:")
print(OUTPUT_FILE)