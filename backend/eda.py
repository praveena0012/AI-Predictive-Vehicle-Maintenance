from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / "bus_maintenance_data.csv"
OUTPUT = BASE / "eda_results"

# Create folder for graphs
OUTPUT.mkdir(exist_ok=True)

# Load bus dataset
df = pd.read_csv(DATA)

print("Dataset size:", df.shape)

print("\nDataset information:")
print(df.dtypes)

print("\nNumeric summary:")
print(df.describe())

print("\nMaintenance distribution:")
print(df["Need_Maintenance"].value_counts())

print("\nMaintenance percentage:")
print(df["Need_Maintenance"].value_counts(normalize=True) * 100)


# --------------------------------
# Graph 1 - Maintenance Distribution
# --------------------------------

df["Need_Maintenance"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Bus Maintenance Requirement Distribution")
plt.xlabel("Need Maintenance (0 = No, 1 = Yes)")
plt.ylabel("Number of Buses")
plt.tight_layout()

plt.savefig(OUTPUT / "maintenance_distribution.png")
plt.close()


# --------------------------------
# Graph 2 - Vehicle Age
# --------------------------------

df["Vehicle_Age"].plot(
    kind="hist",
    bins=15
)

plt.title("Distribution of Bus Vehicle Age")
plt.xlabel("Vehicle Age")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(OUTPUT / "vehicle_age_distribution.png")
plt.close()


# --------------------------------
# Graph 3 - Mileage
# --------------------------------

df["Mileage"].plot(
    kind="hist",
    bins=20
)

plt.title("Distribution of Bus Mileage")
plt.xlabel("Mileage")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(OUTPUT / "mileage_distribution.png")
plt.close()


# --------------------------------
# Graph 4 - Brake Condition
# --------------------------------

pd.crosstab(
    df["Brake_Condition"],
    df["Need_Maintenance"]
).plot(kind="bar")

plt.title("Brake Condition vs Maintenance Requirement")
plt.xlabel("Brake Condition")
plt.ylabel("Number of Buses")
plt.legend(["No Maintenance", "Need Maintenance"])
plt.tight_layout()

plt.savefig(OUTPUT / "brake_condition.png")
plt.close()


# --------------------------------
# Graph 5 - Tire Condition
# --------------------------------

pd.crosstab(
    df["Tire_Condition"],
    df["Need_Maintenance"]
).plot(kind="bar")

plt.title("Tire Condition vs Maintenance Requirement")
plt.xlabel("Tire Condition")
plt.ylabel("Number of Buses")
plt.legend(["No Maintenance", "Need Maintenance"])
plt.tight_layout()

plt.savefig(OUTPUT / "tire_condition.png")
plt.close()


# --------------------------------
# Graph 6 - Battery Status
# --------------------------------

pd.crosstab(
    df["Battery_Status"],
    df["Need_Maintenance"]
).plot(kind="bar")

plt.title("Battery Status vs Maintenance Requirement")
plt.xlabel("Battery Status")
plt.ylabel("Number of Buses")
plt.legend(["No Maintenance", "Need Maintenance"])
plt.tight_layout()

plt.savefig(OUTPUT / "battery_status.png")
plt.close()


print("\nEDA completed successfully.")
print("Graphs saved in:")
print(OUTPUT)