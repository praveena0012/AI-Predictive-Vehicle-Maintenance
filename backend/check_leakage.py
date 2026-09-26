from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "bus_maintenance_data.csv"

df = pd.read_csv(DATA)

print("TARGET DISTRIBUTION")
print("===================")
print(df["Need_Maintenance"].value_counts())


columns = [
    "Maintenance_History",
    "Reported_Issues",
    "Service_History",
    "Accident_History",
    "Tire_Condition",
    "Brake_Condition",
    "Battery_Status"
]


for column in columns:

    print("\n\n====================================")
    print(column, "vs Need_Maintenance")
    print("====================================")

    table = pd.crosstab(
        df[column],
        df["Need_Maintenance"],
        margins=True
    )

    print(table)


print("\n\nCORRELATION WITH TARGET")
print("=======================")

numeric_columns = df.select_dtypes(include="number")

correlations = (
    numeric_columns
    .corr()["Need_Maintenance"]
    .sort_values(ascending=False)
)

print(correlations)