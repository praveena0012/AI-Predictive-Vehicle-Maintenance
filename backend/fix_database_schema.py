import sqlite3
import sys

DB_FILE = "vehicle_maintenance.db"

conn = sqlite3.connect(DB_FILE)

try:
    cursor = conn.cursor()

    # Safety check: make sure the required tables exist
    tables = {
        row[0]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    required_tables = {
        "buses",
        "maintenance_alert",
        "maintenance_records",
    }

    missing = required_tables - tables

    if missing:
        print("ERROR: Missing required tables:", missing)
        sys.exit(1)

    # Check that every maintenance record's alert_id exists
    # in the CURRENT maintenance_alert table.
    invalid_records = cursor.execute("""
        SELECT mr.id, mr.alert_id
        FROM maintenance_records mr
        LEFT JOIN maintenance_alert ma
            ON mr.alert_id = ma.id
        WHERE mr.alert_id IS NOT NULL
          AND ma.id IS NULL
    """).fetchall()

    if invalid_records:
        print("ERROR: Some maintenance records reference alerts")
        print("that do not exist in maintenance_alert:")
        print(invalid_records)
        print("Database was NOT changed.")
        sys.exit(1)

    print("Existing maintenance records are compatible.")
    print("Starting schema repair...")

    # Foreign keys must be disabled before rebuilding the table.
    conn.commit()
    cursor.execute("PRAGMA foreign_keys = OFF")

    cursor.execute("BEGIN TRANSACTION")

    # Create corrected table
    cursor.execute("""
        CREATE TABLE maintenance_records_new (
            id INTEGER NOT NULL,
            bus_id INTEGER NOT NULL,
            alert_id INTEGER,
            maintenance_date DATETIME NOT NULL,
            maintenance_type VARCHAR NOT NULL,
            work_performed VARCHAR NOT NULL,
            technician_name VARCHAR NOT NULL,
            cost FLOAT NOT NULL,
            status VARCHAR NOT NULL,
            notes VARCHAR,
            PRIMARY KEY (id),
            FOREIGN KEY(bus_id) REFERENCES buses (id),
            FOREIGN KEY(alert_id) REFERENCES maintenance_alert (id)
        )
    """)

    # Copy existing records
    cursor.execute("""
        INSERT INTO maintenance_records_new (
            id,
            bus_id,
            alert_id,
            maintenance_date,
            maintenance_type,
            work_performed,
            technician_name,
            cost,
            status,
            notes
        )
        SELECT
            id,
            bus_id,
            alert_id,
            maintenance_date,
            maintenance_type,
            work_performed,
            technician_name,
            cost,
            status,
            notes
        FROM maintenance_records
    """)

    # Replace old table
    cursor.execute("DROP TABLE maintenance_records")
    cursor.execute(
        "ALTER TABLE maintenance_records_new RENAME TO maintenance_records"
    )

    conn.commit()

    # Turn FK checking back on
    cursor.execute("PRAGMA foreign_keys = ON")

    fk_errors = cursor.execute("PRAGMA foreign_key_check").fetchall()

    print("\nNew maintenance_records foreign keys:")
    print(cursor.execute(
        "PRAGMA foreign_key_list(maintenance_records)"
    ).fetchall())

    if fk_errors:
        print("\nWARNING: Foreign-key problems found:")
        print(fk_errors)
    else:
        print("\nSUCCESS: No foreign-key problems found.")
        print("Database schema repair completed.")

except Exception as e:
    conn.rollback()
    print("ERROR:", e)

finally:
    conn.close()