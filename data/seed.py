"""
data/seed.py
------------
Seeds the ClinicalIQ SQLite database with Apollo Health Clinic data.

Run this before ingest.py:
    python data/seed.py

What this script does:
  1. Creates data/clinic_data.db
  2. Drops and recreates all tables (idempotent -- safe to run multiple times)
  3. Inserts sample data for doctors, services, health packages, and price history

Why idempotent?
  If participants need to reset to clean data, they just re-run this script.
  No manual table dropping or file deletion needed.

Why are prices NOT in the documents?
  Consultation fees and service prices live exclusively in this database.
  Putting prices in the markdown documents would create two sources of truth.
  When prices change (re-run seed.py), the documents would still show the old
  figures. The compliance node checks that ClinicalIQ never quotes a price from
  a document -- prices must always come from a tool call to this database.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "clinic_data.db"


def seed() -> None:
    print(f"Seeding ClinicalIQ database at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------
    # doctors
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS doctors")
    conn.execute("""
        CREATE TABLE doctors (
            doctor_id         TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            specialty         TEXT NOT NULL,
            available_days    TEXT NOT NULL,
            consultation_fee  INTEGER NOT NULL
        )
    """)

    doctors = [
        ("dr_001", "Dr. Meera Nair",    "General Medicine", "Mon-Fri",      500),
        ("dr_002", "Dr. Rajesh Kumar",  "Cardiology",       "Mon/Wed/Fri",  800),
        ("dr_003", "Dr. Sunita Sharma", "Paediatrics",      "Tue/Thu/Sat",  600),
        ("dr_004", "Dr. Arun Pillai",   "Orthopaedics",     "Mon/Tue/Thu",  700),
        ("dr_005", "Dr. Priya Menon",   "Dermatology",      "Wed/Fri/Sat",  600),
        ("dr_006", "Dr. Vikram Iyer",   "Ophthalmology",    "Mon/Thu/Sat",  650),
        ("dr_007", "Dr. Anita Bose",    "ENT",              "Tue/Wed/Fri",  600),
        ("dr_008", "Dr. Suresh Rao",    "Pulmonology",      "Mon/Wed/Sat",  750),
    ]
    conn.executemany(
        "INSERT INTO doctors VALUES (?, ?, ?, ?, ?)",
        doctors,
    )
    print(f"  Inserted {len(doctors)} doctors")

    # ------------------------------------------------------------------
    # services
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS services")
    conn.execute("""
        CREATE TABLE services (
            service_id            TEXT PRIMARY KEY,
            name                  TEXT NOT NULL,
            department            TEXT NOT NULL,
            average_duration_mins INTEGER NOT NULL,
            price                 INTEGER NOT NULL
        )
    """)

    services = [
        ("ecg",           "ECG",                       "Cardiology",   20,  350),
        ("blood_panel",   "Basic Blood Panel",         "Diagnostics",  30,  450),
        ("chest_xray",    "Chest X-Ray",               "Radiology",    15,  500),
        ("urine_routine", "Urine Routine",             "Diagnostics",  10,  200),
        ("echo",          "Echocardiogram",            "Cardiology",   45, 1800),
        ("pulmonary_func","Pulmonary Function Test",   "Pulmonology",  30,  900),
        ("eye_test",      "Vision & Pressure Test",    "Ophthalmology",20,  300),
        ("skin_patch",    "Skin Patch Test",           "Dermatology",  40,  600),
        ("bone_density",  "Bone Density Scan",         "Orthopaedics", 25, 1200),
        ("audiometry",    "Audiometry",                "ENT",          20,  500),
    ]
    conn.executemany(
        "INSERT INTO services VALUES (?, ?, ?, ?, ?)",
        services,
    )
    print(f"  Inserted {len(services)} services")

    # ------------------------------------------------------------------
    # health_packages
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS health_packages")
    conn.execute("""
        CREATE TABLE health_packages (
            package_id       TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            tests_included   TEXT NOT NULL,
            price            INTEGER NOT NULL,
            recommended_for  TEXT NOT NULL
        )
    """)

    packages = [
        (
            "pkg_001",
            "Full Body Checkup",
            "CBC/LFT/KFT/Lipids/Thyroid/Sugar/ECG/Chest X-Ray",
            3500,
            "Adults 30+",
        ),
        (
            "pkg_002",
            "Cardiac Screening",
            "ECG/Echo/Lipid Panel/BP Monitoring",
            2800,
            "Adults 40+",
        ),
        (
            "pkg_003",
            "Women's Wellness",
            "CBC/Pap Smear/Bone Density/Thyroid/Sugar",
            2500,
            "Women 25+",
        ),
        (
            "pkg_004",
            "Diabetic Care Package",
            "HbA1c/Fasting Sugar/KFT/LFT/Eye Test/Foot Exam",
            2200,
            "Diabetics",
        ),
    ]
    conn.executemany(
        "INSERT INTO health_packages VALUES (?, ?, ?, ?, ?)",
        packages,
    )
    print(f"  Inserted {len(packages)} health packages")

    # ------------------------------------------------------------------
    # price_history
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS price_history")
    conn.execute("""
        CREATE TABLE price_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            product_type   TEXT NOT NULL,
            product_id     TEXT NOT NULL,
            old_price      INTEGER NOT NULL,
            new_price      INTEGER NOT NULL,
            effective_date TEXT NOT NULL,
            changed_by     TEXT NOT NULL
        )
    """)

    price_history = [
        ("service", "echo",    1500, 1800, "2024-04-01", "admin"),
        ("package", "pkg_001", 3000, 3500, "2024-07-01", "admin"),
    ]
    conn.executemany(
        "INSERT INTO price_history (product_type, product_id, old_price, new_price, effective_date, changed_by) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        price_history,
    )
    print(f"  Inserted {len(price_history)} price history rows")

    conn.commit()
    conn.close()
    print(f"\nDone. Database seeded at {DB_PATH}")
    print("Run 'python data/ingest.py' next to build the vector store.")


if __name__ == "__main__":
    seed()
