"""
s07/tests/conftest.py
---------------------
Pytest configuration and fixtures for Session 7 tests.

Creates a minimal in-memory SQLite database that mirrors the schema of
clinic_data.db so tests run without requiring data/seed.py to have been run.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

# Make s07/solution importable
SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))


@pytest.fixture()
def test_db(tmp_path):
    """Create a minimal clinic_data.db in a temp directory with known seed data."""
    db_path = tmp_path / "clinic_data.db"
    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        CREATE TABLE doctors (
            doctor_id         TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            specialty         TEXT NOT NULL,
            available_days    TEXT NOT NULL,
            consultation_fee  INTEGER NOT NULL
        );

        CREATE TABLE services (
            service_id            TEXT PRIMARY KEY,
            name                  TEXT NOT NULL,
            department            TEXT NOT NULL,
            average_duration_mins INTEGER NOT NULL,
            price                 INTEGER NOT NULL
        );

        INSERT INTO doctors VALUES
            ('dr_001', 'Dr. Meera Nair',    'General Medicine', 'Mon-Fri',      500),
            ('dr_002', 'Dr. Rajesh Kumar',  'Cardiology',       'Mon/Wed/Fri',  800),
            ('dr_003', 'Dr. Sunita Sharma', 'Paediatrics',      'Tue/Thu/Sat',  600),
            ('dr_004', 'Dr. Arun Pillai',   'Orthopaedics',     'Mon/Tue/Thu',  700);

        INSERT INTO services VALUES
            ('ecg',           'ECG',                 'Cardiology',  20, 350),
            ('echo',          'Echocardiogram',       'Cardiology',  45, 1800),
            ('blood_panel',   'Basic Blood Panel',    'Diagnostics', 30, 450),
            ('chest_xray',    'Chest X-Ray',          'Radiology',   15, 500),
            ('urine_routine', 'Urine Routine',        'Diagnostics', 10, 200);
    """)
    conn.commit()
    conn.close()
    return db_path
