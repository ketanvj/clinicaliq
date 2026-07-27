"""
s05/tests/conftest.py
---------------------
Pytest configuration for Session 5 tests.

Sets dummy API keys and clears any stale clinicaliq modules at collection time.
The seeded_db fixture creates a temporary SQLite database with real schema and
sample data so tool tests can run actual SQL without touching the production db.
"""
import os
import sqlite3
import sys

import pytest

for _key in list(sys.modules):
    if _key == "clinicaliq" or _key.startswith("clinicaliq."):
        sys.modules.pop(_key)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


@pytest.fixture
def seeded_db(tmp_path):
    """Temporary SQLite database with Apollo Health Clinic schema and sample rows.

    Used by tool tests via monkeypatch.setattr("clinicaliq.tools.DB_PATH", seeded_db).
    The schema matches data/seed.py exactly.
    """
    db_path = tmp_path / "test_clinic.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE doctors (
            doctor_id        TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            specialty        TEXT NOT NULL,
            available_days   TEXT NOT NULL,
            consultation_fee INTEGER NOT NULL
        );
        CREATE TABLE services (
            service_id            TEXT PRIMARY KEY,
            name                  TEXT NOT NULL,
            department            TEXT NOT NULL,
            average_duration_mins INTEGER NOT NULL,
            price                 INTEGER NOT NULL
        );
        CREATE TABLE health_packages (
            package_id      TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            tests_included  TEXT NOT NULL,
            price           INTEGER NOT NULL,
            recommended_for TEXT NOT NULL
        );
        INSERT INTO doctors VALUES
            ('dr_001', 'Dr. Meera Nair',    'General Medicine', 'Mon-Fri',      500),
            ('dr_002', 'Dr. Rajesh Kumar',  'Cardiology',       'Mon/Wed/Fri',  800),
            ('dr_003', 'Dr. Sunita Sharma', 'Paediatrics',      'Tue/Thu/Sat',  600);
        INSERT INTO services VALUES
            ('ecg',         'ECG',               'Cardiology',  20,   350),
            ('blood_panel', 'Basic Blood Panel', 'Diagnostics', 30,   450),
            ('echo',        'Echocardiogram',    'Cardiology',  45,  1800);
        INSERT INTO health_packages VALUES
            ('pkg_001', 'Full Body Checkup',  'CBC/LFT/ECG',       3500, 'Adults 30+'),
            ('pkg_002', 'Cardiac Screening',  'ECG/Echo/Lipids',   2800, 'Adults 40+');
    """)
    conn.commit()
    conn.close()
    return db_path
