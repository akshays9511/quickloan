"""
s05/tests/conftest.py
---------------------
Pytest configuration for Session 5 tests.

Sets dummy API keys and clears any stale quickloan modules at collection time.
The seeded_db fixture creates a temporary SQLite database with real schema and
sample data so tool tests can run actual SQL without touching the production db.
"""
import os
import sqlite3
import sys

import pytest

for _key in list(sys.modules):
    if _key == "quickloan" or _key.startswith("quickloan."):
        sys.modules.pop(_key)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


@pytest.fixture
def seeded_db(tmp_path):
    """Temporary SQLite database with FastFinance schema and sample rows.

    Used by tool tests via monkeypatch.setattr("quickloan.tools.DB_PATH", seeded_db).
    The schema matches data/seed.py exactly.
    """
    db_path = tmp_path / "test_fastfinance.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE loan_products (
            product_id          TEXT PRIMARY KEY,
            product_name        TEXT NOT NULL,
            min_tenure_months   INTEGER,
            max_tenure_months   INTEGER,
            max_loan_amount     INTEGER,
            processing_fee_pct  REAL
        );
        CREATE TABLE rate_slabs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id       TEXT NOT NULL,
            min_cibil        INTEGER NOT NULL,
            max_cibil        INTEGER NOT NULL,
            annual_rate_pct  REAL NOT NULL
        );
        CREATE TABLE eligibility_rules (
            product_id          TEXT PRIMARY KEY,
            min_cibil           INTEGER NOT NULL,
            min_monthly_income  INTEGER NOT NULL,
            min_age             INTEGER NOT NULL,
            max_age             INTEGER NOT NULL,
            employment_types    TEXT NOT NULL
        );
        CREATE TABLE branch_contacts (
            branch_id  TEXT PRIMARY KEY,
            city       TEXT NOT NULL,
            address    TEXT NOT NULL,
            phone      TEXT NOT NULL,
            email      TEXT NOT NULL
        );
        INSERT INTO loan_products VALUES
            ('personal_loan', 'Personal Loan', 12,  60,  500000,  2.0),
            ('home_loan',     'Home Loan',     60,  360, 10000000, 1.0),
            ('gold_loan',     'Gold Loan',      3,  24,  1500000, 1.5);
        INSERT INTO rate_slabs (product_id, min_cibil, max_cibil, annual_rate_pct) VALUES
            ('personal_loan', 750, 900, 11.5),
            ('personal_loan', 720, 749, 13.0),
            ('home_loan',     750, 900,  8.75),
            ('gold_loan',       0, 900, 10.5);
        INSERT INTO eligibility_rules VALUES
            ('personal_loan', 700, 25000, 23, 58, 'salaried/self-employed'),
            ('home_loan',     720, 40000, 23, 65, 'salaried/self-employed'),
            ('gold_loan',       0, 10000, 18, 75, 'salaried/self-employed/retired');
        INSERT INTO branch_contacts VALUES
            ('b001', 'Pune',      'Aundh Road, Pune 411007',       '020-45678901', 'pune@fastfinance.in'),
            ('b002', 'Mumbai',    'Lower Parel, Mumbai 400013',    '022-45678902', 'mumbai@fastfinance.in'),
            ('b003', 'Bengaluru', 'Koramangala, Bengaluru 560034', '080-45678903', 'bengaluru@fastfinance.in');
    """)
    conn.commit()
    conn.close()
    return db_path
