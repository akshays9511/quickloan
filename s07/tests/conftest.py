"""
s07/tests/conftest.py
---------------------
Pytest configuration and fixtures for Session 7 tests.

Creates a minimal in-memory SQLite database that mirrors the schema of
fastfinance_data.db so tests run without requiring data/seed.py to have been run.
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
    """Create a minimal fastfinance_data.db in a temp directory with known seed data."""
    db_path = tmp_path / "fastfinance_data.db"
    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        CREATE TABLE loan_products (
            product_id          TEXT PRIMARY KEY,
            product_name        TEXT NOT NULL,
            min_tenure_months   INTEGER NOT NULL,
            max_tenure_months   INTEGER NOT NULL,
            max_loan_amount     INTEGER NOT NULL,
            processing_fee_pct  REAL NOT NULL
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

        INSERT INTO loan_products VALUES
            ('personal_loan', 'Personal Loan', 12, 60,  500000, 2.0),
            ('home_loan',     'Home Loan',      60, 360, 10000000, 1.0),
            ('business_loan', 'Business Loan',  12, 84,  2500000, 2.5),
            ('gold_loan',     'Gold Loan',      3,  24,  1500000, 1.5);

        INSERT INTO rate_slabs (product_id, min_cibil, max_cibil, annual_rate_pct) VALUES
            ('personal_loan', 750, 900, 11.5),
            ('personal_loan', 720, 749, 13.0),
            ('personal_loan', 700, 719, 15.0),
            ('home_loan',     750, 900, 8.75),
            ('home_loan',     720, 749, 9.5),
            ('business_loan', 750, 900, 14.0);

        INSERT INTO eligibility_rules VALUES
            ('personal_loan', 700, 25000, 23, 58, 'salaried/self-employed'),
            ('home_loan',     720, 40000, 23, 65, 'salaried/self-employed'),
            ('business_loan', 700, 50000, 25, 65, 'self-employed'),
            ('gold_loan',     0,   10000, 18, 75, 'salaried/self-employed/retired');
    """)
    conn.commit()
    conn.close()
    return db_path
