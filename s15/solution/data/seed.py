"""
data/seed.py
------------
Seeds the QuickLoan SQLite database with FastFinance India data.

Run this before ingest.py:
    python data/seed.py

What this script does:
  1. Creates data/fastfinance_data.db
  2. Drops and recreates all tables (idempotent -- safe to run multiple times)
  3. Inserts sample data for loan products, eligibility rules, rate slabs,
     branch contacts, and rate history

Why are rates NOT in the documents?
  Interest rates and processing fees live exclusively in this database
  (rate_slabs and loan_products tables). Putting "personal loan: 11.5%" in
  a document would create two sources of truth. When rates change (re-run
  seed.py), the document would still show the old figure. The compliance node
  checks that QuickLoan never quotes rates from documents -- rates must always
  come from a tool call to this database.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "fastfinance_data.db"


def seed() -> None:
    print(f"Seeding QuickLoan database at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------
    # loan_products
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS loan_products")
    conn.execute("""
        CREATE TABLE loan_products (
            product_id          TEXT PRIMARY KEY,
            product_name        TEXT NOT NULL,
            min_tenure_months   INTEGER NOT NULL,
            max_tenure_months   INTEGER NOT NULL,
            max_loan_amount     INTEGER NOT NULL,
            processing_fee_pct  REAL NOT NULL
        )
    """)

    loan_products = [
        ("personal_loan", "Personal Loan",  12, 60,  500000,  2.0),
        ("home_loan",     "Home Loan",       60, 360, 10000000, 1.0),
        ("business_loan", "Business Loan",   12, 84,  2500000,  2.5),
        ("gold_loan",     "Gold Loan",        3, 24,  1500000,  1.5),
    ]
    conn.executemany(
        "INSERT INTO loan_products VALUES (?, ?, ?, ?, ?, ?)",
        loan_products,
    )
    print(f"  Inserted {len(loan_products)} loan products")

    # ------------------------------------------------------------------
    # eligibility_rules
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS eligibility_rules")
    conn.execute("""
        CREATE TABLE eligibility_rules (
            product_id          TEXT PRIMARY KEY,
            min_cibil           INTEGER NOT NULL,
            min_monthly_income  INTEGER NOT NULL,
            min_age             INTEGER NOT NULL,
            max_age             INTEGER NOT NULL,
            employment_types    TEXT NOT NULL
        )
    """)

    eligibility_rules = [
        ("personal_loan", 700, 25000, 23, 58, "salaried/self-employed"),
        ("home_loan",     720, 40000, 23, 65, "salaried/self-employed"),
        ("business_loan", 700, 50000, 25, 65, "self-employed"),
        ("gold_loan",       0, 10000, 18, 75, "salaried/self-employed/retired"),
    ]
    conn.executemany(
        "INSERT INTO eligibility_rules VALUES (?, ?, ?, ?, ?, ?)",
        eligibility_rules,
    )
    print(f"  Inserted {len(eligibility_rules)} eligibility rules")

    # ------------------------------------------------------------------
    # rate_slabs  (credit-score-based pricing)
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS rate_slabs")
    conn.execute("""
        CREATE TABLE rate_slabs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id       TEXT NOT NULL,
            min_cibil        INTEGER NOT NULL,
            max_cibil        INTEGER NOT NULL,
            annual_rate_pct  REAL NOT NULL
        )
    """)

    rate_slabs = [
        # personal_loan: 3 slabs
        ("personal_loan", 750, 900, 11.5),
        ("personal_loan", 720, 749, 13.0),
        ("personal_loan", 700, 719, 15.0),
        # home_loan: 2 slabs
        ("home_loan",     750, 900,  8.75),
        ("home_loan",     720, 749,  9.5),
        # business_loan: 2 slabs
        ("business_loan", 750, 900, 14.0),
        ("business_loan", 700, 749, 16.5),
        # gold_loan: flat rate (no CIBIL requirement)
        ("gold_loan",       0, 900, 10.5),
    ]
    conn.executemany(
        "INSERT INTO rate_slabs (product_id, min_cibil, max_cibil, annual_rate_pct) "
        "VALUES (?, ?, ?, ?)",
        rate_slabs,
    )
    print(f"  Inserted {len(rate_slabs)} rate slabs")

    # ------------------------------------------------------------------
    # branch_contacts
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS branch_contacts")
    conn.execute("""
        CREATE TABLE branch_contacts (
            branch_id  TEXT PRIMARY KEY,
            city       TEXT NOT NULL,
            address    TEXT NOT NULL,
            phone      TEXT NOT NULL,
            email      TEXT NOT NULL
        )
    """)

    branch_contacts = [
        ("branch_001", "Pune",      "Aundh Road, Pune 411007",           "020-45678901", "pune@fastfinance.in"),
        ("branch_002", "Mumbai",    "Lower Parel, Mumbai 400013",         "022-45678902", "mumbai@fastfinance.in"),
        ("branch_003", "Bengaluru", "Koramangala, Bengaluru 560034",     "080-45678903", "bengaluru@fastfinance.in"),
    ]
    conn.executemany(
        "INSERT INTO branch_contacts VALUES (?, ?, ?, ?, ?)",
        branch_contacts,
    )
    print(f"  Inserted {len(branch_contacts)} branch contacts")

    # ------------------------------------------------------------------
    # rate_history
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS rate_history")
    conn.execute("""
        CREATE TABLE rate_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id     TEXT NOT NULL,
            old_rate       REAL NOT NULL,
            new_rate       REAL NOT NULL,
            effective_date TEXT NOT NULL,
            changed_by     TEXT NOT NULL
        )
    """)

    rate_history = [
        ("personal_loan", 12.0, 11.5, "2024-10-01", "admin"),
        ("home_loan",      9.0,  8.75, "2025-02-01", "admin"),
    ]
    conn.executemany(
        "INSERT INTO rate_history (product_id, old_rate, new_rate, effective_date, changed_by) "
        "VALUES (?, ?, ?, ?, ?)",
        rate_history,
    )
    print(f"  Inserted {len(rate_history)} rate history rows")

    conn.commit()
    conn.close()
    print(f"\nDone. Database seeded at {DB_PATH}")
    print("Run 'python data/ingest.py' next to build the vector store.")


if __name__ == "__main__":
    seed()
