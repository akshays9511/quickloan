"""
QuickLoan -- Session 7: MCP Server (US-06 Part 1)
==================================================
Standalone MCP server that exposes QuickLoan's two database tools --
query_rates and query_eligibility -- over the MCP protocol. The agent
(quickloan/tools.py) launches this as a stdio subprocess and discovers
both tools automatically; no tool-calling code lives in the agent itself.

Run standalone:
    python s14b/solution/mcp_server.py

Inspect with MCP Inspector:
    npx @modelcontextprotocol/inspector python s14b/solution/mcp_server.py
    Open http://localhost:5173 -- both tools should appear.
"""

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------

mcp = FastMCP("quickloan-tools")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH  = DATA_DIR / "fastfinance_data.db"


def _get_db_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection to the given database path.

    check_same_thread=False: SQLite normally raises ProgrammingError if a connection
    is used from a thread other than the one that created it. LangGraph runs tool calls
    in a thread-pool executor, so the tool may execute on a different thread than the
    one that opened the connection. This flag disables that check.
    """
    try:
        return sqlite3.connect(str(db_path), check_same_thread=False)
    except Exception as e:
        raise RuntimeError(f"Could not connect to database at {db_path}: {e}")


def _execute_query(sql: str, params: tuple = ()) -> list:
    """Open a connection, run a single query, and return all rows."""
    conn = None
    try:
        conn = _get_db_connection(DB_PATH)
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database query failed: {e}") from e
    finally:
        if conn is not None:
            conn.close()


@mcp.tool()
def query_rates(product_id: str = "all") -> str:
    """Fetch current FastFinance India interest rates from the database.

    Args:
        product_id: Which loan rates to return. Options:
            "personal_loan" -- personal loan rate slabs by CIBIL score
            "home_loan"     -- home loan rate slabs by CIBIL score
            "business_loan" -- business loan rate slabs by CIBIL score
            "gold_loan"     -- gold loan flat rate
            "all"           -- all products (default)

    Returns formatted rate information as a plain-text string.
    """
    product_id = product_id.lower()

    if product_id == "all":
        rows = _execute_query(
            "SELECT lp.product_name, rs.min_cibil, rs.max_cibil, rs.annual_rate_pct "
            "FROM rate_slabs rs JOIN loan_products lp ON rs.product_id = lp.product_id "
            "ORDER BY lp.product_name, rs.min_cibil DESC"
        )
    else:
        rows = _execute_query(
            "SELECT lp.product_name, rs.min_cibil, rs.max_cibil, rs.annual_rate_pct "
            "FROM rate_slabs rs JOIN loan_products lp ON rs.product_id = lp.product_id "
            "WHERE rs.product_id = ? "
            "ORDER BY rs.min_cibil DESC",
            (product_id,),
        )

    lines = [
        f"{name}: {rate:.2f}% p.a. (CIBIL {min_cibil}-{max_cibil})"
        for name, min_cibil, max_cibil, rate in rows
    ]
    return "\n".join(lines) if lines else f"No rate data found for product: '{product_id}'."


@mcp.tool()
def query_eligibility(product_id: str = "all") -> str:
    """Fetch FastFinance India loan eligibility criteria from the database.

    Args:
        product_id: Which loan eligibility to return. Options:
            "personal_loan" -- personal loan eligibility rules
            "home_loan"     -- home loan eligibility rules
            "business_loan" -- business loan eligibility rules
            "gold_loan"     -- gold loan eligibility rules
            "all"           -- all products (default)

    Returns formatted eligibility information as a plain-text string.
    """
    product_id = product_id.lower()

    if product_id == "all":
        rows = _execute_query(
            "SELECT lp.product_name, er.min_cibil, er.min_monthly_income, "
            "er.min_age, er.max_age, er.employment_types "
            "FROM eligibility_rules er JOIN loan_products lp ON er.product_id = lp.product_id "
            "ORDER BY lp.product_name"
        )
    else:
        rows = _execute_query(
            "SELECT lp.product_name, er.min_cibil, er.min_monthly_income, "
            "er.min_age, er.max_age, er.employment_types "
            "FROM eligibility_rules er JOIN loan_products lp ON er.product_id = lp.product_id "
            "WHERE er.product_id = ? "
            "ORDER BY lp.product_name",
            (product_id,),
        )

    if not rows:
        return f"No eligibility data found for product: '{product_id}'."

    parts = [
        f"{name}\n"
        f"  Min CIBIL: {min_cibil} | Min income: Rs. {min_income}/mo | "
        f"Age: {min_age}-{max_age} | {emp_types}"
        for name, min_cibil, min_income, min_age, max_age, emp_types in rows
    ]
    return "\n\n".join(parts)


@mcp.tool()
def calculate_emi(product_id: str, principal: float, annual_rate_pct: float, tenure_months: int) -> str:
    """Calculate the Equated Monthly Installment (EMI) for a FastFinance India loan.

    Uses the standard reducing-balance EMI formula:
        EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    where r is the monthly interest rate (annual_rate_pct / 12 / 100) and n is
    the tenure in months. This is a deterministic calculation done in code --
    the agent must never compute EMI itself, only via this tool, so it always
    matches what a real customer would be quoted.

    Args:
        product_id: One of "personal_loan", "home_loan", "business_loan", "gold_loan".
        principal: Loan amount in Rupees (e.g. 500000 for Rs. 5,00,000).
        annual_rate_pct: Annual interest rate in percent (e.g. 11.5 for 11.5% p.a.).
            Get this from query_rates first -- never guess it.
        tenure_months: Loan tenure in months (e.g. 36 for 3 years).

    Returns a formatted EMI breakdown (monthly EMI, total payment, total
    interest) as a plain-text string, with a warning appended if the
    requested amount or tenure falls outside the product's normal range.
    """
    product_id = product_id.lower()

    if principal <= 0:
        return "Error: principal must be a positive loan amount."
    if tenure_months <= 0:
        return "Error: tenure_months must be a positive number of months."
    if annual_rate_pct < 0:
        return "Error: annual_rate_pct cannot be negative."

    rows = _execute_query(
        "SELECT product_name, min_tenure_months, max_tenure_months, max_loan_amount "
        "FROM loan_products WHERE product_id = ?",
        (product_id,),
    )
    if not rows:
        return (
            f"Error: unknown product '{product_id}'. Valid options: "
            "personal_loan, home_loan, business_loan, gold_loan."
        )

    product_name, min_tenure, max_tenure, max_amount = rows[0]

    monthly_rate = annual_rate_pct / 12 / 100
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        factor = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * factor / (factor - 1)

    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    lines = [
        f"EMI Calculation -- {product_name}",
        f"  Loan amount    : Rs. {principal:,.0f}",
        f"  Interest rate  : {annual_rate_pct:.2f}% p.a.",
        f"  Tenure         : {tenure_months} months",
        f"  Monthly EMI    : Rs. {emi:,.2f}",
        f"  Total payment  : Rs. {total_payment:,.2f}",
        f"  Total interest : Rs. {total_interest:,.2f}",
    ]

    if not (min_tenure <= tenure_months <= max_tenure):
        lines.append(
            f"  Note: {product_name} tenure is normally {min_tenure}-{max_tenure} months; "
            f"{tenure_months} months is outside that range."
        )
    if principal > max_amount:
        lines.append(
            f"  Note: {product_name} maximum loan amount is Rs. {max_amount:,}; "
            f"Rs. {principal:,.0f} exceeds that."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()  # STDIO transport by default
