"""
QuickLoan -- Session 7: MCP Server (US-06 Part 1)
==================================================
STARTER FILE -- your task is to implement the two TODO sections below.

Goal
  Build a standalone MCP server that exposes QuickLoan's two database
  tools -- query_rates and query_eligibility -- over the MCP protocol.
  When finished, MCP Inspector should be able to discover both tools and
  call them without touching any agent code.

What is already done for you
  - FastMCP server created: mcp = FastMCP("quickloan-tools")
  - Both @mcp.tool() decorators and function signatures are in place
  - DB_PATH points to the same fastfinance_data.db used in Session 5
  - mcp.run() at the bottom starts the STDIO server

Your task
  Implement the SQL queries inside TODO 1 (query_rates) and TODO 2
  (query_eligibility). The logic is identical to s05/solution/quickloan/tools.py --
  open that file, find the two @tool functions, and adapt them here.
  The only change: replace @tool with @mcp.tool() (already done).

Run when done
  python s07/starter/mcp_server.py

Inspect with MCP Inspector
  npx @modelcontextprotocol/inspector python s07/starter/mcp_server.py
  Open http://localhost:5173 -- both tools should appear.
"""

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instantiation -- already done for you
# ---------------------------------------------------------------------------

mcp = FastMCP("quickloan-tools")

# ---------------------------------------------------------------------------
# Configuration -- already done for you
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH  = DATA_DIR / "fastfinance_data.db"

# ---------------------------------------------------------------------------
# TODO 1: Implement query_rates
# Hint: copy the query_rates() function from s05/solution/quickloan/tools.py.
#       The SQL queries and return format are identical.
#       The only difference: @tool becomes @mcp.tool() (already in place).
# ---------------------------------------------------------------------------

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
    # TODO 1: Connect to DB_PATH with sqlite3.connect()
    # If product_id == "all":
    #   SELECT lp.product_name, rs.min_cibil, rs.max_cibil, rs.annual_rate_pct
    #   FROM rate_slabs rs JOIN loan_products lp ON rs.product_id = lp.product_id
    #   ORDER BY lp.product_name, rs.min_cibil DESC
    # Else (filter by product_id):
    #   Same SELECT with "WHERE rs.product_id = ? ORDER BY rs.min_cibil DESC"
    #   Pass (product_id,) as the parameter
    # Format each row as: f"{name}: {rate:.2f}% p.a. (CIBIL {min_cibil}-{max_cibil})"
    # Close the connection and return "\n".join(lines) or a "No rate data found" message
    raise NotImplementedError("TODO 1: implement the SQL queries for query_rates()")


# ---------------------------------------------------------------------------
# TODO 2: Implement query_eligibility
# Hint: copy the query_eligibility() function from s05/solution/quickloan/tools.py.
#       Same SQL, same return format, same @mcp.tool() decorator.
# ---------------------------------------------------------------------------

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
    # TODO 2: Connect to DB_PATH with sqlite3.connect()
    # If product_id == "all":
    #   SELECT lp.product_name, er.min_cibil, er.min_monthly_income,
    #          er.min_age, er.max_age, er.employment_types
    #   FROM eligibility_rules er JOIN loan_products lp ON er.product_id = lp.product_id
    #   ORDER BY lp.product_name
    # Else (filter by product_id):
    #   Same SELECT with "WHERE er.product_id = ? ORDER BY lp.product_name"
    #   Pass (product_id,) as the parameter
    # If no rows found: return f"No eligibility data found for product: '{product_id}'."
    # Format each row as:
    #   f"{name}\n  Min CIBIL: {min_cibil} | Min income: Rs. {min_income}/mo | "
    #   f"Age: {min_age}-{max_age} | {emp_types}"
    # Return entries joined by "\n\n"
    raise NotImplementedError("TODO 2: implement the SQL queries for query_eligibility()")


# ---------------------------------------------------------------------------
# Entry point -- already done for you
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()  # STDIO transport by default
