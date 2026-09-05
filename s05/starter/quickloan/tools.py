"""
quickloan/tools.py
------------------
LLM clients and database tool functions for QuickLoan.

Session 5: adds query_rates() and query_eligibility() so the LLM can
look up live data instead of relying on hardcoded rates.
"""
import os
import sqlite3

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import DB_PATH, MODEL_NAME, TEMPERATURE, MAX_TOKENS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=10,
)


# ---------------------------------------------------------------------------
# TODO 1 of 4 -- Implement query_rates()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. If product_id.lower() == "all":
#        rows = conn.execute(
#            "SELECT lp.product_name, rs.min_cibil, rs.max_cibil, rs.annual_rate_pct "
#            "FROM rate_slabs rs JOIN loan_products lp ON rs.product_id = lp.product_id "
#            "ORDER BY lp.product_name, rs.min_cibil DESC"
#        ).fetchall()
#      Otherwise use a parameterised query:
#        rows = conn.execute(
#            "SELECT lp.product_name, rs.min_cibil, rs.max_cibil, rs.annual_rate_pct "
#            "FROM rate_slabs rs JOIN loan_products lp ON rs.product_id = lp.product_id "
#            "WHERE rs.product_id = ? "
#            "ORDER BY rs.min_cibil DESC",
#            (product_id,),
#        ).fetchall()
#   3. conn.close()
#   4. For each (name, min_cibil, max_cibil, rate) append:
#        f"{name}: {rate:.2f}% p.a. (CIBIL {min_cibil}-{max_cibil})"
#   5. Return "\n".join(lines) if lines else f"No rate data found for product: '{product_id}'."
# ---------------------------------------------------------------------------
@tool
def query_rates(product_id: str = "all") -> str:
    """Fetch current FastFinance India interest rates from the database.

    Args:
        product_id: Which loan rates to return. Options:
            "personal_loan", "home_loan", "business_loan", "gold_loan", or "all" (default).

    Returns formatted rate information as a plain-text string.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- Implement query_eligibility()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. If product_id.lower() == "all":
#        rows = conn.execute(
#            "SELECT lp.product_name, er.min_cibil, er.min_monthly_income, "
#            "er.min_age, er.max_age, er.employment_types "
#            "FROM eligibility_rules er JOIN loan_products lp ON er.product_id = lp.product_id "
#            "ORDER BY lp.product_name"
#        ).fetchall()
#      Otherwise (use parameterised query -- prevents SQL injection):
#        rows = conn.execute(
#            "... WHERE er.product_id = ?",
#            (product_id,),
#        ).fetchall()
#   3. conn.close()
#   4. If not rows: return f"No eligibility data found for product: '{product_id}'."
#   5. For each (name, min_cibil, min_income, min_age, max_age, emp_types) append:
#        f"{name}\n  Min CIBIL: {min_cibil} | Min income: Rs. {min_income}/mo | Age: {min_age}-{max_age} | {emp_types}"
#      Return "\n\n".join(parts)
# ---------------------------------------------------------------------------
@tool
def query_eligibility(product_id: str = "all") -> str:
    """Fetch FastFinance India loan eligibility criteria from the database.

    Args:
        product_id: Which loan eligibility to return.
                    Options: "personal_loan", "home_loan", "business_loan", "gold_loan", or "all" (default).

    Returns formatted eligibility information as a plain-text string.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- Bind tools to the LLM
# ---------------------------------------------------------------------------
# Create llm_with_tools by binding both tools to llm:
#   llm_with_tools = llm.bind_tools([query_rates, query_eligibility])
#
# This tells the LLM what tools are available so it can decide when to call them.
# llm_with_tools is used for the FIRST call in respond(). The second call
# (after tools have run) uses plain llm.
# ---------------------------------------------------------------------------
# TODO: add llm_with_tools = llm.bind_tools([query_rates, query_eligibility])


def _run_tool(tool_name: str, tool_args: dict) -> str:
    """Dispatch a tool call by name. Provided -- no changes needed."""
    _registry = {
        "query_rates":       query_rates,
        "query_eligibility": query_eligibility,
    }
    if tool_name not in _registry:
        return f"Unknown tool: {tool_name}"
    try:
        return _registry[tool_name].invoke(tool_args)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
