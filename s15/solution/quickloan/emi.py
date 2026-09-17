"""
quickloan/emi.py
-----------------
Pure EMI calculation logic, shared between the MCP tool (mcp_server.py,
used by the chat agent) and the Streamlit EMI Calculator tab (app.py) so
the formula lives in exactly one place instead of being duplicated.
"""


def calculate_emi_breakdown(principal: float, annual_rate_pct: float, tenure_months: int) -> dict:
    """Return {'emi', 'total_payment', 'total_interest'} for a loan.

    Uses the standard reducing-balance EMI formula:
        EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    where r is the monthly interest rate (annual_rate_pct / 12 / 100) and n
    is the tenure in months.

    Raises ValueError on invalid input (non-positive principal/tenure, or a
    negative rate) so callers can turn it into whatever error format fits
    their context (a tool's plain-text response, a Streamlit st.error, etc.)
    """
    if principal <= 0:
        raise ValueError("principal must be a positive loan amount.")
    if tenure_months <= 0:
        raise ValueError("tenure_months must be a positive number of months.")
    if annual_rate_pct < 0:
        raise ValueError("annual_rate_pct cannot be negative.")

    monthly_rate = annual_rate_pct / 12 / 100
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        factor = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * factor / (factor - 1)

    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    return {
        "emi": emi,
        "total_payment": total_payment,
        "total_interest": total_interest,
    }
