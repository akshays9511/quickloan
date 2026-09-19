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


def calculate_apr(
    principal: float, annual_rate_pct: float, tenure_months: int, processing_fee_pct: float
) -> dict:
    """Return the Annual Percentage Rate (APR) for a loan, per RBI's Key Fact
    Statement (KFS) methodology under the Digital Lending Directions: the
    annualized IRR of the loan's real cash flows -- what the borrower actually
    receives (principal minus the upfront processing fee) against the EMI
    stream they actually pay, computed at the nominal contracted rate.

    The nominal annual_rate_pct alone understates the true cost whenever a
    processing fee is charged, which is why RBI requires APR (not just the
    nominal rate) to be disclosed. There's no closed-form solution for the
    IRR, so this bisects on the monthly discount rate.

    Raises ValueError on invalid input (via calculate_emi_breakdown, or if
    the fee consumes the entire disbursed amount).
    """
    breakdown = calculate_emi_breakdown(principal, annual_rate_pct, tenure_months)
    emi = breakdown["emi"]

    processing_fee = principal * processing_fee_pct / 100
    net_disbursed  = principal - processing_fee
    if net_disbursed <= 0:
        raise ValueError("processing_fee_pct leaves no net disbursed amount.")

    def _pv(monthly_rate: float) -> float:
        if monthly_rate == 0:
            return emi * tenure_months
        return emi * (1 - (1 + monthly_rate) ** -tenure_months) / monthly_rate

    lo, hi = 0.0, 1.0  # monthly rate search range: 0% to 1200% p.a., wide enough for any real fee
    for _ in range(100):
        mid = (lo + hi) / 2
        if _pv(mid) > net_disbursed:
            lo = mid
        else:
            hi = mid
    monthly_irr = (lo + hi) / 2
    apr = ((1 + monthly_irr) ** 12 - 1) * 100

    return {
        "apr": apr,
        "processing_fee": processing_fee,
        "net_disbursed": net_disbursed,
    }
