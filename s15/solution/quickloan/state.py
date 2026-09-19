from typing import TypedDict


class QuickLoanState(TypedDict):
    customer_message:  str
    response:          str
    history:           list[dict]
    query_type:        str
    retrieved_docs:    list[str]
    specialist:        str
    compliance_status: str
    blocked_reason:    str   # "" = clean; "injection" or "pii" = blocked by guard
    # RBI compliance audit trail -- populated by the Compliance Agent so app.py
    # can log a full record (what was flagged, why, what it became) even after
    # compliance_status itself is overwritten from "FAIL: ..." to "REVISED".
    compliance_reason: str   # "" for PASS; the FAIL reason, kept even after REVISED
    original_response: str   # the pre-revision draft; "" unless compliance_status == REVISED
