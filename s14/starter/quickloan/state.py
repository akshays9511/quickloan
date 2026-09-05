from typing import TypedDict


class QuickLoanState(TypedDict):
    customer_message:  str
    response:          str
    history:           list[dict]
    query_type:        str
    retrieved_docs:    list[str]
    specialist:        str
    compliance_status: str
    blocked_reason:    str   # "" = clean; "pii", "injection", or "llamaguard" = blocked by guard
    llamaguard_score:  float # probability score from Llama Prompt Guard 2 (0.0–1.0); -1.0 = not called
