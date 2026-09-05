from typing import TypedDict


class QuickLoanState(TypedDict):
    customer_message: str
    response:         str
    history:          list[dict]

    # -----------------------------------------------------------------------
    # TODO 1 of 4 -- Add the query_type field
    # -----------------------------------------------------------------------
    # query_type is written by classify() and read by route_query().
    # Type hint: str
    # Valid values: "SIMPLE", "COMPLEX", "OUT_OF_SCOPE"
    #
    # TODO: add  query_type: str
