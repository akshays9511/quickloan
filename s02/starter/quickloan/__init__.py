"""
QuickLoan package -- Session 2: Multi-Turn Memory (US-02)
=========================================================

This file runs automatically when Python imports the quickloan package.
Use it to set up the environment before any other module loads.
"""
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# ---------------------------------------------------------------------------
# TODO 1 of 4 -- Environment setup
# ---------------------------------------------------------------------------
# Same as Session 1: import and call load_dotenv() so GROQ_API_KEY is
# available before tools.py tries to read it.
#
#   from dotenv import load_dotenv
#   load_dotenv()
#
# ---------------------------------------------------------------------------
