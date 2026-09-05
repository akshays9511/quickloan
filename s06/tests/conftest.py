"""
s06/tests/conftest.py
---------------------
Pytest configuration for Session 6 tests.

Sets dummy API keys before any imports so evaluate.py does not raise
ValueError("GROQ_API_KEY not found") at collection time.
"""
import os
import sys

for _key in list(sys.modules):
    if _key == "evaluate" or _key.startswith("evaluate."):
        sys.modules.pop(_key)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
