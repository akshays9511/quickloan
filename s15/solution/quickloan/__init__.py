import os
import sys

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Node/agent modules print status lines containing "→" for readability.
# Windows consoles (cmd/PowerShell) default stdout to the system codepage
# (cp1252), which cannot encode that character and crashes the whole graph
# mid-run. Reconfiguring to UTF-8 with errors="replace" makes console output
# safe on Windows, Docker/Linux, and under Streamlit alike.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv
load_dotenv()

if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
    os.environ.setdefault("LANGCHAIN_PROJECT",  os.getenv("LANGSMITH_PROJECT", "batch1-quickloan"))
