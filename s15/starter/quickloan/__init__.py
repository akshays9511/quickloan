import os
import sys

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Applied at package import so every entry point is covered -- the Streamlit app,
# `python -m quickloan.agent`, and pytest alike.
#
# On Windows the console defaults to cp1252, which cannot encode characters the
# LLM routinely emits (U+2019 curly quote, U+2011 non-breaking hyphen, U+202F
# narrow no-break space). Printing a response containing one raises
# UnicodeEncodeError and kills the graph mid-request. errors="replace" guarantees
# a log or response line can never crash a run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
    os.environ.setdefault("LANGCHAIN_PROJECT",  os.getenv("LANGSMITH_PROJECT", "batch1-quickloan"))
