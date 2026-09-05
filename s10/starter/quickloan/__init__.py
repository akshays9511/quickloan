import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from dotenv import load_dotenv
load_dotenv()

if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
    os.environ.setdefault("LANGCHAIN_PROJECT",  os.getenv("LANGSMITH_PROJECT", "batch1-quickloan"))
