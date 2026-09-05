# QuickLoan

AI loan pre-qualification assistant for FastFinance India (NBFC). Built during the Agentic AI Engineering course (Batch 1) as a Launchpad project.

## Setup

```bash
# 1. Install dependencies (do this once)
pip install -r requirements.txt

# 2. Copy environment variables and fill in your API keys
cp .env.example .env   # Mac/Linux
# copy .env.example .env   (Windows)

```

> **data/seed.py and data/ingest.py** — not included yet. They set up the SQLite database and ChromaDB vector store, which are introduced in later sessions.

## Session Folders

Session starter code is released one session at a time. Today's folder:

- `s01/` — Session 1: Basic conversational agent

To run after completing the TODOs:

```bash
cd s01/starter/
python -m quickloan.agent
```

Each session folder has a `CLAUDE_CODE_PROMPTS.md` with ready-to-use prompts for Claude Code.

## Reference

- PRD: `quickloan-prd.md`
- Glossary: `ai-glossary.md`
