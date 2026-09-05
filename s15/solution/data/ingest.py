"""
data/ingest.py
--------------
Ingests FastFinance India loan documents into ChromaDB.

Run after seed.py:
    python data/ingest.py

What this script does:
  1. Reads all .md files from data/documents/
  2. Splits them into overlapping chunks (better retrieval precision)
  3. Generates embeddings using a free local model (all-MiniLM-L6-v2)
  4. Writes the vector store to data/vectorstore/ (persisted to disk)

First run: downloads ~90 MB of embedding model weights to ~/.cache/huggingface/.
Subsequent runs: uses the cache. No internet connection required after first run.

The script is idempotent -- it deletes and rebuilds the vector store on every run.
This mirrors seed.py's DROP TABLE / CREATE TABLE pattern. Participants can add or
edit a document and re-run ingest.py to make the change searchable, without any
leftover stale chunks from the previous run.

Why local embeddings (HuggingFace) rather than OpenAI embeddings?
  Using OpenAI for embeddings would require participants to have an OpenAI key
  from Session 1. We delay the OpenAI key requirement to Session 6 (eval judge).
  all-MiniLM-L6-v2 is fast, free, and produces good retrieval quality for
  English policy documents of this length.

Why are rates NOT in these documents?
  Rates are in data/fastfinance_data.db (see seed.py). Putting "personal loan:
  11.5%" in a document would create two sources of truth. When rates change
  (seed.py is re-run), the document would still show the old figure while the
  database has the new one. The compliance node checks that QuickLoan never
  retrieves rate information from documents -- rates must come from a tool call.

Windows path note:
  ChromaDB's persist_directory requires a string. We pass str(VECTOR_DIR) where
  VECTOR_DIR is a pathlib.Path object. This gives the correct path on Windows
  without manual string manipulation.
"""

import shutil
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

DATA_DIR   = Path(__file__).parent
DOCS_DIR   = DATA_DIR / "documents"
VECTOR_DIR = DATA_DIR / "vectorstore"

EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# Smaller chunks (500 chars) give more precise retrieval but may split context
# across chunks. 50-char overlap ensures a sentence that straddles a boundary
# is preserved in at least one chunk.


def load_documents() -> List[Document]:
    """Load every .md file in the documents directory.

    Each document is tagged with its filename in metadata["source"].
    That tag appears in LangSmith traces (from Session 4 onward, when basic
    tracing is introduced with US-03) so you can see exactly which document
    contributed to each agent response.
    """
    if not DOCS_DIR.exists():
        print(f"Error: Documents directory not found at {DOCS_DIR}", file=sys.stderr)
        print("Make sure you are running this script from the quickloan/ folder.", file=sys.stderr)
        sys.exit(1)

    docs = []
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"Error: No .md files found in {DOCS_DIR}", file=sys.stderr)
        sys.exit(1)

    for path in md_files:
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        if not loaded:
            print(f"  Warning: {path.name} loaded as empty -- skipping", file=sys.stderr)
            continue
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)
        print(f"  Loaded: {path.name:40s} ({len(loaded[0].page_content):,} chars)")

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for retrieval.

    RecursiveCharacterTextSplitter tries to split at paragraph boundaries first,
    then sentence boundaries, then character boundaries. This preserves semantic
    context better than splitting at fixed character positions.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def main() -> None:
    print("Ingesting FastFinance India documents into ChromaDB")
    print(f"  Source : {DOCS_DIR}")
    print(f"  Target : {VECTOR_DIR}\n")

    # Delete the existing vector store before rebuilding. This makes the script
    # idempotent: running it twice produces the same result as running it once.
    # Without this, Chroma.from_documents() appends chunks to any existing
    # collection, causing duplicate retrieval results in Session 4.
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)
        print(f"  Cleared existing vector store (idempotent reset)\n")

    print("Loading documents...")
    docs = load_documents()

    chunks = split_documents(docs)
    print(f"\nSplit {len(docs)} documents into {len(chunks)} chunks")
    print(f"  chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}\n")

    print(f"Loading embedding model: {EMBED_MODEL}")
    print("  First run downloads ~90 MB. Subsequent runs use cache.\n")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    print("Building vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )

    print(f"\nDone. {len(chunks)} chunks stored at {VECTOR_DIR}")
    print("Run 'python data/ingest.py' again after adding or editing documents.")
    print("\nSetup complete. Run 'python s01/solution/main.py' to start QuickLoan.")


if __name__ == "__main__":
    main()
