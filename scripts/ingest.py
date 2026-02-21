"""
WhatsApp RAG Bot - Chat Data Ingestion Script
Parses clearedtext.txt (INPUT/OUTPUT pairs) and loads into ChromaDB.

Usage:
    python ingest.py              # Ingest clearedtext.txt into ChromaDB
    python ingest.py --stats      # Show ChromaDB collection stats
"""

import os
import sys
import re
import chromadb
from sentence_transformers import SentenceTransformer

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHAT_FILE = os.path.join(PROJECT_ROOT, "clearedtext.txt")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "global_style"
MAX_OUTPUT_LENGTH = 300  # Skip forwarded messages / study notes


def parse_chat_file(filepath: str) -> list:
    """Parse INPUT/OUTPUT pairs from clearedtext.txt"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    pairs = []
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Match INPUT: ... followed by OUTPUT: ... (multi-line outputs supported)
        match = re.match(r"INPUT:\s*(.*?)(?:\nOUTPUT:\s*)(.*)", block, re.DOTALL)
        if match:
            input_text = match.group(1).strip()
            output_text = match.group(2).strip()

            # Skip empty, too-short, or placeholder-only pairs
            if not input_text or not output_text:
                continue
            if output_text in [".", ",", "[PHONE]", "[EMAIL]"]:
                continue

            # Skip forwarded messages / study notes (too long to be personal style)
            if len(output_text) > MAX_OUTPUT_LENGTH:
                print(f"  [SKIP] Too long ({len(output_text)} chars): {output_text[:50]}...")
                continue

            pairs.append({"input": input_text, "output": output_text})

    return pairs


def ingest(pairs: list):
    """Embed INPUT messages and store OUTPUT replies in ChromaDB."""
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Connecting to ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh ingestion - delete old collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing '{COLLECTION_NAME}' collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Global style examples from WhatsApp chat history"},
    )

    print(f"Embedding {len(pairs)} pairs...")

    documents = []
    metadatas = []
    ids = []
    embeddings = []

    for i, pair in enumerate(pairs):
        # Document = your reply (OUTPUT)
        # Embedding = the trigger message (INPUT) so we match by incoming similarity
        # Metadata = the trigger text for reference
        documents.append(pair["output"])
        metadatas.append({"trigger_message": pair["input"]})
        ids.append(f"pair_{i}")
        embeddings.append(model.encode(pair["input"]).tolist())

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    print(f"\n=== Ingestion Complete ===")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Documents  : {collection.count()}")
    print(f"  ChromaDB   : {CHROMA_DIR}")


def show_stats():
    """Print ChromaDB collection stats."""
    print(f"Connecting to ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"\n=== ChromaDB Stats ===")
        print(f"  Collection : {COLLECTION_NAME}")
        print(f"  Documents  : {collection.count()}")

        # Show a few sample documents
        sample = collection.peek(5)
        if sample and sample["documents"]:
            print(f"\n--- Sample Entries ---")
            for doc, meta in zip(sample["documents"], sample["metadatas"]):
                trigger = meta.get("trigger_message", "?")
                print(f"  IN:  {trigger}")
                print(f"  OUT: {doc}")
                print()
    except Exception:
        print(f"No collection '{COLLECTION_NAME}' found. Run ingestion first.")


def main():
    if "--stats" in sys.argv:
        show_stats()
        return

    if not os.path.exists(CHAT_FILE):
        print(f"ERROR: Chat file not found: {CHAT_FILE}")
        print(f"Place your clearedtext.txt in the project root.")
        sys.exit(1)

    print(f"Parsing: {CHAT_FILE}")
    pairs = parse_chat_file(CHAT_FILE)
    print(f"Found {len(pairs)} valid INPUT/OUTPUT pairs\n")

    if not pairs:
        print("ERROR: No valid pairs found!")
        sys.exit(1)

    # Preview
    print("--- Preview (first 3 pairs) ---")
    for p in pairs[:3]:
        print(f"  IN:  {p['input']}")
        print(f"  OUT: {p['output']}")
        print()

    ingest(pairs)


if __name__ == "__main__":
    main()
