import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

JSONL_PATH = Path("raw_docs/rdr2_rag.jsonl")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rdr2"
MAX_CHROMA_BATCH = 5000
CHUNK_SIZE = 700
EMBED_BATCH_SIZE = 32


def chunk_text(text: str, max_chars: int = CHUNK_SIZE):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += para + "\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def batched_indices(total_size: int, batch_size: int):
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        yield start, end


def main():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {JSONL_PATH}")

    all_chunks = []
    all_metadatas = []

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad JSON on line {line_num}: {e}")
                continue

            title = record.get("title", "Untitled")
            url = record.get("url", "")
            text = record.get("text", "").strip()

            if not text:
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "title": title,
                    "url": url,
                    "chunk_index": i,
                })

    print(f"Prepared {len(all_chunks)} chunks.")

    if not all_chunks:
        print("No chunks found. Exiting.")
        return

    print("Loading embedding model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = embed_model.encode(
        all_chunks,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True
    ).tolist()

    print("Connecting to Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing = collection.get()
    if existing and existing.get("ids"):
        print(f"Clearing existing collection with {len(existing['ids'])} items...")
        collection.delete(ids=existing["ids"])

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    print("Adding documents to Chroma in batches...")
    for start, end in batched_indices(len(ids), MAX_CHROMA_BATCH):
        print(f"Adding batch {start} to {end - 1} ...")
        collection.add(
            ids=ids[start:end],
            documents=all_chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=all_metadatas[start:end],
        )

    print("Indexed into Chroma.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database path: {CHROMA_DIR}")
    print(f"Total chunks stored: {len(ids)}")


if __name__ == "__main__":
    main()
