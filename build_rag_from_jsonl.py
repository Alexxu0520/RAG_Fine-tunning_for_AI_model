import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

JSONL_PATH = Path("raw_docs/rdr2_api.jsonl")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rdr2"

def chunk_text(text: str, max_chars: int = 700):
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

def main():
    all_chunks = []
    all_metadatas = []

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            title = record["title"]
            url = record["url"]
            text = record["text"]

            for i, chunk in enumerate(chunk_text(text)):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "title": title,
                    "url": url,
                    "chunk_index": i,
                })

    print(f"Prepared {len(all_chunks)} chunks.")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(all_chunks).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    print("Indexed into Chroma.")

if __name__ == "__main__":
    main()