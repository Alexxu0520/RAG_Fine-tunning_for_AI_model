# RDR2 Knowledge Assistant (RAG + LoRA + Reranker Comparison)

## Project Overview

This project builds a Red Dead Redemption 2 question-answering assistant using:

- Retrieval-Augmented Generation (RAG)
- LoRA fine-tuning (QLoRA-style adapter training)
- Cross-encoder reranking
- FastAPI-based interactive comparison interface

The system compares multiple inference pipelines to evaluate how retrieval,
fine-tuning, and reranking affect answer quality.

Supported pipelines:

Base
Base + RAG
Base + RAG (no reranker)
LoRA
LoRA + RAG
LoRA + RAG (no reranker)


---

## System Architecture

Pipeline structure:

User Question
    ↓
Retriever (Chroma Vector DB)
    ↓
(optional) Cross‑encoder reranker
    ↓
Qwen2.5‑3B‑Instruct
    ↓
(optional) LoRA adapter
    ↓
Final Answer


Comparison modes:

| Mode | RAG | LoRA | Reranker |
|------|-----|------|----------|
| base | ❌ | ❌ | ❌ |
| base_rag | ✅ | ❌ | ✅ |
| base_rag_no_rerank | ✅ | ❌ | ❌ |
| lora | ❌ | ✅ | ❌ |
| lora_rag | ✅ | ✅ | ✅ |
| lora_rag_no_rerank | ✅ | ✅ | ❌ |


---

## Dataset Source

Dataset scraped from:

https://reddead.fandom.com/wiki/Category:Red_Dead_Redemption_II

Pipeline:

Wiki scrape
↓
clean JSONL corpus
↓
split into

RAG dataset
LoRA training dataset


Outputs:

raw_docs/rdr2_root_raw.jsonl
raw_docs/rdr2_rag.jsonl


---

## Project Structure

Example structure:

rdr2_project/

raw_docs/
chroma_db/
outputs/
lora_data/

scrape_rdr2_root.py
prepare_rag_json.py
build_rag_from_jsonl.py

python prepare_lora_json_llm.py
train_lora_patched.py

infer_compare_all.py
evaluate_lora_vs_rag.py

app.py

static/
    index.html
    styles.css
    app.js

eval_questions.json
README.md


---

## Installation

Recommended environment:

Python 3.10+
CUDA-enabled GPU recommended

Install dependencies:

pip install torch transformers peft trl chromadb sentence-transformers fastapi uvicorn


---

## Pipeline Steps

### 1. Scrape wiki dataset

python scrape_rdr2_root.py --max-depth 2 --min-chars 120

Output:

raw_docs/rdr2_root_raw.jsonl


---

### 2. Prepare RAG dataset

python prepare_rag_json.py --input raw_docs/rdr2_root_raw.jsonl --output raw_docs/rdr2_rag.jsonl


---

### 3. Build vector database

python build_rag_from_jsonl.py

Output:

chroma_db/

Embedding model used:

sentence-transformers/all-MiniLM-L6-v2


---

### 4. Generate LoRA training dataset

python prepare_lora_json_llm.py --input raw_docs/rdr2_root_raw.jsonl

Outputs:

lora_data/rdr2_lora_source_llm_v2.jsonl
lora_data/rdr2_lora_train_llm_v2.jsonl


---

### 5. Train LoRA adapter

python train_lora_patched.py --train-file lora_data/rdr2_lora_train_llm_v2.jsonl --output-dir outputs/qwen25_rdr2_lora


---

## Running Inference (CLI)

Example:

python infer_compare_all.py --mode base --question "Where is Saint Denis?"


Compare all pipelines:

python infer_compare_all.py --mode all --question "Who is Arthur Morgan?"


---

## Running Web Interface

Start server:

python -m uvicorn app:app --reload

Open:

http://127.0.0.1:8000

Features:

interactive question input
pipeline selection
adapter switching
side‑by‑side model comparison


---

## Reranker Explanation

Retriever model:

sentence-transformers/all-MiniLM-L6-v2

Reranker model:

cross-encoder/ms-marco-MiniLM-L-6-v2

Workflow:

retrieve top‑6 chunks
rerank chunks
keep top‑3
generate grounded answer

No‑rerank mode skips reranking and directly uses first retrieved chunks.


---

## Evaluation

Evaluation dataset:

eval_questions.json

Run evaluation:

python evaluate_lora_vs_rag.py

Compares:

base
base_rag
lora
lora_rag


---

## Example Result

Question:

Where is Saint Denis?

Best-performing pipeline:

LoRA + RAG + reranker

Example answer:

Saint Denis is the capital of Lemoyne located in the Bayou Nwa region on the
banks of the Lannahechee River.


---

## Key Contributions

This project demonstrates:

dataset scraping pipeline
RAG retrieval system
LoRA fine‑tuning workflow
reranker integration
multi‑pipeline comparison framework
FastAPI evaluation interface


---

## Future Improvements

Possible extensions:

hybrid retrieval (BM25 + embeddings)
larger reranker model
automatic scoring metrics (BLEU / ROUGE / embedding similarity)
streaming UI responses
