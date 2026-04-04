# RDR2 Assistant Project — Run Guide

This project contains three main QA modes:

1. **Qwen only** — no retrieval  
   Script: `test_qwen.py`

2. **RAG retrieval** — retrieve from Chroma, then answer with Qwen  
   Script: `ask_rag_retri.py`  
   (`ask_rag.py` is an older/similar version)

3. **RAG + reranking** — retrieve top-k chunks, rerank them, then answer with Qwen  
   Script: `ask_rag_reranker.py`

It also includes a script to build the Chroma database from scraped JSONL data:

- `build_rag_from_jsonl.py`

## 1. Recommended folder structure

```text
tuning/
├── test_qwen.py
├── ask_rag.py
├── ask_rag_retri.py
├── ask_rag_reranker.py
├── build_rag_from_jsonl.py
├── raw_docs/
│   └── rdr2_api.jsonl
├── chroma_db/              # created after building the index
└── docs/                   # optional older local text docs
```

## 2. Python version

Recommended:
- **Python 3.10 or 3.11**

You can try 3.14 if it already works on your machine, but some ML packages are often smoother on 3.10/3.11.

## 3. Install dependencies

Open terminal in the project folder and run:

```bash
python -m pip install -U torch transformers accelerate sentence-transformers chromadb
```

If you also want scraping support later, install:

```bash
python -m pip install requests beautifulsoup4 lxml
```

## 4. What each script does

### `test_qwen.py`
Runs **Qwen2.5-3B-Instruct only**, without retrieval.

Run:

```bash
python test_qwen.py
```

### `build_rag_from_jsonl.py`
Builds the Chroma vector database from `raw_docs/rdr2_api.jsonl`.

Run this when:
- you changed `raw_docs/rdr2_api.jsonl`
- you changed chunking logic
- you want to rebuild the index

Run:

```bash
python build_rag_from_jsonl.py
```

Expected output:

```text
Prepared XX chunks.
Indexed into Chroma.
```

### `ask_rag_retri.py`
Runs **retrieval + Qwen**.

Run:

```bash
python ask_rag_retri.py
```

Then type a question, for example:

```text
What mission introduces the legendary bear?
```

### `ask_rag_reranker.py`
Runs **retriever + reranker + Qwen**.

This is the best version right now.

Run:

```bash
python ask_rag_reranker.py
```

Then type a question, for example:

```text
What mission introduces the legendary bear?
```

## 5. Typical workflow

### If you already have `raw_docs/rdr2_api.jsonl` and want to rebuild the index

```bash
python build_rag_from_jsonl.py
python ask_rag_reranker.py
```

### If `chroma_db/` is already built and you only want to ask questions

```bash
python ask_rag_reranker.py
```

### If you want to test the base model only

```bash
python test_qwen.py
```

## 6. Suggested test questions

```text
What mission introduces the legendary bear?
Where is the Legendary Bharati Grizzly Bear located?
How does Arthur get tuberculosis?
Where can I find the White Arabian horse?
```

## 7. Which script should I use?

### Use `test_qwen.py` if:
- you want to test **Qwen alone**
- you want a baseline without RAG

### Use `ask_rag_retri.py` if:
- you want **basic retrieval + generation**
- you want to compare against reranking

### Use `ask_rag_reranker.py` if:
- you want the **best current pipeline**
- you want **retriever + reranker + Qwen**

## 8. Common warnings

You may see warnings like:
- `UNEXPECTED ... position_ids`
- Hugging Face symlink warnings on Windows
- unauthenticated HF Hub warnings

These are usually **not fatal** if the model still loads and runs.

## 9. Troubleshooting

### Error: `collection not found`
This usually means the Chroma index was not built yet.

Run:

```bash
python build_rag_from_jsonl.py
```

### Error: missing `raw_docs/rdr2_api.jsonl`
Make sure this file exists in:

```text
raw_docs/rdr2_api.jsonl
```

### The answer looks wrong
Possible reasons:
- the wrong chunk was retrieved
- the scraped text is noisy
- the index needs rebuilding
- the question is not covered well in the corpus

Try:
1. rerun with `ask_rag_reranker.py`
2. inspect the retrieved chunks
3. improve the underlying JSONL data
4. rebuild the index

## 10. Current model stack

### Base generator
- `Qwen/Qwen2.5-3B-Instruct`

### Retriever embedding model
- `sentence-transformers/all-MiniLM-L6-v2`

### Reranker
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Vector DB
- `Chroma`

## 11. Short explanation of the three systems

### Qwen only
```text
question → Qwen → answer
```

### RAG retrieval
```text
question → retriever → Chroma chunks → Qwen → answer
```

### RAG + reranking
```text
question → retriever → top-k chunks → reranker → best chunks → Qwen → answer
```

## 12. Best script for your report/demo

For demos or experiments, use:

```bash
python ask_rag_reranker.py
```

That is the most complete version in this folder.
