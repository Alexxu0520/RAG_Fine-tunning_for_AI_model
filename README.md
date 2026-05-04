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

| Mode | RAG | LoRA | Reranker |
|------|-----|------|----------|
| base | ❌ | ❌ | ❌ |
| base_rag | ✅ | ❌ | ✅ |
| base_rag_no_rerank | ✅ | ❌ | ❌ |
| lora | ❌ | ✅ | ❌ |
| lora_rag | ✅ | ✅ | ✅ |
| lora_rag_no_rerank | ✅ | ✅ | ❌ |


---

## System Architecture

```
User Question
    ↓
Retriever (Chroma Vector DB)
    ↓
(optional) Cross-encoder reranker
    ↓
Qwen2.5-3B-Instruct
    ↓
(optional) LoRA adapter
    ↓
Final Answer
```


---

## Dataset Source

Scraped from:

https://reddead.fandom.com/wiki/Category:Red_Dead_Redemption_II

Pipeline:

```
Wiki scrape
    ↓
clean JSONL corpus (rdr2_root_raw.jsonl)
    ↓
      ├── RAG dataset  (prepare_rag_json.py)
      └── LoRA dataset (prepare_lora_json_llm.py)
```

Outputs:

- `raw_docs/rdr2_root_raw.jsonl` — 2,559 raw pages
- `raw_docs/rdr2_rag.jsonl` — 2,556 cleaned pages for retrieval
- `lora_data/rdr2_lora_source_qa.jsonl` — 2,538 pages with LLM-generated QA pairs
- `lora_data/rdr2_lora_train_qa.jsonl` — 7,614 training examples


---

## Project Structure

```
rdr2_project/
├── raw_docs/
├── chroma_db/
├── outputs/
├── lora_data/
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── scrape_rdr2_root.py
├── prepare_rag_json.py
├── build_rag_from_jsonl.py
├── prepare_lora_json_llm.py
├── filter_lora_data.py
├── train_lora_patched.py
├── infer_compare_all.py
├── evaluate_lora_vs_rag.py
├── app.py
├── rag_models.py
├── eval_questions.json
└── README.md
```


---

## Installation

Recommended environment:

- Python 3.10+
- CUDA-enabled GPU recommended

Install dependencies:

```bash
pip install torch transformers peft trl chromadb sentence-transformers fastapi uvicorn
```


---

## Pipeline Steps

### 1. Scrape wiki dataset

```bash
python scrape_rdr2_root.py --max-depth 2 --min-chars 120
```

Output: `raw_docs/rdr2_root_raw.jsonl`


---

### 2. Prepare RAG dataset

```bash
python prepare_rag_json.py \
  --input raw_docs/rdr2_root_raw.jsonl \
  --output raw_docs/rdr2_rag.jsonl
```

Cleaning strategy:
- Removes wiki template artifacts, bare brackets, pure-number lines
- Footer markers (`Gallery`, `References`, `Navigation`) only truncate content
  appearing in the **last 25%** of a page — earlier occurrences are kept
- Minimum 140 chars after cleaning
- Result: **2,556 pages**, ~6.9M chars (93% text coverage of raw corpus)


---

### 3. Build vector database

```bash
python build_rag_from_jsonl.py
```

Output: `chroma_db/`

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: 700 chars with **150-char overlap** between consecutive chunks


---

### 4. Generate LoRA training dataset

```bash
python prepare_lora_json_llm.py --input raw_docs/rdr2_root_raw.jsonl
```

Outputs:
- `lora_data/rdr2_lora_source_qa.jsonl`
- `lora_data/rdr2_lora_train_qa.jsonl`

Cleaning strategy:
- Strips infobox labels, section headers, Quick Answers blocks, FAQ lines
- Preserves mid-sentence game-title references (e.g. "protagonist of Red Dead Redemption 2")
- Footer markers (`Gallery`, `References`, etc.) only truncate content in the **last 25%** of a page
- Uses **Qwen2.5-3B-Instruct** (32K context) to generate 3 QA pairs per page from the full cleaned text
- Generator prompt enforces: diverse question words, no title-echo questions, page-type-specific question strategies, complete-sentence answers, and a plain Western voice
- All LLM output is kept (no validation filtering at generation time)


---

### 5. Filter LoRA training data

```bash
python filter_lora_data.py --stats
```

Output: `lora_data/rdr2_lora_train_filtered.jsonl`

Filter rules (drop rate ~10%):
- Template questions: `What is the name of the mission?`, `What is X?`, generic `Who wrote the article?`
- Answers that are fragments, uncertain (`not specified`, `the protagonist`), or under 5 words
- `where` questions whose answers contain no location words
- Answers that simply restate the question or echo the page title


---

### 6. Train LoRA adapter

```bash
python train_lora_patched.py --train-file lora_data/rdr2_lora_train_filtered.jsonl --output-dir outputs/qwen25_rdr2_lora_v2
```


---

## Running Inference (CLI)

Single pipeline:

```bash
python infer_compare_all.py --mode base --question "Where is Saint Denis?"
```

All pipelines:

```bash
python infer_compare_all.py --mode all --question "Who is Arthur Morgan?"
```

With LoRA adapter:

```bash
python infer_compare_all.py \
  --adapter-path outputs/qwen25_rdr2_lora_v2 \
  --mode lora_rag \
  --question "Who is Arthur Morgan?"
```


---

## Running Web Interface

```bash
python -m uvicorn app:app --reload
```

Open: http://127.0.0.1:8000

Features:
- Interactive question input
- Pipeline selection
- Adapter switching
- Side-by-side model comparison


---

## Reranker

- Retriever: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Workflow: retrieve top-6 chunks → rerank → keep top-3 → generate answer
- No-rerank mode skips reranking and uses the first retrieved chunks directly


---

## Evaluation

```bash
python evaluate_lora_vs_rag.py
```

Evaluation dataset: `eval_questions.json`

Compares: `base`, `base_rag`, `lora`, `lora_rag`


---

## Data Quality Notes

### RAG cleaning fix
The original `prepare_rag_json.py` broke on footer section markers (`Gallery`,
`References`) that appear as mid-page section headings on wiki articles. This
caused pages to be truncated to just their table of contents, reducing text
coverage to ~15.7%. The fix applies footer cutoff only in the last 25% of a
page, raising coverage to ~93%.

### LoRA cleaning fix
The original pipeline stripped standalone `"Red Dead Redemption 2"` lines
unconditionally. On wiki pages these lines are hyperlinks that complete a
sentence on the previous line (e.g. "primary protagonist of / Red Dead
Redemption 2"), producing broken answers like `"primary protagonist of."`.
The fix preserves the line when the previous line ends without terminal
punctuation.


---

## Example Result

Question: `Where is Saint Denis?`

Best-performing pipeline: **LoRA + RAG + reranker**

Answer:
> Saint Denis is the capital of Lemoyne located in the Bayou Nwa region on
> the banks of the Lannahechee River.


---

## Future Improvements

- Hybrid retrieval (BM25 + dense embeddings)
- Larger reranker model
- Automatic scoring metrics (BLEU / ROUGE / embedding similarity)
- Streaming UI responses
