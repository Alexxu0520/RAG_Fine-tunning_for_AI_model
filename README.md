# RDR2 Knowledge Assistant — RAG + LoRA + Hybrid Retrieval

## Project Overview

A Red Dead Redemption 2 question-answering assistant built with:

- **Retrieval-Augmented Generation (RAG)** — dense vector search via ChromaDB
- **Hybrid retrieval** — BM25 + dense + Reciprocal Rank Fusion (RRF)
- **LoRA fine-tuning** — QLoRA-style adapter on Qwen2.5-3B-Instruct
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2`
- **FastAPI web UI** — side-by-side pipeline comparison

Six inference pipelines are compared simultaneously:

| Mode | RAG | Retrieval | LoRA | Reranker |
|------|-----|-----------|------|----------|
| `base` | ❌ | — | ❌ | ❌ |
| `base_rag` | ✅ | Dense | ❌ | ✅ |
| `base_hybrid_rag` | ✅ | BM25 + Dense | ❌ | ✅ |
| `lora` | ❌ | — | ✅ | ❌ |
| `lora_rag` | ✅ | Dense | ✅ | ✅ |
| `lora_hybrid_rag` | ✅ | BM25 + Dense | ✅ | ✅ |

---

## System Architecture

```
User Question
    │
    ├─── [RAG path] ──────────────────────────────────────────┐
    │        │                                                 │
    │   Dense retrieval          BM25 retrieval                │
    │   (ChromaDB + MiniLM)      (rank_bm25)                   │
    │        └──────── RRF fusion ────────────┘                │
    │                       ↓                                  │
    │              Cross-encoder reranker                      │
    │              top-6 → top-3 chunks                        │
    │                       ↓                                  │
    │              Grounded prompt                             │
    │                                                          │
    └─── [Direct path] ───────────────────────────────────────┘
                           ↓
              Qwen2.5-3B-Instruct (4-bit NF4)
              + optional LoRA adapter (qwen25_rdr2_lora_v2)
                           ↓
                      Final Answer
```

---

## Dataset

**Source:** https://reddead.fandom.com/wiki/Category:Red_Dead_Redemption_II

**Pipeline:**

```
Wiki scrape  →  rdr2_root_raw.jsonl
                    │
                    ├── prepare_rag_json.py     →  rdr2_rag.jsonl  →  ChromaDB
                    │
                    └── prepare_lora_json_llm.py
                              ↓
                        rdr2_lora_source_qa.jsonl   (LLM-generated QA)
                              ↓
                        filter_lora_data.py
                              ↓
                        rdr2_lora_train_filtered.jsonl
                              ↓
                        train_lora_patched.py
                              ↓
                        outputs/qwen25_rdr2_lora_v2
```

**Corpus stats:**

| File | Records | Notes |
|------|---------|-------|
| `raw_docs/rdr2_root_raw.jsonl` | ~2,559 pages | Raw scraped wiki |
| `raw_docs/rdr2_rag.jsonl` | ~2,556 pages | Cleaned for retrieval |
| `lora_data/rdr2_lora_source_qa.jsonl` | ~2,538 pages | LLM-generated QA |
| `lora_data/rdr2_lora_train_filtered.jsonl` | ~6,900 examples | Final training set |

---

## Project Structure

```
rdr2_project/
├── raw_docs/
│   ├── rdr2_root_raw.jsonl          # raw scraped corpus
│   ├── rdr2_rag.jsonl               # cleaned RAG corpus
│   └── rdr2_root_titles.json
├── lora_data/
│   ├── rdr2_lora_source_qa.jsonl    # LLM-generated QA pairs
│   ├── rdr2_lora_train_filtered.jsonl  # filtered training set
│   └── test100_*.jsonl              # 100-record test batches
├── chroma_db/                       # vector store (not committed)
├── outputs/                         # LoRA adapters (not committed)
│   └── qwen25_rdr2_lora_v2/        # 476-step full-dataset adapter
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
├── eval_questions.json
├── evaluation_results_v2.json
└── README.md
```

---

## Installation

```bash
pip install torch transformers peft trl chromadb sentence-transformers \
            rank_bm25 fastapi uvicorn
```

Requires a CUDA-capable GPU (tested on 12 GB VRAM).

---

## Pipeline Steps

### 1. Scrape wiki

```bash
python scrape_rdr2_root.py --max-depth 2 --min-chars 120
```

Output: `raw_docs/rdr2_root_raw.jsonl`

---

### 2. Prepare RAG corpus

```bash
python prepare_rag_json.py \
  --input raw_docs/rdr2_root_raw.jsonl \
  --output raw_docs/rdr2_rag.jsonl
```

- Strips wiki template artifacts, bare brackets, pure-number lines
- Footer markers (`Gallery`, `References`) only truncate content in the **last 25%** of a page
- Minimum 140 chars after cleaning → ~93% text coverage

---

### 3. Build vector database

```bash
python build_rag_from_jsonl.py
```

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: 700 chars, 150-char overlap
- Output: `chroma_db/`

---

### 4. Generate LoRA training data

```bash
python prepare_lora_json_llm.py --input raw_docs/rdr2_root_raw.jsonl
```

If interrupted, resume from last checkpoint:

```bash
python prepare_lora_json_llm.py --input raw_docs/rdr2_root_raw.jsonl --resume
```

- **Generator**: Qwen2.5-3B-Instruct, 3 QA pairs per page
- **Excerpt truncation**: 2600 chars max, cut at nearest sentence boundary
- **Document-type hint**: newspaper/letter/poem pages get an extra prompt
  instructing the model to name subjects explicitly and avoid generic questions
- **OOM handling**: out-of-memory pages are skipped and logged, run continues

---

### 5. Filter training data

```bash
python filter_lora_data.py --stats
```

Output: `lora_data/rdr2_lora_train_filtered.jsonl`

Filter rules (drop ~15%):

| Rule | Catches |
|------|---------|
| `BAD_QUESTION_EXACT` | "What is the name of the mission?" |
| `BAD_QUESTION_GENERIC` | "Who wrote the article?", "What is the main topic?" |
| `BAD_QUESTION_BARE_DOC` | Questions using "the article / poem / letter / excerpt" |
| `BAD_QUESTION_GOLD_MEDAL` | Questions mentioning "gold medal" conditions |
| `BAD_QUESTION_WHATIS` | "What is [Title]?" where answer is just a definition |
| `BAD_ANSWER_FRAGMENTS` | Answers starting with "not specified", "I don't know" |
| `PROTAGONIST_IN_ANSWER` | Answers containing "the protagonist / the player" anywhere |
| `EXCERPT_ARTIFACT` | Answers containing "excerpt" or "the page" |
| `is_fragment` | Answers under 8 words |
| `echoes_title` | Answer restates the question or equals the page title |
| `question_answer_mismatch` | "Where" question with no location in answer |

---

### 6. Train LoRA adapter

```bash
python train_lora_patched.py \
  --train-file lora_data/rdr2_lora_train_filtered.jsonl \
  --output-dir outputs/qwen25_rdr2_lora_v2
```

**Current adapter** (`qwen25_rdr2_lora_v2`):
- 476 steps, 1 epoch on full filtered dataset
- Final loss: ~0.64, token accuracy: ~87%

---

## Running Inference (CLI)

```bash
# Single mode
python infer_compare_all.py --mode lora_hybrid_rag --question "Where is Saint Denis?"

# All 6 modes
python infer_compare_all.py --mode all --question "Who is Arthur Morgan?"
```

---

## Running Web Interface

```bash
python -m uvicorn app:app --reload
```

Open: http://127.0.0.1:8000

---

## Evaluation

```bash
python evaluate_lora_vs_rag.py --questions eval_questions.json --output evaluation_results_v2.json
```

Runs all 6 modes across every question, prints per-mode answers + timing, saves JSON results and a summary table.

Run a subset of modes:

```bash
python evaluate_lora_vs_rag.py --questions eval_questions.json --modes base lora_hybrid_rag
```

**Key findings from 20-question eval** (`evaluation_results_v2.json`):

| Rank | Mode | Notes |
|------|------|-------|
| 🥇 | `lora_hybrid_rag` | Best overall; grounded, concise |
| 🥈 | `base_hybrid_rag` | Close second; retrieval dominates |
| 🥉 | `lora_rag` / `base_rag` | Good on factual questions |
| ❌ | `base` | Severe hallucinations |
| 💀 | `lora` | Hallucinates without RAG; LoRA alone needs more training |

RAG quality is the dominant factor. LoRA adds minor style improvements but does not replace retrieval for factual accuracy.

---

## Hybrid Retrieval

`retrieve_chunks_hybrid()` in `infer_compare_all.py` combines:

1. **Dense** — ChromaDB top-10 by cosine similarity
2. **BM25** — `rank_bm25` top-10 by term overlap
3. **RRF fusion** — Reciprocal Rank Fusion (k=60) merges both ranked lists
4. **Reranker** — cross-encoder re-scores merged pool, keeps top-3

RRF score: `Σ 1 / (k + rank_i)` for each retriever.

---

## Data Quality Notes

**Footer truncation fix** — `prepare_rag_json.py` and `prepare_lora_json_llm.py`
originally cut pages at any occurrence of `Gallery` / `References`. On wiki pages
these appear as mid-page section headings, reducing text coverage to ~16%.
Fix: apply cutoff only in the last 25% of the page → 93% coverage.

**Game title line fix** — Standalone `"Red Dead Redemption 2"` lines on wiki pages
are hyperlinks that complete a sentence on the previous line. Stripping them
unconditionally produces broken text like `"primary protagonist of."`.
Fix: preserve the line when the previous line ends without terminal punctuation.

**Excerpt truncation fix** — `_prose_excerpt()` previously returned the full
cleaned text regardless of `max_chars`, causing GPU OOM on long pages.
Fix: truncate at nearest sentence boundary within 2600 chars.

**Document-page QA fix** — Newspaper/letter/poem pages produced generic questions
("Who wrote the article?"). Fix: detect document pages by category/title keywords,
inject a prompt hint requiring subject names; filter side catches any remaining
bare-reference questions.
