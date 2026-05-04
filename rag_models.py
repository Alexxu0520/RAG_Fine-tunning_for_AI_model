from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import chromadb
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rdr2"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_NAME = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float | None = None


_generation_model = None
_generation_tokenizer = None
_embed_model = None
_reranker_model = None
_bm25_index: BM25Okapi | None = None
_bm25_chunks: List[RetrievedChunk] = []


def get_tokenizer():
    global _generation_tokenizer
    if _generation_tokenizer is None:
        _generation_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    return _generation_tokenizer



def get_generation_model():
    global _generation_model
    if _generation_model is None:
        _generation_model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
    return _generation_model



def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model



def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_NAME)
    return _reranker_model



def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def get_bm25_index() -> Tuple[BM25Okapi, List[RetrievedChunk]]:
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    docs = result["documents"]
    metas = result["metadatas"]

    _bm25_chunks = [
        RetrievedChunk(text=doc, metadata=meta or {})
        for doc, meta in zip(docs, metas)
    ]
    tokenized = [_tokenize(c.text) for c in _bm25_chunks]
    _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_chunks


def _reciprocal_rank_fusion(
    rankings: List[List[int]], k: int = 60
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve_chunks_hybrid(
    question: str,
    n_dense: int = 10,
    n_bm25: int = 10,
    top_n: int = 6,
) -> List[RetrievedChunk]:
    # Dense retrieval
    collection = get_collection()
    embed_model = get_embed_model()
    query_embedding = embed_model.encode([question]).tolist()[0]
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_dense,
        include=["documents", "metadatas", "distances"],
    )
    dense_docs = dense_results["documents"][0]
    dense_metas = dense_results["metadatas"][0]

    # BM25 retrieval
    bm25, all_chunks = get_bm25_index()
    bm25_scores = bm25.get_scores(_tokenize(question))
    bm25_top_indices = bm25_scores.argsort()[::-1][:n_bm25].tolist()

    # Build a unified doc pool keyed by (title, text hash)
    pool: Dict[str, RetrievedChunk] = {}

    for doc, meta in zip(dense_docs, dense_metas):
        key = doc[:120]
        pool[key] = RetrievedChunk(text=doc, metadata=meta or {})

    for idx in bm25_top_indices:
        chunk = all_chunks[idx]
        key = chunk.text[:120]
        pool.setdefault(key, chunk)

    pool_keys = list(pool.keys())

    dense_ranking = [pool_keys.index(doc[:120]) for doc in dense_docs if doc[:120] in pool_keys]
    bm25_ranking = [pool_keys.index(all_chunks[i].text[:120]) for i in bm25_top_indices if all_chunks[i].text[:120] in pool_keys]

    fused = _reciprocal_rank_fusion([dense_ranking, bm25_ranking])
    top_chunks = [pool[pool_keys[idx]] for idx, _ in fused[:top_n]]
    return top_chunks


def answer_hybrid_reranker(question: str, n_dense: int = 10, n_bm25: int = 10, top_n: int = 2) -> Dict[str, Any]:
    hybrid_chunks = retrieve_chunks_hybrid(question, n_dense=n_dense, n_bm25=n_bm25, top_n=6)
    reranked = rerank_chunks(question, hybrid_chunks, top_n=top_n)
    answer = generate_with_context(question, reranked)
    return {
        "model": "hybrid_reranker",
        "question": question,
        "answer": answer,
        "chunks": [
            {"metadata": c.metadata, "score": c.score, "text": c.text}
            for c in reranked
        ],
    }


def ask_qwen(question: str, system_prompt: str | None = None, max_new_tokens: int = 220) -> str:
    tokenizer = get_tokenizer()
    model = get_generation_model()

    messages = [
        {
            "role": "system",
            "content": system_prompt or "You are a helpful game assistant for Red Dead Redemption 2.",
        },
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]



def retrieve_chunks(question: str, n_results: int = 4) -> List[RetrievedChunk]:
    collection = get_collection()
    embed_model = get_embed_model()
    query_embedding = embed_model.encode([question]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    chunks: List[RetrievedChunk] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        dist = distances[i] if i < len(distances) else None
        score = None if dist is None else float(-dist)
        chunks.append(RetrievedChunk(text=doc, metadata=meta or {}, score=score))
    return chunks



def rerank_chunks(question: str, chunks: Sequence[RetrievedChunk], top_n: int = 2) -> List[RetrievedChunk]:
    if not chunks:
        return []
    reranker = get_reranker_model()
    pairs = [[question, c.text] for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(
        [RetrievedChunk(text=c.text, metadata=c.metadata, score=float(s)) for c, s in zip(chunks, scores)],
        key=lambda c: c.score if c.score is not None else float("-inf"),
        reverse=True,
    )
    return ranked[:top_n]



def generate_with_context(question: str, chunks: Sequence[RetrievedChunk], max_new_tokens: int = 220) -> str:
    context = "\n\n".join(chunk.text for chunk in chunks)
    system_prompt = (
        "You are a helpful Red Dead Redemption 2 assistant. "
        "Answer using the provided context when possible. "
        "If the answer is not in the context, say you are not sure."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return ask_qwen(user_prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens)



def answer_qwen_only(question: str) -> Dict[str, Any]:
    answer = ask_qwen(question)
    return {"model": "qwen_only", "question": question, "answer": answer, "chunks": []}



def answer_retriever(question: str, n_results: int = 4) -> Dict[str, Any]:
    chunks = retrieve_chunks(question, n_results=n_results)
    answer = generate_with_context(question, chunks)
    return {
        "model": "rag_retriever",
        "question": question,
        "answer": answer,
        "chunks": [
            {"metadata": c.metadata, "score": c.score, "text": c.text}
            for c in chunks
        ],
    }



def answer_reranker(question: str, n_results: int = 6, top_n: int = 2) -> Dict[str, Any]:
    retrieved = retrieve_chunks(question, n_results=n_results)
    reranked = rerank_chunks(question, retrieved, top_n=top_n)
    answer = generate_with_context(question, reranked)
    return {
        "model": "rag_reranker",
        "question": question,
        "answer": answer,
        "chunks": [
            {"metadata": c.metadata, "score": c.score, "text": c.text}
            for c in reranked
        ],
        "initial_retrieval_count": len(retrieved),
    }
