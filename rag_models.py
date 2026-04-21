from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import chromadb
import torch
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
