import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rdr2"

# retriever embedding model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# reranker model
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# generator model
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def retrieve_candidates(question: str, n_results: int = 5):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    query_embedding = embed_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas


def rerank_candidates(question: str, docs, metas, top_n: int = 2):
    reranker = CrossEncoder(RERANKER_NAME)

    pairs = [[question, doc] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, metas, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top_ranked = ranked[:top_n]
    top_docs = [x[0] for x in top_ranked]
    top_metas = [x[1] for x in top_ranked]
    top_scores = [float(x[2]) for x in top_ranked]

    return top_docs, top_metas, top_scores


def generate_answer(question: str, context: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Red Dead Redemption 2 assistant. "
                "Answer only from the provided context. "
                "If the answer is not in the context, say you are not sure."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=180
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response


if __name__ == "__main__":
    question = input("Ask a question about RDR2: ").strip()

    # Step 1: retrieve top-k candidates
    docs, metas = retrieve_candidates(question, n_results=5)

    print("\n=== Initial Retrieved Candidates ===")
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        print(f"\n--- Candidate {i} ---")
        print(meta)
        print(doc[:500], "...\n")

    # Step 2: rerank top-k candidates
    top_docs, top_metas, top_scores = rerank_candidates(question, docs, metas, top_n=2)

    print("\n=== Reranked Top Chunks ===")
    for i, (doc, meta, score) in enumerate(zip(top_docs, top_metas, top_scores), start=1):
        print(f"\n--- Top {i} | score={score:.4f} ---")
        print(meta)
        print(doc[:700], "...\n")

    # Step 3: send best chunks to Qwen
    context = "\n\n".join(top_docs)
    answer = generate_answer(question, context)

    print("\n=== Model Answer ===")
    print(answer)