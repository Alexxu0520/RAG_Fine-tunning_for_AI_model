import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rdr2"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def retrieve_context(question: str, n_results: int = 1):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas

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
                "make sure go through all docs"
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
        max_new_tokens=150
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

    docs, metas = retrieve_context(question, n_results=1)
    context = "\n\n".join(docs)

    print("\n=== Retrieved Context ===")
    print(context)

    print("\n=== Sources ===")
    for meta in metas:
        print(meta)

    answer = generate_answer(question, context)

    print("\n=== Model Answer ===")
    print(answer)