
import argparse
import chromadb
import torch
from peft import PeftModel
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "outputs/qwen25_rdr2_lora"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rdr2"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_tokenizer = None
_model = None
_embedder = None
_reranker = None
_collection = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

def get_lora_model(adapter_path: str = ADAPTER_PATH):
    global _model
    if _model is None:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        _model = PeftModel.from_pretrained(base_model, adapter_path)
        _model.eval()
    return _model

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

def retrieve_chunks(question: str, top_k: int = 6):
    embedder = get_embedder()
    collection = get_collection()
    q_emb = embedder.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

def rerank_chunks(question: str, chunks, top_n: int = 3):
    reranker = get_reranker()
    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(
        [{"score": float(score), **chunk} for chunk, score in zip(chunks, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_n]

def generate_answer(user_prompt: str, adapter_path: str = ADAPTER_PATH, max_new_tokens: int = 256):
    tokenizer = get_tokenizer()
    model = get_lora_model(adapter_path)

    messages = [
        {"role": "system", "content": "You are a helpful Red Dead Redemption 2 assistant."},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def answer_lora_only(question: str, adapter_path: str = ADAPTER_PATH):
    return generate_answer(question, adapter_path=adapter_path)

def answer_lora_with_rag(
    question: str,
    adapter_path: str = ADAPTER_PATH,
    use_reranker: bool = True,
    top_k: int = 6,
    top_n: int = 3,
):
    chunks = retrieve_chunks(question, top_k=top_k)
    if use_reranker:
        chunks = rerank_chunks(question, chunks, top_n=top_n)
    else:
        chunks = chunks[:top_n]

    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        title = chunk["metadata"].get("title", "Unknown")
        context_blocks.append(f"[{idx}] {title}\n{chunk['text']}")

    grounded_prompt = (
        "Answer the question using the retrieved Red Dead Redemption 2 context below.\n"
        "If the context is insufficient, say so briefly.\n\n"
        f"Question: {question}\n\n"
        "Context:\n"
        + "\n\n".join(context_blocks)
    )
    return generate_answer(grounded_prompt, adapter_path=adapter_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--mode", choices=["lora", "lora_rag", "lora_rag_no_rerank"], default="lora")
    args = parser.parse_args()

    if args.mode == "lora":
        answer = answer_lora_only(args.question, adapter_path=args.adapter_path)
    elif args.mode == "lora_rag":
        answer = answer_lora_with_rag(args.question, adapter_path=args.adapter_path, use_reranker=True)
    else:
        answer = answer_lora_with_rag(args.question, adapter_path=args.adapter_path, use_reranker=False)

    print("\nANSWER:\n")
    print(answer)

if __name__ == "__main__":
    main()
