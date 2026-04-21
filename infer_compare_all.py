import argparse
import re
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
_base_model = None
_lora_model = None
_embedder = None
_reranker = None
_collection = None

def strip_source_page(text: str) -> str:
    text = re.sub(r"\n\s*Source page:\s*https?://\S+\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Source page:\s*https?://\S+", "", text, flags=re.IGNORECASE)
    return text.strip()

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

def _make_quantized_base():
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model

def get_base_model():
    global _base_model
    if _base_model is None:
        _base_model = _make_quantized_base()
    return _base_model

def get_lora_model(adapter_path: str = ADAPTER_PATH):
    global _lora_model
    if _lora_model is None:
        base_model = _make_quantized_base()
        _lora_model = PeftModel.from_pretrained(base_model, adapter_path)
        _lora_model.eval()
    return _lora_model

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

def generate_answer(user_prompt: str, model, max_new_tokens: int = 256):
    tokenizer = get_tokenizer()
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

def build_grounded_prompt(question: str, chunks):
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        title = chunk["metadata"].get("title", "Unknown")
        context_blocks.append(f"[{idx}] {title}\n{chunk['text']}")
    return (
        "Answer the question using the retrieved Red Dead Redemption 2 context below. "
        "Do not include raw URLs or a 'Source page' line in the answer. "
        "If the context is insufficient, say so briefly.\n\n"
        f"Question: {question}\n\n"
        "Context:\n" + "\n\n".join(context_blocks)
    )

def answer_base_only(question: str):
    model = get_base_model()
    return strip_source_page(generate_answer(question, model))

def answer_lora_only(question: str, adapter_path: str = ADAPTER_PATH):
    model = get_lora_model(adapter_path)
    return strip_source_page(generate_answer(question, model))

def answer_base_with_rag(question: str, use_reranker: bool = True, top_k: int = 6, top_n: int = 3):
    model = get_base_model()
    chunks = retrieve_chunks(question, top_k=top_k)
    if use_reranker:
        chunks = rerank_chunks(question, chunks, top_n=top_n)
    else:
        chunks = chunks[:top_n]
    grounded_prompt = build_grounded_prompt(question, chunks)
    return strip_source_page(generate_answer(grounded_prompt, model))

def answer_lora_with_rag(
    question: str,
    adapter_path: str = ADAPTER_PATH,
    use_reranker: bool = True,
    top_k: int = 6,
    top_n: int = 3,
):
    model = get_lora_model(adapter_path)
    chunks = retrieve_chunks(question, top_k=top_k)
    if use_reranker:
        chunks = rerank_chunks(question, chunks, top_n=top_n)
    else:
        chunks = chunks[:top_n]
    grounded_prompt = build_grounded_prompt(question, chunks)
    return strip_source_page(generate_answer(grounded_prompt, model))

def run_all(question: str, adapter_path: str):
    results = []
    results.append(("base", answer_base_only(question)))
    results.append(("base_rag", answer_base_with_rag(question, use_reranker=True)))
    results.append(("lora", answer_lora_only(question, adapter_path=adapter_path)))
    results.append(("lora_rag", answer_lora_with_rag(question, adapter_path=adapter_path, use_reranker=True)))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument(
        "--mode",
        choices=["base", "base_rag", "base_rag_no_rerank", "lora", "lora_rag", "lora_rag_no_rerank", "all"],
        default="lora",
    )
    args = parser.parse_args()

    if args.mode == "base":
        answer = answer_base_only(args.question)
        print("\nANSWER:\n")
        print(answer)
    elif args.mode == "base_rag":
        answer = answer_base_with_rag(args.question, use_reranker=True)
        print("\nANSWER:\n")
        print(answer)
    elif args.mode == "base_rag_no_rerank":
        answer = answer_base_with_rag(args.question, use_reranker=False)
        print("\nANSWER:\n")
        print(answer)
    elif args.mode == "lora":
        answer = answer_lora_only(args.question, adapter_path=args.adapter_path)
        print("\nANSWER:\n")
        print(answer)
    elif args.mode == "lora_rag":
        answer = answer_lora_with_rag(args.question, adapter_path=args.adapter_path, use_reranker=True)
        print("\nANSWER:\n")
        print(answer)
    elif args.mode == "lora_rag_no_rerank":
        answer = answer_lora_with_rag(args.question, adapter_path=args.adapter_path, use_reranker=False)
        print("\nANSWER:\n")
        print(answer)
    else:
        results = run_all(args.question, args.adapter_path)
        print("\nCOMPARISON RESULTS:\n")
        for mode_name, answer in results:
            print(f"===== {mode_name} =====")
            print(answer)
            print()

if __name__ == "__main__":
    main()
