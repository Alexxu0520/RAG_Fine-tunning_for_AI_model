
import argparse
import json
from pathlib import Path

from infer_lora import answer_lora_only, answer_lora_with_rag
from rag_models import answer_qwen_only, answer_rag_retriever, answer_rag_reranker

def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError("Question file must be a list or a dict with a 'questions' key.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--output", default="evaluation_lora_vs_rag.json")
    parser.add_argument("--adapter-path", default="outputs/qwen25_rdr2_lora")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    results = []

    for idx, question in enumerate(questions, start=1):
        print("=" * 80)
        print(f"Question {idx}/{len(questions)}: {question}")
        print("=" * 80)

        row = {
            "question": question,
            "qwen_only": answer_qwen_only(question),
            "rag_retriever": answer_rag_retriever(question),
            "rag_reranker": answer_rag_reranker(question),
            "lora_only": answer_lora_only(question, adapter_path=args.adapter_path),
            "lora_rag": answer_lora_with_rag(question, adapter_path=args.adapter_path, use_reranker=False),
            "lora_rag_reranker": answer_lora_with_rag(question, adapter_path=args.adapter_path, use_reranker=True),
        }
        results.append(row)

        for key in ["qwen_only", "rag_retriever", "rag_reranker", "lora_only", "lora_rag", "lora_rag_reranker"]:
            print(f"\n--- {key} ---\n{row[key]}\n")

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved evaluation results to: {output_path}")

if __name__ == "__main__":
    main()
