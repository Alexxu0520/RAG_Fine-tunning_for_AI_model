import argparse
import json
import time
from pathlib import Path

from infer_compare_all import (
    ADAPTER_PATH,
    answer_base_only,
    answer_base_with_rag,
    answer_base_with_hybrid_rag,
    answer_lora_only,
    answer_lora_with_rag,
    answer_lora_with_hybrid_rag,
)

MODES = [
    "base",
    "base_rag",
    "base_hybrid_rag",
    "lora",
    "lora_rag",
    "lora_hybrid_rag",
]


def run_mode(mode: str, question: str, adapter_path: str) -> tuple[str, float]:
    t0 = time.time()
    if mode == "base":
        answer = answer_base_only(question)
    elif mode == "base_rag":
        answer = answer_base_with_rag(question, use_reranker=True)
    elif mode == "base_hybrid_rag":
        answer = answer_base_with_hybrid_rag(question)
    elif mode == "lora":
        answer = answer_lora_only(question, adapter_path=adapter_path)
    elif mode == "lora_rag":
        answer = answer_lora_with_rag(question, adapter_path=adapter_path, use_reranker=True)
    elif mode == "lora_hybrid_rag":
        answer = answer_lora_with_hybrid_rag(question, adapter_path=adapter_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return answer, round(time.time() - t0, 1)


def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError("Question file must be a list or a dict with a 'questions' key.")


def print_summary(results: list):
    print("\n" + "=" * 80)
    print("SUMMARY — answer length (chars) per mode")
    print("=" * 80)

    col_w = 18
    header = f"{'question':<30}" + "".join(f"{m:>{col_w}}" for m in MODES)
    print(header)
    print("-" * len(header))

    for row in results:
        q_short = row["question"][:28] + ".." if len(row["question"]) > 30 else row["question"]
        line = f"{q_short:<30}"
        for mode in MODES:
            val = row.get(mode, {})
            chars = len(val.get("answer", "")) if isinstance(val, dict) else 0
            line += f"{chars:>{col_w}}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
        help="Subset of modes to run (default: all)",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    results = []

    for idx, question in enumerate(questions, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(questions)}] {question}")
        print("=" * 80)

        row = {"question": question}
        for mode in args.modes:
            try:
                answer, elapsed = run_mode(mode, question, args.adapter_path)
            except Exception as e:
                answer, elapsed = f"ERROR: {e}", 0.0
            row[mode] = {"answer": answer, "seconds": elapsed}
            print(f"\n--- {mode} ({elapsed}s) ---\n{answer}")

        results.append(row)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
