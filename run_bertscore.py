"""
Compute BERTScore for all 6 modes against reference answers.

Usage:
  # 20-question set (fast, hand-written refs):
  python run_bertscore.py

  # 200-question set (requires evaluation_results_200.json first):
  python run_bertscore.py --eval-results evaluation_results_200.json \
                          --ground-truth eval_ground_truth_200.json \
                          --output bertscore_results_200.json

Generate evaluation_results_200.json first:
  python evaluate_lora_vs_rag.py \
      --questions eval_questions_200.json \
      --output evaluation_results_200.json
"""
import json
import argparse
from bert_score import score as bs_score

parser = argparse.ArgumentParser()
parser.add_argument("--eval-results", default="evaluation_results_v2.json")
parser.add_argument("--ground-truth",  default="eval_ground_truth.json")
parser.add_argument("--output",        default="bertscore_results.json")
parser.add_argument("--model",         default="distilbert-base-uncased")
args = parser.parse_args()

with open(args.ground_truth, encoding="utf-8") as f:
    gt = json.load(f)
with open(args.eval_results, encoding="utf-8") as f:
    results = json.load(f)

gt_map = {d["question"]: d["answer"] for d in gt}
modes  = ["base", "base_rag", "base_hybrid_rag", "lora", "lora_rag", "lora_hybrid_rag"]

print(f"Ground truth : {args.ground_truth} ({len(gt_map)} questions)")
print(f"Eval results : {args.eval_results} ({len(results)} questions)")
print(f"BERTScore model: {args.model}\n")

mode_scores = {}
for mode in modes:
    preds, refs = [], []
    for r in results:
        q = r["question"]
        if q in gt_map and mode in r:
            preds.append(r[mode]["answer"])
            refs.append(gt_map[q])

    if not preds:
        print(f"  {mode}: no matching questions, skipping")
        continue

    P, R, F1 = bs_score(preds, refs, model_type=args.model,
                         lang="en", verbose=False)
    mode_scores[mode] = {"P": float(P.mean()), "R": float(R.mean()), "F1": float(F1.mean()), "n": len(preds)}

print(f"\n{'Mode':<22} {'n':>4} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)
for mode in modes:
    if mode not in mode_scores:
        continue
    s = mode_scores[mode]
    print(f"{mode:<22} {s['n']:>4} {s['P']:>10.4f} {s['R']:>10.4f} {s['F1']:>10.4f}")

with open(args.output, "w") as f:
    json.dump(mode_scores, f, indent=2)
print(f"\nSaved to {args.output}")

if "200" in args.ground_truth:
    print("\nNOTE: 180/200 questions overlap with the LoRA training set.")
    print("BERTScore for lora/lora_rag/lora_hybrid_rag is an upper-bound estimate.")
    print("base and base_rag/hybrid scores are unaffected.")
