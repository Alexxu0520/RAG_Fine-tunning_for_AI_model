"""Compute BERTScore for all 6 modes against hand-written ground truth."""
import json
from bert_score import score as bs_score

with open("eval_ground_truth.json") as f:
    gt = json.load(f)
with open("evaluation_results_v2.json") as f:
    results = json.load(f)

# Build question -> ground truth mapping
gt_map = {d["question"]: d["answer"] for d in gt}

modes = ["base", "base_rag", "base_hybrid_rag", "lora", "lora_rag", "lora_hybrid_rag"]

print("Computing BERTScore (model: distilbert-base-uncased)...\n")

mode_scores = {}
for mode in modes:
    preds = []
    refs  = []
    for r in results:
        q = r["question"]
        if q in gt_map and mode in r:
            preds.append(r[mode]["answer"])
            refs.append(gt_map[q])

    P, R, F1 = bs_score(preds, refs,
                         model_type="distilbert-base-uncased",
                         lang="en", verbose=False)
    mode_scores[mode] = {
        "P":  float(P.mean()),
        "R":  float(R.mean()),
        "F1": float(F1.mean()),
    }

print(f"{'Mode':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 54)
for mode in modes:
    s = mode_scores[mode]
    print(f"{mode:<22} {s['P']:>10.4f} {s['R']:>10.4f} {s['F1']:>10.4f}")

with open("bertscore_results.json", "w") as f:
    json.dump(mode_scores, f, indent=2)

print("\nSaved to bertscore_results.json")
