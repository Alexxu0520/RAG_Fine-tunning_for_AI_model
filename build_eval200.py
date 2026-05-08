"""
Sample 180 diverse QA pairs from rdr2_lora_train_filtered.jsonl,
combine with existing 20 hand-written ground truth answers,
produce eval_ground_truth_200.json (200 entries).
"""
import json, random, re
from pathlib import Path
from collections import defaultdict

random.seed(42)

EXISTING_FILE = "eval_ground_truth.json"
TRAIN_FILE    = "lora_data/rdr2_lora_train_filtered.jsonl"
OUT_FILE      = "eval_ground_truth_200.json"
TARGET_NEW    = 180

# ── Load existing 20 ──────────────────────────────────────────────────────────
with open(EXISTING_FILE) as f:
    existing = json.load(f)
existing_questions = {e["question"].lower().strip() for e in existing}

# ── Load training data ────────────────────────────────────────────────────────
all_records = []
with open(TRAIN_FILE) as f:
    for line in f:
        d = json.loads(line)
        msgs = d["messages"]
        q = next((m["content"] for m in msgs if m["role"] == "user"), None)
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not q or not a:
            continue
        # Skip if question is in existing 20
        if q.lower().strip() in existing_questions:
            continue
        # Skip very short answers (< 15 words)
        if len(a.split()) < 15:
            continue
        # Skip answers that still contain artifact phrases
        if re.search(r'\b(the excerpt|the article|the poem|the letter|protagonist of\.|player character)\b', a, re.I):
            continue
        all_records.append({
            "question": q,
            "answer":   a,
            "category": d.get("metadata", {}).get("category_guess", "unknown"),
            "title":    d.get("metadata", {}).get("title", ""),
        })

print(f"Pool size after filtering: {len(all_records)}")

# ── Stratified sampling by category ──────────────────────────────────────────
by_cat = defaultdict(list)
for r in all_records:
    by_cat[r["category"]].append(r)

cat_counts = {k: len(v) for k, v in by_cat.items()}
print("Category distribution:", dict(sorted(cat_counts.items(), key=lambda x: -x[1])))

# Allocate slots proportionally, capped per category
total_pool = len(all_records)
allocated  = {}
for cat, recs in by_cat.items():
    share = round(TARGET_NEW * len(recs) / total_pool)
    allocated[cat] = max(1, min(share, len(recs)))

# Adjust to hit exactly TARGET_NEW
diff = TARGET_NEW - sum(allocated.values())
cats_sorted = sorted(allocated, key=lambda c: -len(by_cat[c]))
for i in range(abs(diff)):
    cat = cats_sorted[i % len(cats_sorted)]
    allocated[cat] += 1 if diff > 0 else -1
    allocated[cat] = max(0, allocated[cat])

print("Allocated per category:", dict(sorted(allocated.items(), key=lambda x: -x[1])))

sampled = []
for cat, n in allocated.items():
    pool = by_cat[cat]
    random.shuffle(pool)
    sampled.extend(pool[:n])

random.shuffle(sampled)
print(f"Sampled: {len(sampled)}")

# ── Combine ───────────────────────────────────────────────────────────────────
combined = existing + [{"question": r["question"], "answer": r["answer"]}
                       for r in sampled]

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"\nWrote {len(combined)} entries to {OUT_FILE}")
print(f"  Original 20 hand-written : {len(existing)}")
print(f"  New from training data   : {len(sampled)}")
print("\nNOTE: 180 new questions overlap with LoRA training set.")
print("BERTScore for LoRA modes on these may be slightly inflated.")
print("Interpret as upper-bound estimate for LoRA, fair comparison for base vs RAG.")
