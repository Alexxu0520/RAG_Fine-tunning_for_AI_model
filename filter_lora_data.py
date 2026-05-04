import argparse
import json
import re
from pathlib import Path

# ── Bad question patterns ──────────────────────────────────────────────────────

BAD_QUESTION_EXACT = re.compile(
    r"^what is the name of the (mission|article|item|page|newspaper|poem|letter|book)\??$",
    re.IGNORECASE,
)

BAD_QUESTION_WHATIS = re.compile(
    r"^what is ['\"]?[A-Z][^?]{0,60}\??$"
)

BAD_QUESTION_GENERIC = re.compile(
    r"^(who wrote the (article|letter|newspaper|poem)"
    r"|what is the main topic of the (article|page)"
    r"|what is the (article|page|item) about"
    r"|who is the main character (in|of) (this|the) (mission|article|page)"
    r"|what is the (main )?purpose of (this|the)"
    r")\??$",
    re.IGNORECASE,
)

# ── Bad answer patterns ────────────────────────────────────────────────────────

BAD_ANSWER_FRAGMENTS = re.compile(
    r"^(the protagonist|the player|the main character|"
    r"not specified|not mentioned|not present|"
    r"i (don'?t|do not) know|it is unknown|"
    r"no information|unclear|n/a)\b",
    re.IGNORECASE,
)

# Answer is suspiciously short (fewer than 5 words and no period — likely a fragment)
def is_fragment(answer: str) -> bool:
    words = answer.split()
    return len(words) < 5 and not answer.endswith(".")

# ── Question–answer alignment ──────────────────────────────────────────────────

LOCATION_WORDS = re.compile(
    r"\b(in|at|near|north|south|east|west|located|ranch|town|"
    r"camp|valley|river|ridge|bayou|creek|fort|station|lake|region|territory)\b",
    re.IGNORECASE,
)

PERSON_WORDS = re.compile(
    r"\b(he|she|they|his|her|their|mr|mrs|named|known as|called)\b",
    re.IGNORECASE,
)

def question_answer_mismatch(question: str, answer: str) -> bool:
    q = question.lower()
    if q.startswith("where") and not LOCATION_WORDS.search(answer):
        return True
    return False

# ── Title-echo check ──────────────────────────────────────────────────────────

def echoes_title(question: str, answer: str, title: str) -> bool:
    # Answer is just the title
    if answer.strip().lower() == title.strip().lower():
        return True
    # Answer just restates the question (circular)
    q_core = re.sub(r"[^a-z0-9 ]", "", question.lower())
    a_core = re.sub(r"[^a-z0-9 ]", "", answer.lower())
    if q_core and q_core in a_core and len(a_core) < len(q_core) + 20:
        return True
    return False

# ── Main filter ───────────────────────────────────────────────────────────────

def should_drop(question: str, answer: str, title: str) -> tuple[bool, str]:
    if BAD_QUESTION_EXACT.match(question.strip()):
        return True, "template_question_name"
    if BAD_QUESTION_GENERIC.match(question.strip()):
        return True, "template_question_generic"
    if BAD_QUESTION_WHATIS.match(question.strip()):
        return True, "template_question_whatis"
    if BAD_ANSWER_FRAGMENTS.match(answer.strip()):
        return True, "bad_answer_fragment_or_uncertain"
    if is_fragment(answer):
        return True, "answer_too_short"
    if echoes_title(question, answer, title):
        return True, "answer_echoes_title_or_question"
    if question_answer_mismatch(question, answer):
        return True, "question_answer_mismatch"
    return False, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="lora_data/rdr2_lora_train_qa.jsonl")
    parser.add_argument("--output", default="lora_data/rdr2_lora_train_filtered.jsonl")
    parser.add_argument("--stats", action="store_true", help="Print drop reason breakdown")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    reasons: dict[str, int] = {}

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            messages = record.get("messages", [])
            title = record.get("metadata", {}).get("title", "")

            question = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            answer = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )

            drop, reason = should_drop(question, answer, title)
            if drop:
                dropped += 1
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                fout.write(line + "\n")
                kept += 1

    print(json.dumps({
        "kept": kept,
        "dropped": dropped,
        "drop_rate": f"{dropped / (kept + dropped) * 100:.1f}%",
        "output": str(output_path),
    }, indent=2))

    if args.stats:
        print("\nDrop reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
