import argparse
import json
import re
from pathlib import Path

BAD_PAGE_PATTERNS = [
    "hey there, cowboy",
    "this article looks a little bit small",
    "impress me by contributing",
    "don't be shy now",
]

DROP_LINE_PATTERNS = [
    r"^\[$",
    r"^\]$",
    r"^\[\s*\]$",
    r"^\d+(\.\d+)*$",
    r"^[vde•\s]+$",
]

FOOTER_MARKERS = {
    "Related Content",
    "Navigation",
    "Video Walkthrough",
    "References",
    "Gallery",
}

LIGHT_DROP_EXACT = {
    "Contents",
}

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_for_rag(text: str) -> str:
    text = normalize_text(text)
    low = text.lower()
    if any(p in low for p in BAD_PAGE_PATTERNS):
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []

    for line in lines:
        if any(re.match(rx, line) for rx in DROP_LINE_PATTERNS):
            continue
        if line in LIGHT_DROP_EXACT:
            continue
        if line in FOOTER_MARKERS:
            break
        cleaned.append(line)

    out = []
    prev = None
    for line in cleaned:
        if line != prev:
            out.append(line)
        prev = line

    return "\n".join(out).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_docs/rdr2_root_raw.jsonl")
    parser.add_argument("--output", default="raw_docs/rdr2_rag.jsonl")
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            if args.max_records is not None and idx > args.max_records:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            cleaned_text = clean_for_rag(record.get("text", ""))

            if len(cleaned_text) < args.min_chars:
                skipped += 1
                continue

            out_record = {
                "title": record.get("title"),
                "url": record.get("url"),
                "text": cleaned_text,
                "pageid": record.get("pageid"),
                "categories": record.get("categories", []),
                "source_category": record.get("source_category"),
                "depth_found": record.get("depth_found"),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            kept += 1

    print(json.dumps({
        "kept": kept,
        "skipped": skipped,
        "output": str(output_path),
    }, indent=2))

if __name__ == "__main__":
    main()
