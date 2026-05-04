import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """You are a grizzled frontier storyteller who knows Red Dead Redemption 2 inside and out.
Your job is to prepare high-quality supervised fine-tuning data for a Red Dead Redemption 2 assistant.

Given a wiki-derived page, do these things:
1. Infer the page type.
2. Write 3 natural user questions about the page.
3. Write a separate grounded answer for each question.

Return ONLY valid JSON with this schema:
{
  "page_type": "person|mission|achievement|collectible|system|location|thing",
  "qa_pairs": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}

Voice and style:
- Write in a plain-spoken, direct Western voice. Avoid formal or academic language.
- Avoid words like "objective", "protagonist", "mechanic", "excerpt", "article".
- Replace "main objective" with "what you need to do". Use character names instead of "protagonist".

Question rules:
- The 3 questions must use different question words. Use a mix of Who / What / Where / When / How / Why.
- Do not start more than one question with the same word.
- Do not ask "What is the name of the mission/item/article/page?"
- Do not write a question whose answer is just the page title.
- Do not write questions of the form "What is [title]?" where the answer is just "[title] is a [type]."
- Do not ask questions that can be answered without reading the excerpt.
- Each question must require reading the excerpt to answer.
- Questions must ask about specific facts: events, locations, characters, or outcomes.

Page-type specific guidance:
- For missions: ask about plot events, characters involved, and outcomes. Do NOT ask about gold medal conditions, time limits, or headshot requirements.
- For persons: ask about their role, affiliations, and fate.
- For locations: ask about where it is and what can be found or done there.
- For items/things: ask about purpose, where to find it, or how to use it.

Answer rules:
- Base everything only on the provided title, categories, and excerpt. Do not invent facts.
- Each question must have its own matching answer.
- Answers must be complete sentences, not fragments or single words.
- Answers must be short: 1 to 2 sentences max.
- If a question asks "where", the answer must contain location information.
- If a question asks "who", the answer must clearly identify the person.
- If the question asks what happens after X, the answer must describe what comes after X, not repeat X.
- Do not write an answer that simply restates the question.
- Do not copy raw wiki formatting or template junk.
- Do not include section headers like "Mission Appearances", "Gallery", "References", or "Related Content".
"""

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
    r"^(Contents?)$",
    r"^(History|Overview|Description|Walkthrough|Story|Background|Interactions)$",
    r"^(Trivia|Quotes|Video|Text|Content)$",
    r"^(Mission appearances?|Missable Items in the Mission|Trophies/Achievements)$",
    r"^[vde•\s]+$",
]

INFOBOX_LABELS = {
    "Gameplay", "Game", "Biography", "Gender", "Occupation", "Informations",
    "Newspaper", "Rewards", "Start", "End", "Given by", "Location",
    "Protagonist(s)", "Unlocked by", "Unlocks", "Storyline", "Characters",
    "Information", "Progression", "Additional info", "Family", "Affiliations",
    "Nationality", "Actor", "Mount", "Weapon", "Statistics", "Type", "Item",
    "Collecting for", "Total", "Caption", "Notes", "Tips", "Quick Answers",
    "Provided by: Fandom",
}

ALLOWED_PAGE_TYPES = {"person", "mission", "achievement", "collectible", "system", "location", "thing"}

FOOTER_MARKERS = {"Related Content", "Navigation", "Video Walkthrough", "References", "Gallery"}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    low = text.lower()
    if any(p in low for p in BAD_PAGE_PATTERNS):
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: List[str] = []

    skip_quick_answers = False
    for line in lines:
        if line == "Quick Answers":
            skip_quick_answers = True
            continue
        if skip_quick_answers:
            if line in {"History", "Character", "Personality", "Appearance", "Skills", "Relationships"}:
                skip_quick_answers = False
            else:
                continue

        if line in INFOBOX_LABELS:
            continue
        if any(re.match(rx, line) for rx in DROP_LINE_PATTERNS):
            continue
        # Only drop game-title lines that stand alone as a header (first line or
        # after a blank), not when they complete a sentence on the previous line.
        if line.lower() == "provided by: fandom":
            continue
        if line.lower() in {"red dead redemption 2", "red dead online", "red dead revolver"}:
            prev = cleaned[-1] if cleaned else ""
            if prev and prev[-1] not in ".!?,;:":
                cleaned.append(line)
            continue
        if line.startswith("What ") and line.endswith("?"):
            continue
        cleaned.append(line)

    cutoff = max(1, int(len(cleaned) * 0.75))
    trimmed: List[str] = []
    for i, line in enumerate(cleaned):
        if line in FOOTER_MARKERS and i >= cutoff:
            break
        trimmed.append(line)

    text = "\n".join(trimmed)
    text = re.sub(r"\bin\s+\.", "", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def fallback_page_type(title: str, categories: List[str], text: str) -> str:
    low_title = title.lower()
    cat_low = " | ".join(categories).lower()
    low = f"{title}\n{text[:1800]}".lower()

    if any(k in cat_low for k in ["playable_characters", "characters_in_redemption", "special_characters"]):
        return "person"
    if any(k in cat_low for k in ["stranger_missions", "missions", "redemption_ii_missions"]):
        return "mission"
    if any(k in cat_low for k in ["collectible"]):
        return "collectible"
    if any(k in cat_low for k in ["features", "gameplay", "activities", "cheats", "crafting", "terminology"]):
        return "system"
    if any(k in cat_low for k in ["outfits", "clothing", "weapons", "items", "documents", "letters", "newspaper_articles"]):
        return "thing"

    if "list of " in low_title or low_title.startswith("list "):
        return "thing"
    if any(x in low_title for x in ["easter eggs", "secrets", "missable content", "ending credits"]):
        return "thing"
    if any(x in low_title for x in ["honor", "core", "compendium", "journal", "heads-up display", "eagle eye", "money", "fishing", "hanging"]):
        return "system"
    if any(x in low_title for x in ["card set", "cigarette cards", "dinosaur bones", "dreamcatchers"]):
        return "collectible"
    if any(x in low_title for x in ["completion", "gold medal"]):
        return "achievement"
    if any(x in low_title for x in ["mansion", "camp", "station", "river", "fort", "ranch", "town", "lake"]):
        return "location"
    if any(x in low_title for x in ["recipes", "skull cap", "hat", "outfit"]):
        return "thing"

    if any(x in low for x in ["central character", "primary protagonist", "secondary antagonist", "gang member", "gunslinger", "minor character", "bounty target"]):
        return "person"
    if any(x in low for x in ["story mission", "stranger mission", "side mission", "main mission", "moonshiner story mission"]):
        return "mission"
    if any(x in low for x in ["achievement/trophy", "is an accomplishment", "100% completion"]):
        return "achievement"
    if any(x in low for x in ["is a feature", "are a feature", "is an activity", "is a skill", "is a gameplay mechanic"]):
        return "system"
    if any(x in low for x in ["town", "settlement", "ranch", "river", "camp", "fort", "located in"]):
        return "location"

    return "thing"


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def clean_generated_answer(a: str) -> str:
    a = normalize_whitespace(a)
    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]

    cleaned = []
    for line in lines:
        low = line.lower()
        if low in {
            "mission appearances", "part i", "part ii", "part iii",
            "deaths", "gallery", "references", "related content",
            "video walkthrough", "tracking progress"
        }:
            continue
        cleaned.append(line)

    a = " ".join(cleaned)
    a = re.sub(r"\s+", " ", a).strip()
    return a


def validate_generation(data: Dict[str, Any], title: str, categories: List[str], cleaned_text: str) -> Dict[str, Any]:
    page_type = str(data.get("page_type", "")).strip().lower()
    if page_type not in ALLOWED_PAGE_TYPES:
        page_type = fallback_page_type(title, categories, cleaned_text)

    qa_pairs = data.get("qa_pairs", [])
    if not isinstance(qa_pairs, list):
        qa_pairs = []

    good_pairs = []

    for pair in qa_pairs:
        if not isinstance(pair, dict):
            continue

        q = str(pair.get("question", "")).strip()
        a = str(pair.get("answer", "")).strip()
        a = clean_generated_answer(a)


        good_pairs.append({
            "question": q,
            "answer": a,
        })

    return {
        "page_type": page_type,
        "qa_pairs": good_pairs,
    }


def _prose_excerpt(text: str, max_chars: int = 2600) -> str:
    return text.strip()


class LLMGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not found. This script expects an NVIDIA GPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            device_map="auto",
        )
        self.model.eval()

    @torch.inference_mode()
    def generate_structured(self, title: str, categories: List[str], cleaned_text: str) -> Dict[str, Any]:
        excerpt = _prose_excerpt(cleaned_text, max_chars=2600)
        user_prompt = (
            f"Title: {title}\n"
            f"Categories: {categories}\n"
            f"Excerpt:\n{excerpt}\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        parsed = extract_json_object(text)
        if parsed is None:
            parsed = {}

        return validate_generation(parsed, title, categories, cleaned_text)


def build_training_examples(
    title: str,
    url: str,
    pageid: Any,
    page_type: str,
    qa_pairs: List[Dict[str, str]],
    system_prompt: str,
) -> List[Dict[str, Any]]:
    examples = []

    for pair in qa_pairs:
        q = pair["question"].strip()
        a = pair["answer"].strip()

        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "metadata": {
                "title": title,
                "url": url,
                "category_guess": page_type,
                "pageid": pageid,
                "generator": "llm_validated_qa_pairs",
            },
        })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_docs/rdr2_root_raw.jsonl")
    parser.add_argument("--output-json", default="lora_data/rdr2_lora_source_qa.jsonl")
    parser.add_argument("--output-train", default="lora_data/rdr2_lora_train_qa.jsonl")
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful Red Dead Redemption 2 assistant. Answer using accurate in-game and story knowledge. If the answer is uncertain, say what you are unsure about."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_json = Path(args.output_json)
    output_train = Path(args.output_train)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_train.parent.mkdir(parents=True, exist_ok=True)

    generator = LLMGenerator(args.generator_model, max_new_tokens=args.max_new_tokens)

    kept_source = 0
    skipped_source = 0
    kept_examples = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_json.open("w", encoding="utf-8") as fjson, \
         output_train.open("w", encoding="utf-8") as ftrain:

        for idx, line in enumerate(fin, start=1):
            if args.max_records is not None and idx > args.max_records:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            title = str(record.get("title", "Untitled")).strip()
            url = str(record.get("url", "")).strip()
            pageid = record.get("pageid")
            categories = record.get("categories", [])
            cleaned_text = clean_text(record.get("text", ""))

            if not cleaned_text or len(cleaned_text) < 140:
                skipped_source += 1
                continue

            result = generator.generate_structured(title, categories, cleaned_text)

            source_record = {
                "title": title,
                "url": url,
                "text": cleaned_text,
                "pageid": pageid,
                "categories": categories,
                "category_guess": result["page_type"],
                "qa_pairs": result["qa_pairs"],
            }
            fjson.write(json.dumps(source_record, ensure_ascii=False) + "\n")
            kept_source += 1

            examples = build_training_examples(
                title=title,
                url=url,
                pageid=pageid,
                page_type=result["page_type"],
                qa_pairs=result["qa_pairs"],
                system_prompt=args.system_prompt,
            )
            for ex in examples:
                ftrain.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept_examples += 1

            if idx % 10 == 0:
                print(f"Processed {idx} records...")

    print(json.dumps({
        "kept_source_records": kept_source,
        "skipped_source_records": skipped_source,
        "kept_training_examples": kept_examples,
        "output_json": str(output_json),
        "output_train": str(output_train),
        "generator_model": args.generator_model,
    }, indent=2))


if __name__ == "__main__":
    main()