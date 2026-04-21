import argparse
import json
import re
import time
from collections import deque
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API_URL = "https://reddead.fandom.com/api.php"
ROOT_CATEGORY = "Category:Red Dead Redemption II"
OUTPUT_DIR = Path("raw_docs")
USER_AGENT = "Mozilla/5.0 (compatible; RDR2RootCrawler/1.0)"
REQUEST_DELAY = 0.3

ALLOWED_CATEGORY_NAMESPACE = 14
ALLOWED_PAGE_NAMESPACE = 0

FALLBACK_PAGES = [
    "Missions in Redemption 2",
    "Red Dead Redemption 2",
    "Arthur Morgan",
    "John Marston",
    "Dutch van der Linde",
]

SKIP_TITLE_PREFIXES = (
    "File:",
    "Template:",
    "Talk:",
    "User:",
    "User blog:",
    "Forum:",
    "Message Wall:",
    "Category blog:",
    "Help:",
    "MediaWiki:",
    "Special:",
)

SKIP_TITLE_CONTAINS = (
    "disambiguation",
    "policy",
    "guideline",
    "staff",
    "admin",
    "community",
)

def should_skip_title(title: str) -> bool:
    lowered = title.lower()
    if title.startswith(SKIP_TITLE_PREFIXES):
        return True
    return any(x in lowered for x in SKIP_TITLE_CONTAINS)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "sup", "noscript", "figure", "img", "aside", "nav", "table"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    deduped = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
        prev = line

    return "\n".join(deduped).strip()

def get_category_members(category_title: str):
    members = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = requests.get(
            API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("query", {}).get("categorymembers", [])
        members.extend(batch)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

        time.sleep(REQUEST_DELAY)

    return members

def fetch_page(title: str):
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text|categories",
        "redirects": 1,
    }

    resp = requests.get(
        API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    parsed = data["parse"]
    html = parsed["text"]["*"]
    text = html_to_text(html)
    categories = [c.get("*") for c in parsed.get("categories", []) if c.get("*")]
    url = f"https://reddead.fandom.com/wiki/{title.replace(' ', '_')}"

    return {
        "title": parsed.get("title", title),
        "pageid": parsed.get("pageid"),
        "url": url,
        "text": text,
        "categories": categories,
        "source_category": None,
        "depth_found": None,
    }

def crawl_categories(root_category: str, max_depth: int):
    visited_categories = set()
    discovered_pages = {}
    crawl_log = []
    queue = deque([(root_category, 0)])

    while queue:
        category_title, depth = queue.popleft()

        if category_title in visited_categories:
            continue
        visited_categories.add(category_title)

        print(f"[CRAWL] {category_title} (depth={depth})")

        try:
            members = get_category_members(category_title)
        except Exception as e:
            print(f"[WARN] Failed category {category_title}: {e}")
            crawl_log.append({
                "category": category_title,
                "depth": depth,
                "error": str(e),
            })
            continue

        page_count = 0
        subcat_count = 0
        skipped_count = 0

        for member in members:
            title = member.get("title", "")
            ns = member.get("ns")

            if ns == ALLOWED_PAGE_NAMESPACE:
                if should_skip_title(title):
                    skipped_count += 1
                    continue
                discovered_pages.setdefault(title, {
                    "title": title,
                    "source_category": category_title,
                    "depth_found": depth,
                })
                page_count += 1
            elif ns == ALLOWED_CATEGORY_NAMESPACE:
                subcat_count += 1
                if depth < max_depth and not should_skip_title(title):
                    queue.append((title, depth + 1))
            else:
                skipped_count += 1

        crawl_log.append({
            "category": category_title,
            "depth": depth,
            "pages_found": page_count,
            "subcategories_found": subcat_count,
            "skipped_items": skipped_count,
        })

        time.sleep(REQUEST_DELAY)

    for title in FALLBACK_PAGES:
        if not should_skip_title(title):
            discovered_pages.setdefault(title, {
                "title": title,
                "source_category": "fallback",
                "depth_found": -1,
            })

    return discovered_pages, crawl_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--output", default="raw_docs/rdr2_root_raw.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    discovered_pages, crawl_log = crawl_categories(ROOT_CATEGORY, args.max_depth)
    titles = sorted(discovered_pages.keys())

    with (output_path.parent / "rdr2_root_titles.json").open("w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)

    with (output_path.parent / "rdr2_root_crawl_log.json").open("w", encoding="utf-8") as f:
        json.dump(crawl_log, f, ensure_ascii=False, indent=2)

    kept = 0
    skipped = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out:
        for idx, title in enumerate(titles, start=1):
            if args.max_pages is not None and kept >= args.max_pages:
                break
            try:
                record = fetch_page(title)
                record["source_category"] = discovered_pages[title]["source_category"]
                record["depth_found"] = discovered_pages[title]["depth_found"]

                if len(record["text"]) < args.min_chars:
                    skipped += 1
                    print(f"[SKIP {idx}/{len(titles)}] {title} ({len(record['text'])} chars)")
                    continue

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                print(f"[SAVE {idx}/{len(titles)}] {title} ({len(record['text'])} chars)")
            except Exception as e:
                failed += 1
                print(f"[FAIL {idx}/{len(titles)}] {title}: {e}")

            time.sleep(REQUEST_DELAY)

    print(json.dumps({
        "kept": kept,
        "skipped": skipped,
        "failed": failed,
        "total_candidates": len(titles),
        "output": str(output_path),
    }, indent=2))

if __name__ == "__main__":
    main()
