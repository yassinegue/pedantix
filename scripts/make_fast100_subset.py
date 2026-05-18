from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedantix_project.corpus import iter_pages
from pedantix_project.dataset import WikiPage, save_pages
from pedantix_project.text import word_set


def is_good_fast_page(page: WikiPage, *, max_title_words: int) -> bool:
    title_words = page.title_words
    if not title_words or len(title_words) > max_title_words:
        return False
    if not title_words <= word_set(page.intro):
        return False
    if len(page.intro) < 240 or len(page.intro.split()) < 45:
        return False
    if any(len(word) < 3 for word in title_words):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="data/clean_pages.jsonl")
    parser.add_argument("--output", default="data/fast100_pages.jsonl")
    parser.add_argument("--sample-pages", type=int, default=100)
    parser.add_argument("--max-title-words", type=int, default=2)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    candidates = [
        page
        for page in iter_pages(args.pages)
        if is_good_fast_page(page, max_title_words=args.max_title_words)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    pages = candidates[: args.sample_pages]
    if len(pages) < args.sample_pages:
        raise SystemExit(f"only found {len(pages)} suitable pages")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_pages(pages, output)
    print(f"wrote {len(pages)} pages -> {output}")


if __name__ == "__main__":
    main()
