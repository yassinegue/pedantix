from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .dataset import WikiPage


MARKUP_RE = re.compile(r"(\[\[|\]\]|\{\{|\}\}|<ref\b|thumb\||vignette\||Fichier:|File:|Image:)", re.IGNORECASE)
HOMONYMY_RE = re.compile(
    r"(homonymie|peut faire référence|peut faire reference|peut désigner|peut designer|"
    r"peuvent désigner|peuvent designer|désigne plusieurs|designe plusieurs|"
    r"catégorie:homonymie|categorie:homonymie)",
    re.IGNORECASE,
)
LIST_RE = re.compile(r"^(liste|listes|chronologie|glossaire)\s+(de|des|du|d'|d’|historique)\b", re.IGNORECASE)


def iter_page_rows(path: str | Path) -> Iterable[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_pages(path: str | Path) -> Iterable[WikiPage]:
    for row in iter_page_rows(path):
        yield WikiPage(title=row["title"], intro=row["intro"], url=row.get("url", ""))


def sample_pages(path: str | Path, *, sample_size: int, seed: int, min_intro_words: int = 0) -> list[WikiPage]:
    rng = random.Random(seed)
    sample: list[WikiPage] = []
    eligible = 0
    for page in iter_pages(path):
        if min_intro_words > 0 and len(page.intro.split()) < min_intro_words:
            continue
        eligible += 1
        if len(sample) < sample_size:
            sample.append(page)
            continue
        chosen = rng.randrange(eligible)
        if chosen < sample_size:
            sample[chosen] = page
    return sample


def page_quality_issues(page: WikiPage, *, min_words: int = 40, min_chars: int = 220) -> list[str]:
    title = page.title.strip()
    intro = page.intro.strip()
    text = f"{title}\n{intro}"
    low_title = title.lower()
    issues: list[str] = []

    if len(intro) < min_chars:
        issues.append("too_short_chars")
    if len(intro.split()) < min_words:
        issues.append("too_short_words")
    if LIST_RE.search(low_title):
        issues.append("list_or_glossary_title")
    if intro.startswith("*") or "\n*" in intro[:500] or ": *" in intro[:500] or " ; *" in intro[:500]:
        issues.append("bullet_disambiguation")
    if re.search(r"\best un sigle\b|\best un acronyme\b|\best une abréviation\b|\best une abreviation\b", intro, re.I):
        issues.append("sigle_like")
    if HOMONYMY_RE.search(text):
        issues.append("homonymy_like")
    if MARKUP_RE.search(intro):
        issues.append("markup_residue")
    if re.search(r"\b(cette page|cet article)\s+(présente|presente)\s+(une liste|un glossaire)", intro, re.I):
        issues.append("list_intro")
    if "catégorie:" in intro.lower() or "categorie:" in intro.lower():
        issues.append("category_residue")
    return issues


def is_useful_page(page: WikiPage, *, min_words: int = 40, min_chars: int = 220) -> bool:
    return not page_quality_issues(page, min_words=min_words, min_chars=min_chars)


def filter_corpus(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_words: int = 40,
    min_chars: int = 220,
) -> tuple[int, int, Counter[str]]:
    kept = 0
    total = 0
    issue_counts: Counter[str] = Counter()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for page in iter_pages(input_path):
            total += 1
            issues = page_quality_issues(page, min_words=min_words, min_chars=min_chars)
            issue_counts.update(issues)
            if issues:
                continue
            out.write(json.dumps({"title": page.title, "intro": page.intro, "url": page.url}, ensure_ascii=False) + "\n")
            kept += 1
    return total, kept, issue_counts


def audit_corpus(path: str | Path, *, min_words: int = 40, min_chars: int = 220, shortest: int = 40) -> dict:
    lengths: list[int] = []
    word_counts: list[int] = []
    issue_counts: Counter[str] = Counter()
    title_prefixes: Counter[str] = Counter()
    shortest_rows: list[tuple[int, int, str, str, list[str]]] = []
    rejected = 0

    for page in iter_pages(path):
        intro = page.intro
        chars = len(intro)
        words = len(intro.split())
        lengths.append(chars)
        word_counts.append(words)
        issues = page_quality_issues(page, min_words=min_words, min_chars=min_chars)
        issue_counts.update(issues)
        if issues:
            rejected += 1
        title_prefixes.update([" ".join(page.title.lower().split()[:2])])
        row = (chars, words, page.title, intro[:260], issues)
        if len(shortest_rows) < shortest:
            shortest_rows.append(row)
        else:
            max_idx = max(range(len(shortest_rows)), key=lambda idx: shortest_rows[idx][0])
            if chars < shortest_rows[max_idx][0]:
                shortest_rows[max_idx] = row

    return {
        "rows": len(lengths),
        "chars_quantiles": _quantiles(lengths),
        "words_quantiles": _quantiles(word_counts),
        "issue_counts": dict(issue_counts.most_common()),
        "rejected_if_filtered": rejected,
        "top_title_prefixes": title_prefixes.most_common(30),
        "shortest": sorted(shortest_rows),
    }


def _quantiles(values: list[int]) -> dict[int, int]:
    if not values:
        return {}
    sorted_values = sorted(values)
    result: dict[int, int] = {}
    for pct in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        idx = round((pct / 100) * (len(sorted_values) - 1))
        result[pct] = sorted_values[idx]
    return result
