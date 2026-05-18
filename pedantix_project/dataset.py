from __future__ import annotations

import json
import bz2
import re
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from .text import Token, tokenize, word_set


@dataclass(frozen=True)
class WikiPage:
    title: str
    intro: str
    url: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.title}\n\n{self.intro}"

    @property
    def words(self) -> set[str]:
        return word_set(self.full_text)

    @property
    def title_words(self) -> set[str]:
        return word_set(self.title)

    def tokens(self) -> list[Token]:
        return tokenize(self.title, in_title=True) + [Token("\n\n", False, "", "")] + tokenize(
            self.intro, in_title=False
        )


def load_pages(path: str | Path) -> list[WikiPage]:
    pages: list[WikiPage] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                pages.append(WikiPage(title=row["title"], intro=row["intro"], url=row.get("url", "")))
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing required key {exc}") from exc
    return pages


def save_pages(pages: Iterable[WikiPage], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for page in pages:
            row = {"title": page.title, "intro": page.intro, "url": page.url}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_wikipedia_summaries(titles: Iterable[str], *, language: str = "fr") -> list[WikiPage]:
    """Fetch page summaries from the public Wikipedia REST API.

    This keeps the project tiny. For real training, prefer an offline dump and
    convert it to the same JSONL schema.
    """
    pages: list[WikiPage] = []
    for title in titles:
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "pedantix-project/0.1"})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("type") == "disambiguation" or not data.get("extract"):
            continue
        pages.append(
            WikiPage(
                title=data.get("title", title),
                intro=data["extract"],
                url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            )
        )
    return pages


def fetch_popular_titles(
    *,
    language: str = "fr",
    start: date,
    days: int,
    per_day: int,
    sleep_seconds: float = 0.15,
) -> list[str]:
    titles: OrderedDict[str, None] = OrderedDict()
    project = f"{language}.wikipedia.org"
    for offset in range(days):
        day = start + timedelta(days=offset)
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
            f"{project}/all-access/{day:%Y/%m/%d}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "pedantix-project/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue
        articles = data.get("items", [{}])[0].get("articles", [])
        for article in articles:
            title = article.get("article", "")
            if _is_probably_article_title(title):
                titles[title.replace("_", " ")] = None
            if len(titles) >= days * per_day:
                break
        time.sleep(sleep_seconds)
    return list(titles)


def fetch_wikipedia_summaries_batched(
    titles: Iterable[str],
    *,
    language: str = "fr",
    limit: int,
    batch_size: int = 20,
    min_intro_chars: int = 180,
    sleep_seconds: float = 1.0,
    checkpoint_path: str | Path | None = None,
) -> list[WikiPage]:
    pages: list[WikiPage] = load_pages(checkpoint_path) if checkpoint_path and Path(checkpoint_path).exists() else []
    seen: set[str] = {page.title.lower() for page in pages}
    batch: list[str] = []

    def flush(current: list[str]) -> None:
        if not current:
            return
        try:
            fetched = _fetch_summary_batch(current, language=language)
        except Exception:
            time.sleep(max(3.0, sleep_seconds * 3))
            return
        for page in fetched:
            key = page.title.lower()
            if key in seen:
                continue
            if len(page.intro) < min_intro_chars:
                continue
            if _is_excluded_page(page):
                continue
            pages.append(page)
            seen.add(key)
            if len(pages) >= limit:
                break
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            save_pages(pages, checkpoint_path)

    for title in titles:
        if len(pages) >= limit:
            break
        if title.lower().replace("_", " ") in seen:
            continue
        batch.append(title)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
            time.sleep(sleep_seconds)
    if len(pages) < limit:
        flush(batch)
    return pages


def _fetch_summary_batch(titles: list[str], *, language: str) -> list[WikiPage]:
    joined = "|".join(title.replace(" ", "_") for title in titles)
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "inprop": "url",
            "titles": joined,
        }
    )
    url = f"https://{language}.wikipedia.org/w/api.php?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "pedantix-project/0.1"})
    data = _urlopen_json_with_retry(req)
    pages = []
    for item in data.get("query", {}).get("pages", {}).values():
        if "missing" in item:
            continue
        title = item.get("title", "")
        extract = item.get("extract", "").strip()
        if title and extract:
            pages.append(WikiPage(title=title, intro=extract, url=item.get("fullurl", "")))
    return pages


def _urlopen_json_with_retry(req: urllib.request.Request, *, retries: int = 5) -> dict:
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 1.8
    return {}


def _is_probably_article_title(title: str) -> bool:
    if not title or ":" in title:
        return False
    excluded = {
        "Accueil",
        "Main_Page",
        "Spécial:Recherche",
        "-",
    }
    if title in excluded:
        return False
    prefixes = ("Wikipédia:", "Special:", "Fichier:", "Modèle:", "Aide:", "Catégorie:")
    return not title.startswith(prefixes)


def _is_excluded_page(page: WikiPage) -> bool:
    title = page.title.lower()
    intro = page.intro.lower()
    if title.startswith(("liste de", "chronologie de")):
        return True
    if "peut désigner" in intro or "peuvent désigner" in intro:
        return True
    if "page d'homonymie" in intro or "page d’homonymie" in intro:
        return True
    return False


def ingest_pages_articles_dump(
    dump_path: str | Path,
    output_path: str | Path,
    *,
    limit: int,
    min_intro_chars: int = 220,
) -> int:
    """Stream a Wikimedia pages-articles XML .bz2 dump into the project JSONL.

    This is the scalable path for 100k+ pages. It does not load the dump into
    memory and checkpoints by rewriting the JSONL after each accepted page.
    """
    output = Path(output_path)
    existing = load_pages(output) if output.exists() else []
    seen = {page.title.lower() for page in existing}
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(existing)

    with output.open("a", encoding="utf-8") as out:
        for page in iter_pages_articles_dump(dump_path):
            if count >= limit:
                break
            key = page.title.lower()
            if key in seen:
                continue
            if len(page.intro) < min_intro_chars:
                continue
            if _is_excluded_page(page):
                continue
            out.write(json.dumps({"title": page.title, "intro": page.intro, "url": page.url}, ensure_ascii=False) + "\n")
            count += 1
            seen.add(key)
            if count % 1000 == 0:
                out.flush()
    return count


def iter_pages_articles_dump(dump_path: str | Path) -> Iterable[WikiPage]:
    with bz2.open(dump_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if _strip_ns(elem.tag) != "page":
                continue
            title = _find_child_text(elem, "title")
            ns = _find_child_text(elem, "ns")
            redirect = any(_strip_ns(child.tag) == "redirect" for child in elem)
            text = _find_revision_text(elem)
            if ns == "0" and title and text and not redirect and _is_probably_article_title(title):
                intro = extract_intro_from_wikitext(text)
                if intro:
                    url = "https://fr.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                    yield WikiPage(title=title, intro=intro, url=url)
            elem.clear()


def extract_intro_from_wikitext(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref\b[^>/]*/>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref\b.*?</ref>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _remove_balanced(text, "{{", "}}")
    text = _remove_balanced(text, "{|", "|}")
    text = re.sub(r"^\s*\[\[(?:Fichier|File|Image):.*?\]\]\s*$", " ", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.split(r"\n==+\s*", text, maxsplit=1)[0]
    paragraphs = []
    for para in re.split(r"\n\s*\n+", text):
        cleaned = _clean_wiki_markup(para)
        if len(cleaned) >= 30 and not cleaned.lower().startswith(("#redirect", "redirect")):
            paragraphs.append(cleaned)
        if len(" ".join(paragraphs)) >= 1200:
            break
    return " ".join(paragraphs).strip()


def _clean_wiki_markup(text: str) -> str:
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _remove_balanced(text: str, start: str, end: str) -> str:
    result: list[str] = []
    idx = 0
    depth = 0
    while idx < len(text):
        if text.startswith(start, idx):
            depth += 1
            idx += len(start)
            continue
        if depth and text.startswith(end, idx):
            depth -= 1
            idx += len(end)
            continue
        if depth == 0:
            result.append(text[idx])
        idx += 1
    return "".join(result)


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_child_text(elem: ET.Element, name: str) -> str:
    for child in elem:
        if _strip_ns(child.tag) == name:
            return child.text or ""
    return ""


def _find_revision_text(page_elem: ET.Element) -> str:
    for child in page_elem:
        if _strip_ns(child.tag) != "revision":
            continue
        for rev_child in child:
            if _strip_ns(rev_child.tag) == "text":
                return rev_child.text or ""
    return ""
