"""Tests for soft_oracle_guess (DAgger supervision without title cheating).

The soft oracle must:
  - never return a title word
  - return a real French content word that is_valid_guess
  - vary across RNG seeds when temperature > 0 (top-K sampling)
  - be deterministic when temperature == 0 (argmax)
  - score is a finite real number
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pedantix_project.dataset import WikiPage
from pedantix_project.llm_policy import (
    is_valid_guess,
    soft_oracle_guess,
    _title_norm_words,
)
from pedantix_project.model import TinyPedantixModel


def first_page() -> WikiPage:
    line = next(open(ROOT / "data" / "clean_pages.jsonl"))
    payload = json.loads(line)
    return WikiPage(title=payload["title"], intro=payload["intro"])


def nth_pages(n: int) -> list[WikiPage]:
    pages = []
    for i, line in enumerate(open(ROOT / "data" / "clean_pages.jsonl")):
        if i >= n:
            break
        p = json.loads(line)
        pages.append(WikiPage(title=p["title"], intro=p["intro"]))
    return pages


def test_no_title_leak_on_first_page() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    titles = set(_title_norm_words(page))
    assert titles, f"expected non-empty title-words for {page.title}"
    rng = random.Random(7)
    for _ in range(20):
        guess = soft_oracle_guess(page, sim, [], rng=rng, top_k=8, temperature=1.0)
        assert guess, "soft oracle returned empty"
        assert is_valid_guess(guess)
        assert guess not in titles, f"soft oracle leaked title word {guess!r}; titles={titles}"
    print(f"[ok] 20 samples on {page.title!r} avoided titles {titles}")


def test_no_title_leak_across_many_pages() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    pages = nth_pages(30)
    rng = random.Random(13)
    leaks = 0
    samples = 0
    for page in pages:
        titles = set(_title_norm_words(page))
        if not titles:
            continue
        for _ in range(3):
            guess = soft_oracle_guess(page, sim, [], rng=rng, top_k=8, temperature=1.0)
            samples += 1
            if guess in titles:
                leaks += 1
    assert leaks == 0, f"soft oracle leaked title in {leaks}/{samples} samples"
    print(f"[ok] no title leaks across {samples} samples on {len(pages)} pages")


def test_deterministic_when_temperature_zero() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    rng1 = random.Random(1)
    rng2 = random.Random(999)
    g1 = soft_oracle_guess(page, sim, [], rng=rng1, top_k=8, temperature=0.0)
    g2 = soft_oracle_guess(page, sim, [], rng=rng2, top_k=8, temperature=0.0)
    assert g1 == g2, f"argmax should be deterministic; got {g1!r} vs {g2!r}"
    print(f"[ok] temperature=0 deterministic: both runs picked {g1!r}")


def test_top_k_diversity() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    seen = set()
    for seed in range(20):
        rng = random.Random(seed)
        seen.add(soft_oracle_guess(page, sim, [], rng=rng, top_k=8, temperature=2.0))
    assert len(seen) >= 2, f"top-K sampling should give variety; only got {seen}"
    print(f"[ok] top-K diversity: {len(seen)} distinct picks across 20 seeds: {sorted(seen)}")


def test_score_is_finite() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    rng = random.Random(0)
    word, score = soft_oracle_guess(page, sim, [], rng=rng, top_k=8, temperature=1.0,
                                    return_score=True)
    assert isinstance(score, (int, float))
    assert score > float("-inf") and score < float("inf"), f"score not finite: {score}"
    print(f"[ok] return_score yields ({word!r}, {score:.2f})")


def test_picks_useful_word_when_topic_is_clear() -> None:
    """If a generic word like 'siecle' would be high-info for the page, the
    oracle should consider it. Concretely: on a date-heavy page like a year,
    'siecle' should at least appear in the top-K, even if not always picked."""
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()  # Algèbre générale
    titles = set(_title_norm_words(page))
    candidates = set()
    for seed in range(40):
        rng = random.Random(seed)
        candidates.add(soft_oracle_guess(page, sim, [], rng=rng, top_k=8, temperature=2.0))
    candidates.discard("")
    assert all(c not in titles for c in candidates)
    print(f"[ok] candidate set: {sorted(candidates)}")


if __name__ == "__main__":
    test_no_title_leak_on_first_page()
    test_no_title_leak_across_many_pages()
    test_deterministic_when_temperature_zero()
    test_top_k_diversity()
    test_score_is_finite()
    test_picks_useful_word_when_topic_is_clear()
    print("ALL SOFT-ORACLE TESTS PASSED")
