"""Sanity tests for the strong oracle used by DAgger.

We use a real held-out page and verify:
  - empty-history call returns a valid non-empty guess
  - given a near-solve state (most title words already revealed), the oracle
    prefers the remaining title word over generic starters
  - the oracle never returns an already-guessed word
  - the oracle never returns an invalid guess
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pedantix_project.dataset import WikiPage
from pedantix_project.llm_policy import (
    is_valid_guess,
    strong_oracle_guess,
)
from pedantix_project.model import TinyPedantixModel


def first_page() -> WikiPage:
    line = next(open(ROOT / "data" / "clean_pages.jsonl"))
    payload = json.loads(line)
    return WikiPage(title=payload["title"], intro=payload["intro"])


def title_words(page: WikiPage) -> list[str]:
    out = []
    for tok in page.tokens():
        if tok.is_word and tok.in_title:
            if tok.norm and is_valid_guess(tok.norm) and tok.norm not in out:
                out.append(tok.norm)
    return out


def test_empty_history_returns_valid_guess() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    guess = strong_oracle_guess(page, sim, [])
    assert guess, "oracle returned empty string on fresh state"
    assert is_valid_guess(guess), f"oracle returned invalid guess: {guess!r}"
    print(f"[ok] empty-history guess for {page.title!r}: {guess!r}")


def test_near_solve_prefers_title_word() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    tws = title_words(page)
    assert len(tws) >= 2, f"need a multi-word title; got {tws}"
    # Mark all title words except the last as already revealed (via exact hits)
    fake_history = []
    for w in tws[:-1]:
        fake_history.append({
            "guess": w, "exact": 1, "semantic": 0, "title": 1,
            "reward": 200.0, "solved": False, "invalid": False, "repeated": False,
        })
    remaining = tws[-1]
    guess = strong_oracle_guess(page, sim, fake_history)
    assert guess == remaining, (
        f"oracle should pick the remaining title word {remaining!r}, got {guess!r}"
    )
    print(f"[ok] near-solve picks remaining title word: {guess!r}")


def test_never_repeats() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    # Block the natural best guess and see what's chosen next.
    first = strong_oracle_guess(page, sim, [])
    fake_history = [{
        "guess": first, "exact": 0, "semantic": 0, "title": 0,
        "reward": -10.0, "solved": False, "invalid": False, "repeated": False,
    }]
    second = strong_oracle_guess(page, sim, fake_history)
    assert second != first, f"oracle repeated guess {first!r}"
    assert is_valid_guess(second)
    print(f"[ok] no repeats: 1st={first!r} 2nd={second!r}")


def test_return_score() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    page = first_page()
    word, score = strong_oracle_guess(page, sim, [], return_score=True)
    assert isinstance(score, (int, float))
    assert is_valid_guess(word)
    print(f"[ok] return_score yields ({word!r}, {score:.2f})")


if __name__ == "__main__":
    test_empty_history_returns_valid_guess()
    test_near_solve_prefers_title_word()
    test_never_repeats()
    test_return_score()
    print("ALL ORACLE TESTS PASSED")
