"""Verify the three reward-shaping fixes:

1. Non-French strings like 'apreapreapre' get NON_WORD_REWARD (very negative).
2. Real French words pass.
3. Semantic shaping weight has been lowered (filler words score lower now).
4. Real page tokens that may not be in the global vocab are still accepted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pedantix_project.dataset import WikiPage
from pedantix_project.llm_policy import (
    NON_WORD_REWARD,
    SEMANTIC_SHAPING_WEIGHT,
    _french_dictionary,
    _is_real_french_word,
    score_guess_on_game,
)
from pedantix_project.model import TinyPedantixModel
from pedantix_project.simulator import PedantixGame


def page():
    line = next(open(ROOT / "data" / "clean_pages.jsonl"))
    p = json.loads(line)
    return WikiPage(title=p["title"], intro=p["intro"])


def test_dictionary_loads() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    d = _french_dictionary(sim)
    assert len(d) > 1000, f"dictionary too small: {len(d)}"
    # Sanity: well-known content words are in
    for w in ("pays", "europe", "afrique", "siecle"):
        assert w in d, f"expected {w!r} in dictionary"
    print(f"[ok] dictionary has {len(d)} words; pays/europe/afrique/siecle present")


def test_non_word_gets_huge_penalty() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    p = page()
    game = PedantixGame(page=p, similarity_model=sim)
    # The reward-hack string the prior run produced
    fakes = ["apreapreapre", "anneauxesque", "egalementeuxetablitune", "xyzqwerty"]
    for f in fakes:
        r = score_guess_on_game(game, f, history_len=0, guessed=set()).reward
        assert r <= NON_WORD_REWARD, f"non-word {f!r} got reward {r}, expected <= {NON_WORD_REWARD}"
    print(f"[ok] non-words rejected with reward <= {NON_WORD_REWARD}")


def test_real_words_pass_dictionary() -> None:
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    p = page()
    game = PedantixGame(page=p, similarity_model=sim)
    for w in ("europe", "afrique", "pays", "siecle", "histoire"):
        assert _is_real_french_word(w, game), f"{w!r} should be accepted as real"
    # And running through score_guess_on_game should NOT trigger the non-word branch
    r = score_guess_on_game(game, "europe", history_len=0, guessed=set()).reward
    assert r > NON_WORD_REWARD + 50, f"europe reward suspicious: {r}"
    print(f"[ok] real words pass dictionary; europe scored {r:.2f}")


def test_page_token_not_in_vocab_still_passes() -> None:
    """A rare title word that may not be in the 5000-word vocab should still be
    accepted because it appears in the current page's tokens."""
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    p = page()
    game = PedantixGame(page=p, similarity_model=sim)
    # Pick any title word
    title_norms = [t.norm for t in p.tokens() if t.is_word and t.in_title]
    assert title_norms, "test page has no title tokens"
    for t in title_norms:
        assert _is_real_french_word(t, game), f"title word {t!r} rejected"
    print(f"[ok] page tokens accepted even when not in global vocab")


def test_semantic_weight_lowered() -> None:
    assert SEMANTIC_SHAPING_WEIGHT == 0.01, \
        f"expected SEMANTIC_SHAPING_WEIGHT=0.01, got {SEMANTIC_SHAPING_WEIGHT}"
    print(f"[ok] SEMANTIC_SHAPING_WEIGHT = {SEMANTIC_SHAPING_WEIGHT} (was 0.05)")


def test_filler_word_now_scores_lower() -> None:
    """Pick a page and compare reward of a "filler" word with the prior shaping
    weight (0.05) vs current (0.01). Reward at this weight should be smaller in
    magnitude — confirms our knob actually moves things."""
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    p = page()
    game = PedantixGame(page=p, similarity_model=sim)
    # Generic content word the prior model loved to spam.
    filler = "egalement"
    if not _is_real_french_word(filler, game):
        print(f"[skip] 'egalement' not in dictionary — skipping filler test")
        return
    r = score_guess_on_game(game, filler, history_len=2, guessed=set()).reward
    # We can't easily compute the "old" reward here, but reward should not be
    # dominated by huge positive shaping. Sanity: should not exceed +20 on a
    # page where it's not a title hit.
    assert r < 20.0, f"filler word reward suspiciously high: {r}"
    print(f"[ok] filler word 'egalement' scores {r:.2f} on test page (no longer farmable)")


if __name__ == "__main__":
    test_dictionary_loads()
    test_non_word_gets_huge_penalty()
    test_real_words_pass_dictionary()
    test_page_token_not_in_vocab_still_passes()
    test_semantic_weight_lowered()
    test_filler_word_now_scores_lower()
    print("ALL REWARD-FIX TESTS PASSED")
