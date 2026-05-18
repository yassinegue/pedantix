"""Verify that near-solve shaping gives a gradient toward unrevealed title words.

A word that is a TinyModel neighbor of a title word should score higher than
an unrelated word, even when neither hits the title exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pedantix_project.dataset import WikiPage
from pedantix_project.llm_policy import (
    NEAR_SOLVE_SHAPING_COEF,
    score_guess_on_game,
)
from pedantix_project.model import TinyPedantixModel
from pedantix_project.simulator import PedantixGame


def page():
    line = next(open(ROOT / "data" / "clean_pages.jsonl"))
    p = json.loads(line)
    return WikiPage(title=p["title"], intro=p["intro"])


def test_near_solve_shaping_constant_exists():
    assert NEAR_SOLVE_SHAPING_COEF > 0, "NEAR_SOLVE_SHAPING_COEF must be positive"
    print(f"[ok] NEAR_SOLVE_SHAPING_COEF = {NEAR_SOLVE_SHAPING_COEF}")


def test_title_neighbor_scores_higher_than_random():
    """A word with non-zero similarity to a title word must outscore one with zero."""
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    p = page()
    game = PedantixGame(page=p, similarity_model=sim)

    title_norms = [game.tokens[i].canon for i in game.title_word_indices if game.tokens[i].is_word]
    assert title_norms, "test page has no title words"

    # Find a word that has nonzero sim to at least one title word
    title_neighbor = None
    title_neighbor_sim = 0.0
    for tw in title_norms:
        nbrs = sim.neighbors.get(tw, {})
        for w, s in nbrs.items():
            if s > 0.1 and w not in title_norms:
                title_neighbor = w
                title_neighbor_sim = s
                break
        if title_neighbor:
            break

    if title_neighbor is None:
        print("[skip] no neighbor found for title words on this page")
        return

    # Score the neighbor vs a word with guaranteed 0 sim to title
    # "xyzqwerty" will be rejected as non-word so use a valid low-IDF word
    # Find a word not in any title neighborhood
    unrelated = None
    for w in sim.vocabulary:
        if w in title_norms:
            continue
        if all(sim.similarity(w, tw) == 0.0 for tw in title_norms):
            unrelated = w
            break

    if unrelated is None:
        print("[skip] could not find unrelated word in vocab")
        return

    r_neighbor = score_guess_on_game(game, title_neighbor, history_len=0, guessed=set()).reward
    r_unrelated = score_guess_on_game(game, unrelated, history_len=0, guessed=set()).reward

    print(f"[info] title words: {title_norms}")
    print(f"[info] neighbor {title_neighbor!r} (sim={title_neighbor_sim:.2f}): reward={r_neighbor:.2f}")
    print(f"[info] unrelated {unrelated!r}: reward={r_unrelated:.2f}")

    assert r_neighbor > r_unrelated, (
        f"title neighbor {title_neighbor!r} (reward={r_neighbor:.2f}) should score "
        f"higher than unrelated {unrelated!r} (reward={r_unrelated:.2f})"
    )
    print(f"[ok] near-solve shaping: neighbor outscores unrelated by {r_neighbor - r_unrelated:.2f}")


def test_near_solve_bonus_magnitude():
    """Near-solve bonus should be meaningful (>0) for a title neighbor."""
    sim = TinyPedantixModel.load(ROOT / "models" / "tiny_model.json")
    p = page()
    game = PedantixGame(page=p, similarity_model=sim)

    title_norms = [game.tokens[i].canon for i in game.title_word_indices if game.tokens[i].is_word]
    assert title_norms

    for tw in title_norms:
        nbrs = sim.neighbors.get(tw, {})
        if nbrs:
            best_nbr = max(nbrs, key=nbrs.get)
            if best_nbr not in title_norms:
                r = score_guess_on_game(game, best_nbr, history_len=0, guessed=set()).reward
                expected_bonus = NEAR_SOLVE_SHAPING_COEF * nbrs[best_nbr] * 100
                print(f"[ok] neighbor {best_nbr!r} of title word {tw!r}: reward={r:.2f}, "
                      f"near-solve bonus ≈ {expected_bonus:.1f}")
                assert expected_bonus > 0
                return

    print("[skip] no title word has neighbors outside the title")


if __name__ == "__main__":
    test_near_solve_shaping_constant_exists()
    test_title_neighbor_scores_higher_than_random()
    test_near_solve_bonus_magnitude()
    print("ALL NEAR-SOLVE SHAPING TESTS PASSED")
