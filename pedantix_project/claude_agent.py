"""Claude API agent for Pedantix.

Plays the game using Claude via the Anthropic API. Used for:
1. Establishing a baseline solve rate on held-out pages.
2. Generating high-quality (state → word) SFT training data.

The prompt format matches exactly what Qwen3-4B receives at inference,
so Claude's guesses can be used directly as training completions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .dataset import WikiPage
from .llm_policy import (
    _apply_llm_exact_reveals,
    _is_real_french_word,
    compact_visible_text,
    compute_page_max_sim,
    extract_guess,
    make_prompt,
    score_guess_on_game,
)
from .model import TinyPedantixModel
from .simulator import PedantixGame


_SYSTEM_PROMPT = (
    "Tu es un expert Wikipedia francais. Tu dois deviner le titre d'un article en proposant des mots un par un. "
    "Reponds UNIQUEMENT avec 'MOT: <mot>' en minuscules. Un seul mot, rien d'autre."
)


def play_game_with_claude(
    page: WikiPage,
    sim_model: TinyPedantixModel,
    *,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    max_steps: int = 30,
    retry_delay: float = 2.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Play one Pedantix game with Claude as the agent.

    Returns a dict with keys: title, solved, steps, history.
    Each history entry: {guess, exact, semantic, title, reward, solved, prompt}.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    game = PedantixGame(page, similarity_model=sim_model)
    history: list[dict] = []
    guessed: set[str] = set()

    for step_num in range(max_steps):
        if game.solved:
            break

        visible = compact_visible_text(game)
        prompt = make_prompt(history, max_steps=max_steps, visible_text=visible)

        # Call Claude with retries
        response_text = ""
        for attempt in range(max_retries):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=10,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = msg.content[0].text.strip()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    response_text = "MOT: france"  # safe fallback

        guess = extract_guess(response_text)
        if not guess:
            guess = "france"

        score = compute_page_max_sim(game, guess)
        result = score_guess_on_game(
            game, guess, history_len=len(history), guessed=guessed
        )
        _apply_llm_exact_reveals(game, guess)
        guessed.add(guess)

        history.append(
            {
                "guess": guess,
                "exact": result.exact_hits,
                "score": score,
                "semantic": result.semantic_hits,
                "title": result.title_hits,
                "reward": round(result.reward, 4),
                "solved": result.solved,
                "prompt": prompt,
                "completion": f"MOT: {guess}",
            }
        )

    return {
        "title": page.title,
        "solved": game.solved,
        "steps": len(history),
        "history": history,
    }


def run_claude_eval(
    pages_path: Path,
    sim_model: TinyPedantixModel,
    *,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    n_pages: int = 100,
    max_steps: int = 30,
    out_jsonl: Path,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Play N games with Claude and write per-game results to out_jsonl.

    Returns summary: {solve_rate, avg_steps, n_solved, n_pages}.
    """
    import random

    rng = random.Random(seed)
    pages: list[WikiPage] = []
    with open(pages_path) as f:
        pool = [json.loads(l) for l in f]
    rng.shuffle(pool)
    for p in pool[:n_pages]:
        pages.append(WikiPage(title=p["title"], intro=p["intro"]))

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    solved_count = 0
    total_steps = 0

    with open(out_jsonl, "w") as fout:
        for i, page in enumerate(pages):
            game_result = play_game_with_claude(
                page, sim_model,
                api_key=api_key, model=model, max_steps=max_steps,
            )
            # Strip prompts from eval output to keep file small
            for step in game_result["history"]:
                step.pop("prompt", None)
                step.pop("completion", None)
            fout.write(json.dumps(game_result, ensure_ascii=False) + "\n")
            fout.flush()
            if game_result["solved"]:
                solved_count += 1
            total_steps += game_result["steps"]
            if verbose:
                status = "SOLVED" if game_result["solved"] else f"failed({game_result['steps']})"
                guesses = [s["guess"] for s in game_result["history"][:5]]
                print(f"[{i+1}/{n_pages}] {page.title!r}: {status} guesses={guesses}")

    summary = {
        "solve_rate": solved_count / max(n_pages, 1),
        "n_solved": solved_count,
        "n_pages": n_pages,
        "avg_steps": total_steps / max(n_pages, 1),
        "model": model,
    }
    if verbose:
        print(
            f"\nSummary: solve_rate={summary['solve_rate']:.3f} ({solved_count}/{n_pages}) "
            f"avg_steps={summary['avg_steps']:.1f}"
        )
    return summary


def generate_claude_sft_data(
    pages_path: Path,
    sim_model: TinyPedantixModel,
    *,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    n_pages: int = 1000,
    max_steps: int = 30,
    out_jsonl: Path,
    seed: int = 42,
    eos_token: str = "<|im_end|>",
    verbose: bool = True,
) -> int:
    """Generate SFT training examples from Claude gameplay.

    Each step in each game becomes one training example:
      {"prompt": "<game state>", "completion": "MOT: word<|im_end|>"}

    Only steps where Claude guessed a valid French word are kept.
    Returns the number of training examples written.
    """
    import random

    rng = random.Random(seed)
    pages: list[WikiPage] = []
    with open(pages_path) as f:
        pool = [json.loads(l) for l in f]
    rng.shuffle(pool)
    for p in pool[:n_pages]:
        pages.append(WikiPage(title=p["title"], intro=p["intro"]))

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_examples = 0
    n_filtered = 0

    with open(out_jsonl, "w") as fout:
        for i, page in enumerate(pages):
            game_result = play_game_with_claude(
                page, sim_model,
                api_key=api_key, model=model, max_steps=max_steps,
            )
            page_examples = 0
            sim_game = PedantixGame(page, similarity_model=sim_model)
            for step in game_result["history"]:
                guess = step["guess"]
                if not _is_real_french_word(guess, sim_game):
                    n_filtered += 1
                    continue
                prompt = step.get("prompt", "")
                if not prompt:
                    n_filtered += 1
                    continue
                completion = f"MOT: {guess}{eos_token}"
                fout.write(
                    json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)
                    + "\n"
                )
                n_examples += 1
                page_examples += 1
            if verbose and (i + 1) % 50 == 0:
                print(
                    f"[{i+1}/{n_pages}] examples so far: {n_examples} "
                    f"(filtered {n_filtered})"
                )

    if verbose:
        print(f"Done: {n_examples} training examples from {n_pages} games (filtered {n_filtered})")
    return n_examples
