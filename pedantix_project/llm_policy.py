from __future__ import annotations

import json
import os
import random
import re
import inspect
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .corpus import iter_pages, sample_pages
from .dataset import WikiPage
from .model import TinyPedantixModel
from .rl import STOPWORDS, THEME_SEEDS
from .simulator import PedantixGame
from .text import canonical_word, normalize_word


HF_CACHE_ROOT = Path("/Data/yassine.guennoun/pedantix/models/hf_cache")
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GAME_STEPS = 100
REPEATED_GUESS_REWARD = -100.0
INVALID_GUESS_REWARD = -100.0
NON_WORD_REWARD = -200.0  # not in French dictionary AND not in current page (kills 'apreapreapre'-style exploits)
NO_INFORMATION_REWARD = -30.0
NON_SOLVE_STEP_REWARD = -10.0
SOLVED_TITLE_REWARD = 1000.0
SEMANTIC_SHAPING_WEIGHT = 0.01  # was 0.05 — lowered to weaken the "spam common content words" reward hack
NEAR_SOLVE_SHAPING_COEF = 0.3  # dense gradient: bonus proportional to proximity to any unrevealed title word
LETTER_WORD_PATTERN = r"[^\W\d_](?:[^\W\d_]|['’-])*"
GUESS_RE = re.compile(rf"(?:^|\b)(?:MOT|WORD)\s*:\s*({LETTER_WORD_PATTERN})", re.IGNORECASE | re.UNICODE)
WORD_RE = re.compile(LETTER_WORD_PATTERN, re.UNICODE)
INVALID_GUESSES = {
    "mot",
    "mots",
    "motcle",
    "motcles",
    "prochain",
    "seul",
    "un",
    "une",
    "format",
    "cache",
    "cachee",
    "cacher",
    "page",
    "pedantix",
    "jeu",
    "pay",
    "think",
    "thinking",
    "thought",
    "reasoning",
    "raisonnement",
    "reflexion",
    "réflexion",
    "assistant",
    "tool",
    "tools",
    "appel",
    "fonction",
    "system",
    "user",
    "historique",
    "feedback",
    "exact",
    "proche",
    "titre",
    "solved",
    "visible",
    "texte",
    "reponse",
    "réponse",
}


def enforce_hf_cache(root: str | Path = HF_CACHE_ROOT) -> Path:
    """Pin all HF-related caches under project storage, never the home dir."""
    root = Path(root)
    paths = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "TRANSFORMERS_CACHE": root / "transformers",
        "HF_DATASETS_CACHE": root / "datasets",
        "TORCH_HOME": root / "torch",
        "XDG_CACHE_HOME": root / "xdg",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    return root


enforce_hf_cache()


@dataclass(frozen=True)
class LLMReward:
    reward: float
    guess: str
    exact_hits: int
    semantic_hits: int
    title_hits: int
    solved: bool
    invalid: bool = False


def hf_token_available() -> bool:
    load_hf_token_from_env_files()
    return any(os.environ.get(name) for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"))


def load_hf_token_from_env_files() -> None:
    if any(os.environ.get(name) for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
        return
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path("/Users/yassinegue/Downloads/Agentic-zork/.env"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                continue
            value = value.strip().strip("'\"")
            if value:
                os.environ.setdefault("HF_TOKEN", value)
                return


def extract_guess(completion: str) -> str:
    completion = strip_qwen_thinking(completion)
    match = GUESS_RE.search(completion.strip())
    raw = match.group(1) if match else ""
    if not raw:
        match = WORD_RE.search(completion)
        raw = match.group(0) if match else ""
    return normalize_word(raw)


def strip_qwen_thinking(completion: str) -> str:
    completion = re.sub(r"<think>.*?</think>", " ", completion, flags=re.IGNORECASE | re.DOTALL)
    completion = re.sub(r"</?think>", " ", completion, flags=re.IGNORECASE)
    return completion


def is_valid_guess(guess: str) -> bool:
    return (
        bool(guess)
        and len(guess) >= 2
        and any(ch.isalpha() for ch in guess)
        and not any(ch.isdigit() for ch in guess)
        and guess not in STOPWORDS
        and guess not in INVALID_GUESSES
    )


def make_prompt(history: list[dict], *, max_steps: int, visible_text: str | None = None) -> str:
    lines = [
        "Jeu: Pedantix francais. Propose un seul mot francais informatif.",
        "Ecris seulement le mot choisi apres le prefixe MOT:.",
        "Ne pense pas. N'ecris aucune explication, aucun raisonnement, aucune balise think.",
        "Interdit: repetition, cache, page, pedantix, jeu, explication, think, reasoning.",
        "Prefere des themes larges puis adapte-toi au texte visible.",
        "Objectif prioritaire: trouver le titre exact de la page.",
        f"Essais maximum dans cette partie: {max_steps}.",
        "",
    ]
    if not history:
        lines.extend(
            [
                "Historique: aucun.",
                "Choisis un theme general, pas un mot rare.",
            ]
        )
    else:
        lines.append("Historique des feedbacks:")
        for idx, step in enumerate(history, 1):
            lines.append(
                f"{idx}. mot={step['guess']} exact={step['exact']} proche={step['semantic']} "
                f"titre={step['title']} solved={step['solved']}"
            )
        guessed = ", ".join(step["guess"] for step in history[-12:])
        lines.append(f"Mots deja joues recemment: {guessed}")
    if visible_text:
        lines.extend(
            [
                "",
                "Texte visible:",
                visible_text,
            ]
        )
    lines.append("")
    lines.append("Reponse:")
    return "\n".join(lines)


def format_prompt_for_model(prompt: str, chat_format: str = "none") -> str:
    if chat_format == "none":
        return prompt
    if chat_format == "qwen":
        return f"<|im_start|>user\n{prompt}\n/no_think<|im_end|>\n<|im_start|>assistant\nMOT:"
    raise ValueError(f"unsupported chat_format={chat_format!r}")


def replay_game(
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    history: list[dict],
) -> PedantixGame:
    game = PedantixGame(page, similarity_model=similarity_model)
    for step in history:
        guess = str(step.get("guess", ""))
        if guess:
            _apply_llm_exact_reveals(game, guess)
    return game


def _apply_llm_exact_reveals(game: PedantixGame, guess: str) -> set[int]:
    guess = normalize_word(guess)
    if not guess:
        return set()
    revealed = set()
    game.guessed.add(guess)
    for idx, tok in enumerate(game.tokens):
        if tok.is_word and tok.norm == guess:
            game.revealed.add(idx)
            revealed.add(idx)
    return revealed


def compact_visible_text(game: PedantixGame, *, max_chars: int = 240) -> str:
    parts: list[str] = []
    hidden = False
    for idx, tok in enumerate(game.tokens):
        if not tok.is_word:
            parts.append(tok.text)
            hidden = False
        elif idx in game.revealed:
            parts.append(tok.text)
            hidden = False
        elif not hidden:
            parts.append("__")
            hidden = True
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"__([,;:.!?])", r"__\1", text)
    if len(text) <= max_chars:
        return text.strip()
    head = text[: max_chars // 2].rsplit(" ", 1)[0]
    tail = text[-max_chars // 2 :].split(" ", 1)[-1]
    return f"{head}\n[...]\n{tail}".strip()


def score_completion(
    *,
    title: str,
    intro: str,
    history: list[dict],
    completion: str,
    similarity_model: TinyPedantixModel,
    solve_bonus_scale: float = 1.0,
) -> LLMReward:
    page = WikiPage(title=title, intro=intro)
    game = replay_game(page, similarity_model, history)
    guess = extract_guess(completion)
    guessed = {str(step.get("guess", "")) for step in history}
    return score_guess_on_game(
        game,
        guess,
        history_len=len(history),
        guessed=guessed,
        solve_bonus_scale=solve_bonus_scale,
    )


_FRENCH_DICTIONARY_CACHE: dict[int, frozenset[str]] = {}


def _french_dictionary(similarity_model: TinyPedantixModel | None) -> frozenset[str]:
    """Cached union of (similarity_model.vocabulary, .starter_words, THEME_SEEDS)
    used to reject non-French strings like 'apreapreapre' before they can farm
    semantic shaping reward."""
    if similarity_model is None:
        return frozenset()
    key = id(similarity_model)
    cached = _FRENCH_DICTIONARY_CACHE.get(key)
    if cached is not None:
        return cached
    words: set[str] = set()
    for w in getattr(similarity_model, "vocabulary", []) or ():
        n = normalize_word(w)
        if n:
            words.add(n)
    for w in getattr(similarity_model, "starter_words", []) or ():
        n = normalize_word(w)
        if n:
            words.add(n)
    for w in THEME_SEEDS:
        n = normalize_word(w)
        if n:
            words.add(n)
    cached = frozenset(words)
    _FRENCH_DICTIONARY_CACHE[key] = cached
    return cached


def _is_real_french_word(guess: str, game: PedantixGame) -> bool:
    """A guess is 'real' if it is in the similarity model's vocabulary OR appears
    as a tokenized word in the current page. Page tokens are accepted because a
    rare word in the page may be the very thing we want to guess."""
    if not guess:
        return False
    if game.similarity_model is None:
        return True  # no dictionary available, accept everything
    if guess in _french_dictionary(game.similarity_model):
        return True
    for tok in game.tokens:
        if tok.is_word and tok.norm == guess:
            return True
    return False


def score_guess_on_game(
    game: PedantixGame,
    guess: str,
    *,
    history_len: int,
    guessed: set[str],
    solve_bonus_scale: float = 1.0,
) -> LLMReward:
    guess = normalize_word(guess)
    if not is_valid_guess(guess):
        return LLMReward(INVALID_GUESS_REWARD, guess, 0, 0, 0, game.solved, invalid=True)
    if guess in guessed:
        return LLMReward(REPEATED_GUESS_REWARD, guess, 0, 0, 0, game.solved, invalid=True)
    if not _is_real_french_word(guess, game):
        return LLMReward(NON_WORD_REWARD, guess, 0, 0, 0, game.solved, invalid=True)

    before_title_hits = len(game.title_word_indices & game.revealed)
    exact_indices: set[int] = set()
    for idx, tok in enumerate(game.tokens):
        if tok.is_word and tok.norm == guess:
            exact_indices.add(idx)

    new_exact_indices = exact_indices - game.revealed
    revealed_after = game.revealed | exact_indices
    semantic_hits = 0
    semantic_info = 0.0
    title_semantic_info = 0.0
    semantic_guess = canonical_word(guess)
    if game.similarity_model is not None:
        for idx, tok in enumerate(game.tokens):
            if not tok.is_word or idx in revealed_after:
                continue
            sim = game.similarity_model.similarity(semantic_guess, tok.canon)
            if idx in game.title_word_indices:
                if sim > 0:  # no threshold for title tokens — any similarity gives gradient
                    title_semantic_info += _token_information(game.similarity_model, tok.canon) * sim
            else:
                if sim >= game.semantic_threshold:
                    semantic_hits += 1
                    semantic_info += _token_information(game.similarity_model, tok.canon) * sim

    title_hits = len(game.title_word_indices & revealed_after) - before_title_hits
    exact_hits = len(new_exact_indices)
    was_already_solved = bool(game.title_word_indices) and game.title_word_indices <= game.revealed
    solved = (not was_already_solved) and bool(game.title_word_indices) and game.title_word_indices <= revealed_after
    exact_info = sum(_token_information(game.similarity_model, game.tokens[idx].canon) for idx in new_exact_indices)
    title_info = sum(
        _token_information(game.similarity_model, game.tokens[idx].canon)
        for idx in (game.title_word_indices & revealed_after)
        if idx in new_exact_indices
    )

    reward = NON_SOLVE_STEP_REWARD
    reward += min(40.0, exact_info) * 0.8
    reward += min(80.0, semantic_info) * SEMANTIC_SHAPING_WEIGHT
    reward += min(7.0, title_semantic_info) * 3.0
    reward += min(30.0, title_info) * 25.0
    reward += min(4, title_hits) * 200.0
    if semantic_info >= 8.0 and semantic_hits >= 4:
        reward += min(4.0, semantic_info * SEMANTIC_SHAPING_WEIGHT)
    if solved:
        reward += SOLVED_TITLE_REWARD * solve_bonus_scale
        reward -= 2.0 * history_len
    # Dense near-solve shaping: gradient toward any unrevealed title word
    if game.similarity_model is not None:
        unrevealed_title_canons = [
            game.tokens[i].canon for i in game.title_word_indices
            if i not in revealed_after
        ]
        if unrevealed_title_canons:
            max_title_sim = max(
                game.similarity_model.similarity(semantic_guess, tw)
                for tw in unrevealed_title_canons
            )
            reward += NEAR_SOLVE_SHAPING_COEF * max_title_sim * 100
    # No useful information: check by IDF-weighted info, not raw hit count
    # This ensures low-IDF words like "de/le/les" are penalised even if they match many tokens
    no_useful_info = (exact_info == 0.0 and semantic_info == 0.0
                      and title_hits == 0 and title_semantic_info == 0.0)
    if no_useful_info:
        reward += NO_INFORMATION_REWARD
    elif exact_hits == 0 and title_hits == 0:
        reward -= 4.0
    return LLMReward(
        round(reward, 4),
        guess,
        exact_hits,
        semantic_hits,
        title_hits,
        solved,
    )


def _token_information(similarity_model: TinyPedantixModel | None, word: str) -> float:
    word = canonical_word(word)
    if not word or word in STOPWORDS or len(word) < 3:
        return 0.0
    if similarity_model is None:
        return 1.0
    return max(0.0, min(8.0, float(similarity_model.idf.get(word, 1.0))))


def build_llm_curriculum(
    pages_path: str | Path,
    output_path: str | Path,
    similarity_model: TinyPedantixModel,
    *,
    sample_size: int,
    states_per_page: int,
    max_steps: int,
    max_title_words: int | None,
    action_size: int,
    chat_format: str,
    seed: int,
    trajectory_mode: str = "teacher",
    min_intro_words: int = 0,
    min_history_len: int = 0,
) -> int:
    print(f"sampling {sample_size} pages from {pages_path}", flush=True)
    pages = sample_pages(pages_path, sample_size=sample_size, seed=seed, min_intro_words=min_intro_words)
    if max_title_words is not None:
        pages = [
            page
            for page in pages
            if page.title_words
            and len(page.title_words) <= max_title_words
        ]
    vocabulary = list(similarity_model.vocabulary[:action_size]) if action_size and action_size > 0 else list(similarity_model.vocabulary)
    print(
        f"building curriculum for {len(pages)} filtered pages "
        f"(fast teacher, vocab hints={len(vocabulary)}, min_history_len={min_history_len})",
        flush=True,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    starters = _starter_words(vocabulary, similarity_model)
    page_iter = _progress(pages, desc="LLM curriculum pages")
    with output.open("w", encoding="utf-8") as handle:
        for page in page_iter:
            history: list[dict] = []
            total_states = min_history_len + states_per_page
            if trajectory_mode == "oracle":
                planned_guesses = _oracle_trajectory_guesses(
                    page,
                    similarity_model,
                    vocabulary,
                    starters,
                    max_steps=max_steps,
                    max_guesses=total_states,
                )
            elif trajectory_mode == "teacher":
                planned_guesses = []
            else:
                raise ValueError(f"unsupported trajectory_mode={trajectory_mode!r}")

            rows_written = 0
            for state_idx in range(total_states):
                if trajectory_mode == "oracle":
                    teacher = planned_guesses[state_idx] if state_idx < len(planned_guesses) else ""
                else:
                    teacher = choose_teacher_guess(page, similarity_model, history, vocabulary, starters, state_idx, rng)
                if teacher:
                    # Only write rows once we've warmed up enough history
                    if state_idx >= min_history_len:
                        prompt = make_prompt(
                            history,
                            max_steps=max_steps,
                            visible_text=compact_visible_text(replay_game(page, similarity_model, history)),
                        )
                        formatted_prompt = format_prompt_for_model(prompt, chat_format=chat_format)
                        completion = f" {teacher}" if chat_format == "qwen" else f"MOT: {teacher}"
                        row = {
                            "prompt": formatted_prompt,
                            "completion": completion,
                            "text": formatted_prompt + completion,
                            "title": page.title,
                            "intro": page.intro,
                            "history": json.dumps(history, ensure_ascii=False),
                        }
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                        rows_written += 1
                    history = history + [_feedback_step(page, similarity_model, history, teacher)]
                if rows_written >= states_per_page or len(history) >= max_steps:
                    break
    return written


def _oracle_trajectory_guesses(
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    vocabulary: list[str],
    starters: list[str],
    *,
    max_steps: int,
    max_guesses: int,
) -> list[str]:
    """Build a high-reward solve path for bootstrapping GRPO states.

    GRPO still ignores the stored completion labels; these guesses are used to
    put training prompts on states that a competent policy could actually reach.
    """
    title_words = [word for word in _title_norm_words(page) if is_valid_guess(word)]
    if not title_words:
        return []
    clue_budget = max(0, min(max_steps, max_guesses) - len(title_words))
    if clue_budget <= 0:
        return title_words[:max_guesses]

    guesses: list[str] = []
    history: list[dict] = []
    title_set = set(title_words)

    starter_pool = [word for word in starters[:80] if word not in title_set]
    for _ in range(min(3, clue_budget)):
        word = _best_rewarding_guess(page, similarity_model, history, starter_pool, blocked=title_set | set(guesses))
        if not word:
            break
        guesses.append(word)
        history.append(_feedback_step(page, similarity_model, history, word))

    probe_pool = [word for word in _page_probe_words(page, similarity_model, limit=80) if word not in title_set]
    mixed_pool = probe_pool + [word for word in vocabulary[:500] if word not in title_set]
    while len(guesses) < clue_budget:
        word = _best_rewarding_guess(page, similarity_model, history, mixed_pool, blocked=title_set | set(guesses))
        if not word:
            break
        guesses.append(word)
        history.append(_feedback_step(page, similarity_model, history, word))

    for word in title_words:
        if len(guesses) >= max_guesses:
            break
        if word not in guesses:
            guesses.append(word)
    return guesses[:max_guesses]


def strong_oracle_guess(
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    history: list[dict],
    *,
    extra_candidates: list[str] | None = None,
    neighbor_budget: int = 40,
    solve_bonus_scale: float = 1.0,
    return_score: bool = False,
):
    """Pick the next guess that maximises score_guess_on_game.reward.

    Candidate pool ("close to cheating" — the oracle is allowed to look at the
    page title and intro directly):
      - title tokens (highest information)
      - intro probe tokens (high-IDF non-title words from the page text)
      - similarity-model neighbors of words already played (semantic siblings)
      - starter words (general French openers, for early-game)
      - any caller-provided extras

    Returns the highest-reward valid word that has not been guessed; falls back
    to '' when no candidate qualifies. With return_score=True, returns
    (word, reward).
    """
    guessed = {str(step.get("guess", "")) for step in history}
    game = replay_game(page, similarity_model, history)

    pool: set[str] = set()
    for word in _title_norm_words(page):
        pool.add(word)
    for word in _page_probe_words(page, similarity_model, limit=64):
        pool.add(word)
    seen_neighbors = 0
    for step in history[-6:]:
        prev = str(step.get("guess", ""))
        if not prev:
            continue
        for nb in similarity_model.neighbors.get(canonical_word(prev), {}):
            if seen_neighbors >= neighbor_budget:
                break
            pool.add(normalize_word(nb))
            seen_neighbors += 1
        if seen_neighbors >= neighbor_budget:
            break
    for word in similarity_model.starter_words[:60]:
        pool.add(normalize_word(word))
    for word in extra_candidates or ():
        pool.add(normalize_word(word))

    best_word = ""
    best_reward = float("-inf")
    for word in pool:
        if not word or word in guessed or not is_valid_guess(word):
            continue
        reward = score_guess_on_game(
            game, word, history_len=len(history), guessed=guessed,
            solve_bonus_scale=solve_bonus_scale,
        ).reward
        if reward > best_reward:
            best_word = word
            best_reward = reward

    if return_score:
        return best_word, best_reward
    return best_word


def soft_oracle_guess(
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    history: list[dict],
    *,
    top_k: int = 8,
    temperature: float = 1.0,
    neighbor_budget: int = 40,
    solve_bonus_scale: float = 1.0,
    min_idf: float = 0.0,
    rng: random.Random | None = None,
    return_score: bool = False,
):
    """Pick a high-information word *without* directly leaking the title.

    Unlike strong_oracle_guess, this excludes title tokens from the candidate
    pool, so the supervision teaches the student to *reason about topic from
    context* (revealed words, semantic feedback) rather than memorising
    title→page lookups it cannot perform on held-out pages.

    Candidate pool:
      - high-IDF NON-title words from the page intro (the "siecle/europe" kind
        of theme reveal)
      - similarity-model neighbors of words that already produced exact hits
        (sustainable signal: derived from the prompt's revealed text)
      - similarity-model neighbors of recent guesses (semantic siblings)
      - starter words (general openers)

    Scoring uses score_guess_on_game.reward, which already prizes IDF-weighted
    reveals, semantic_info, and title_semantic_info (proximity to title tokens
    without being them).

    Top-K sampling: with rng + temperature>0, samples from the K best-scored
    candidates weighted by softmax(reward/temperature) so the model sees a
    *distribution* of strong information-seekers rather than always one word.
    With temperature<=0, returns the argmax.
    """
    import math

    rng = rng or random.Random()
    guessed = {str(step.get("guess", "")) for step in history}
    game = replay_game(page, similarity_model, history)

    title_words = set(_title_norm_words(page))

    pool: set[str] = set()
    for word in _page_probe_words(page, similarity_model, limit=80):
        if word not in title_words:
            pool.add(word)

    seen_neighbors = 0
    for step in history[-6:]:
        prev = str(step.get("guess", ""))
        if not prev:
            continue
        prev_canon = canonical_word(prev)
        for nb in similarity_model.neighbors.get(prev_canon, {}):
            if seen_neighbors >= neighbor_budget:
                break
            nb_norm = normalize_word(nb)
            if nb_norm and nb_norm not in title_words:
                pool.add(nb_norm)
                seen_neighbors += 1
        if seen_neighbors >= neighbor_budget:
            break

    for word in similarity_model.starter_words[:80]:
        word = normalize_word(word)
        if word and word not in title_words:
            pool.add(word)

    scored: list[tuple[str, float]] = []
    for word in pool:
        if not word or word in guessed or not is_valid_guess(word):
            continue
        # IDF floor: drop candidates whose canonical form is too common (le, la,
        # est, …) — they have low information value and the model learned to
        # farm them in the prior run.
        if min_idf > 0 and similarity_model is not None:
            idf = float(similarity_model.idf.get(canonical_word(word), 0.0))
            if idf < min_idf:
                continue
        reward = score_guess_on_game(
            game, word, history_len=len(history), guessed=guessed,
            solve_bonus_scale=solve_bonus_scale,
        ).reward
        scored.append((word, reward))

    if not scored:
        return ("", float("-inf")) if return_score else ""

    scored.sort(key=lambda x: -x[1])
    top = scored[: max(1, top_k)]

    if temperature <= 0 or len(top) == 1:
        word, score = top[0]
    else:
        max_r = max(s for _, s in top)
        weights = [math.exp((s - max_r) / max(temperature, 1e-6)) for _, s in top]
        word, score = rng.choices(top, weights=weights, k=1)[0]

    if return_score:
        return word, score
    return word


def _best_rewarding_guess(
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    history: list[dict],
    candidates: list[str],
    *,
    blocked: set[str],
) -> str:
    guessed = {str(step.get("guess", "")) for step in history} | blocked
    game = replay_game(page, similarity_model, history)
    best_word = ""
    best_reward = float("-inf")
    for word in candidates:
        word = normalize_word(word)
        if word in guessed or not is_valid_guess(word):
            continue
        reward = score_guess_on_game(game, word, history_len=len(history), guessed=guessed).reward
        if reward > best_reward:
            best_word = word
            best_reward = reward
    return best_word


def _progress(items: list[WikiPage], *, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, unit="page")


def _tqdm(*args, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(*args, **kwargs)


def choose_teacher_guess(
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    history: list[dict],
    vocabulary: list[str],
    starters: list[str],
    state_idx: int,
    rng: random.Random,
) -> str:
    guessed = {str(step.get("guess", "")) for step in history}
    title_words = _title_norm_words(page)
    if any(int(step.get("title", 0)) > 0 for step in history):
        for word in title_words:
            if word not in guessed and is_valid_guess(word):
                return word
    probes = _page_probe_words(page, similarity_model, limit=32)
    if state_idx >= 2:
        for word in probes:
            if word not in guessed:
                return word
    starter_candidates = [word for word in starters[:80] if word not in guessed and is_valid_guess(word)]
    if starter_candidates:
        return rng.choice(starter_candidates[: min(24, len(starter_candidates))])
    for word in probes + title_words:
        if word not in guessed and is_valid_guess(word):
            return word
    return ""


def _title_norm_words(page: WikiPage) -> list[str]:
    words = []
    for tok in page.tokens():
        if not tok.is_word or not tok.in_title:
            continue
        word = normalize_word(tok.text)
        if word and word not in words:
            words.append(word)
    return words


def _page_probe_words(page: WikiPage, similarity_model: TinyPedantixModel, *, limit: int) -> list[str]:
    scored: dict[str, float] = {}
    for tok in page.tokens():
        if not tok.is_word or tok.in_title:
            continue
        word = normalize_word(tok.text)
        if not is_valid_guess(word):
            continue
        score = _token_information(similarity_model, tok.canon)
        if score <= 0:
            continue
        scored[word] = max(scored.get(word, 0.0), score)
    return [word for word, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def evaluate_llm_policy(
    *,
    pages_path: str | Path,
    model_path: str,
    similarity_model: TinyPedantixModel,
    sample_size: int,
    max_steps: int,
    seed: int,
    output_path: str | Path,
    chat_format: str = "none",
    generation_batch_size: int = 16,
    eval_num_generations: int = 8,
) -> dict:
    enforce_hf_cache()
    transformers, torch = _import_transformers()
    model, tokenizer = _load_model_and_tokenizer_for_inference(model_path, transformers, torch)
    device = _best_device(torch)
    model.to(device)
    model.eval()
    return _run_eval_games(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pages_path=pages_path,
        similarity_model=similarity_model,
        sample_size=sample_size,
        max_steps=max_steps,
        seed=seed,
        output_path=output_path,
        chat_format=chat_format,
        generation_batch_size=generation_batch_size,
        eval_num_generations=eval_num_generations,
    )


def _run_eval_games(
    *,
    model,
    tokenizer,
    device: str,
    pages_path: str | Path,
    similarity_model: TinyPedantixModel,
    sample_size: int,
    max_steps: int,
    seed: int,
    output_path: str | Path,
    chat_format: str = "none",
    generation_batch_size: int = 16,
    eval_num_generations: int = 8,
) -> dict:
    pages = sample_pages(pages_path, sample_size=sample_size, seed=seed)
    histories: list[list[dict]] = [[] for _ in pages]
    done = [False for _ in pages]
    generation_batch_size = max(1, generation_batch_size)
    progress = _tqdm(total=len(pages) * max_steps, desc="LLM eval guesses", unit="guess")
    generation_rejected_invalid = 0
    generation_rejected_repeated = 0
    generation_fallbacks = 0

    try:
        import torch as _torch
        _cuda_available = _torch.cuda.is_available()
        _batches_since_empty = 0
        for _ in range(max_steps):
            active_indices = [idx for idx, is_done in enumerate(done) if not is_done]
            if not active_indices:
                break
            for start in range(0, len(active_indices), generation_batch_size):
                batch_indices = active_indices[start : start + generation_batch_size]
                prompts = []
                batch_histories = []
                for idx in batch_indices:
                    history = histories[idx]
                    prompt = make_prompt(
                        history,
                        max_steps=max_steps,
                        visible_text=compact_visible_text(replay_game(pages[idx], similarity_model, history)),
                    )
                    prompts.append(format_prompt_for_model(prompt, chat_format=chat_format))
                    batch_histories.append(history)
                generation_stats = []
                guesses = generate_next_words(
                    model,
                    tokenizer,
                    prompts,
                    histories=batch_histories,
                    device=device,
                    num_return_sequences=eval_num_generations,
                    generation_stats=generation_stats,
                )
                for stat in generation_stats:
                    generation_rejected_invalid += int(stat.get("rejected_invalid", 0))
                    generation_rejected_repeated += int(stat.get("rejected_repeated", 0))
                    generation_fallbacks += int(stat.get("fallback", False))
                for idx, guess in zip(batch_indices, guesses):
                    step = _feedback_step(pages[idx], similarity_model, histories[idx], guess)
                    histories[idx].append(step)
                    if step["solved"]:
                        done[idx] = True
                if progress is not None:
                    progress.update(len(batch_indices))
                    progress.set_postfix(active=sum(1 for is_done in done if not is_done))
                _batches_since_empty += 1
                # Periodically reclaim per-generate KV-cache fragments. Without
                # this, fragmentation grows across the ~200 generate() calls in
                # a single eval and triggers OOM well before the eval completes.
                if _cuda_available and _batches_since_empty >= 8:
                    _batches_since_empty = 0
                    _torch.cuda.empty_cache()
    finally:
        if progress is not None:
            progress.close()

    rows = []
    solved = 0
    total_steps = 0
    repeated_guesses = 0
    invalid_guesses = 0
    first_10_words: Counter[str] = Counter()

    for page, history in zip(pages, histories):
        for step in history:
            if step.get("repeated"):
                repeated_guesses += 1
            if step.get("invalid"):
                invalid_guesses += 1
        for step in history[:10]:
            if step.get("guess"):
                first_10_words[str(step["guess"])] += 1
        solved += bool(history and history[-1]["solved"])
        total_steps += len(history)
        rows.append({"title": page.title, "solved": bool(history and history[-1]["solved"]), "steps": len(history), "history": history})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "pages": len(pages),
        "solve_rate": round(solved / max(1, len(pages)), 4),
        "mean_steps": round(total_steps / max(1, len(pages)), 4),
        "repeated_guesses": repeated_guesses,
        "invalid_guesses": invalid_guesses,
        "generation_rejected_repeated": generation_rejected_repeated,
        "generation_rejected_invalid": generation_rejected_invalid,
        "generation_fallbacks": generation_fallbacks,
        "common_first_10_words": first_10_words.most_common(25),
        "output": str(output),
    }


def generate_next_word(model, tokenizer, prompt: str, *, history: list[dict] | None = None, device: str) -> str:
    return generate_next_words(model, tokenizer, [prompt], histories=[history or []], device=device)[0]


def generate_next_words(
    model,
    tokenizer,
    prompts: list[str],
    *,
    histories: list[list[dict]],
    device: str,
    num_return_sequences: int = 8,
    generation_stats: list[dict] | None = None,
) -> list[str]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    generation_kwargs = _qwen_no_think_generation_kwargs(tokenizer, getattr(model.config, "name_or_path", ""))
    guessed_sets = [{str(step.get("guess", "")) for step in history or []} for history in histories]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    prompt_length = inputs["input_ids"].shape[1]
    choices: list[list[str]] = [[] for _ in prompts]
    rejected_invalid = [0 for _ in prompts]
    rejected_repeated = [0 for _ in prompts]
    for out_idx, row in enumerate(output):
        prompt_idx = out_idx // num_return_sequences
        completion = tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
        guess = extract_guess(completion)
        if not is_valid_guess(guess):
            rejected_invalid[prompt_idx] += 1
        elif guess in guessed_sets[prompt_idx]:
            rejected_repeated[prompt_idx] += 1
        else:
            choices[prompt_idx].append(guess)

    results = []
    for prompt_idx, guesses in enumerate(choices):
        if guesses:
            results.append(guesses[0])
            if generation_stats is not None:
                generation_stats.append(
                    {
                        "fallback": False,
                        "rejected_invalid": rejected_invalid[prompt_idx],
                        "rejected_repeated": rejected_repeated[prompt_idx],
                    }
                )
            continue
        results.append(_fallback_guess(guessed_sets[prompt_idx]))
        if generation_stats is not None:
            generation_stats.append(
                {
                    "fallback": True,
                    "rejected_invalid": rejected_invalid[prompt_idx],
                    "rejected_repeated": rejected_repeated[prompt_idx],
                }
            )
    return results


def _fallback_guess(guessed: set[str]) -> str:
    for guess in THEME_SEEDS:
        guess = normalize_word(guess)
        if guess not in guessed and is_valid_guess(guess):
            return guess
    for guess in [
        "geographie",
        "biologie",
        "economie",
        "litterature",
        "acteur",
        "football",
        "grec",
        "romain",
        "religion",
        "industrie",
        "mathematique",
    ]:
        guess = normalize_word(guess)
        if guess not in guessed and is_valid_guess(guess):
            return guess
    return "strategie"


def _legacy_generate_next_word(model, tokenizer, prompt: str, *, history: list[dict] | None = None, device: str) -> str:
    guessed = {str(step.get("guess", "")) for step in history or []}
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=8,
        pad_token_id=tokenizer.eos_token_id,
    )
    for row in output:
        completion = tokenizer.decode(row[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        guess = extract_guess(completion)
        if is_valid_guess(guess) and guess not in guessed:
            return guess
    for guess in THEME_SEEDS:
        if guess not in guessed and is_valid_guess(guess):
            return guess
    return extract_guess(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))


def train_llm_sft(
    *,
    train_jsonl: str | Path,
    model_name: str,
    output_dir: str | Path,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lora_rank: int,
    use_cpu: bool,
    seed: int,
    log_path: str | Path | None = None,
    eval_every_n_steps: int = 0,
    eval_pages: int = 50,
    eval_pages_path: str | Path | None = None,
    eval_max_game_steps: int = 30,
    eval_chat_format: str = "qwen",
    eval_num_generations: int = 4,
    eval_batch_size: int = 8,
    tiny_model_path: str | Path | None = None,
    save_steps: int | None = None,
    save_total_limit: int = 2,
    resume_from_checkpoint: str | Path | bool | None = None,
) -> None:
    enforce_hf_cache()
    datasets, trl, peft = _import_training_stack()
    dataset = datasets.load_dataset("json", data_files=str(train_jsonl), split="train")
    has_prompt_completion = {"prompt", "completion"} <= set(dataset.column_names)
    if not has_prompt_completion:
        extra_columns = [column for column in dataset.column_names if column != "text"]
        if extra_columns:
            dataset = dataset.remove_columns(extra_columns)
    if save_steps is None:
        effective_save_steps = eval_every_n_steps if eval_every_n_steps else max(25, max_steps)
    else:
        effective_save_steps = max(1, int(save_steps))
    config_kwargs = dict(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=5,
        save_steps=effective_save_steps,
        save_strategy="steps",
        save_total_limit=max(1, int(save_total_limit)),
        max_length=512,
        report_to=[],
        use_cpu=use_cpu,
        optim="adamw_torch",
        seed=seed,
        model_init_kwargs={"torch_dtype": "auto"},
    )
    if has_prompt_completion:
        config_kwargs["completion_only_loss"] = True
    else:
        config_kwargs["dataset_text_field"] = "text"
    config = trl.SFTConfig(
        **_supported_kwargs(
            trl.SFTConfig,
            **config_kwargs,
        )
    )
    peft_config = _lora_config(peft, lora_rank)

    callbacks = []
    trainer_holder: dict = {}
    if eval_every_n_steps and eval_pages_path and tiny_model_path:
        similarity_model = TinyPedantixModel.load(tiny_model_path)
        evaluator = _make_held_out_evaluator(
            trainer_holder=trainer_holder,
            output_dir=output_dir,
            eval_pages_path=eval_pages_path,
            similarity_model=similarity_model,
            sample_size=eval_pages,
            max_game_steps=eval_max_game_steps,
            chat_format=eval_chat_format,
            eval_num_generations=eval_num_generations,
            eval_batch_size=eval_batch_size,
            seed=seed,
        )
        callbacks.append(
            _build_periodic_eval_callback(
                every=eval_every_n_steps,
                evaluator=evaluator,
                label="eval",
            )
        )

    trainer = trl.SFTTrainer(
        model=model_name,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=callbacks or None,
    )
    trainer_holder["trainer"] = trainer
    resume_arg: object
    if resume_from_checkpoint in (None, False):
        resume_arg = None
    elif resume_from_checkpoint is True:
        resume_arg = True
    else:
        resume_arg = str(resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_arg)
    trainer.save_model(str(output_dir))
    if log_path:
        _write_trainer_log(trainer.state.log_history, log_path)


def train_llm_grpo(
    *,
    train_jsonl: str | Path,
    model_name_or_path: str,
    tiny_model_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    max_completion_length: int,
    learning_rate: float,
    lora_rank: int,
    use_cpu: bool,
    seed: int,
    log_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    logging_steps: int = 1,
    save_steps: int | None = None,
    temperature: float = 0.8,
    top_p: float = 0.9,
    show_progress: bool = True,
    resume_from_checkpoint: str | Path | None = None,
    solve_bonus_scale: float = 1.0,
    beta: float = 0.02,
    eval_every_n_steps: int = 0,
    eval_pages: int = 50,
    eval_pages_path: str | Path | None = None,
    eval_max_game_steps: int = 30,
    eval_chat_format: str = "qwen",
    eval_num_generations: int = 4,
    eval_batch_size: int = 8,
    dagger_every: int = 0,
    dagger_pages: int = 32,
    dagger_rollout_steps: int = 12,
    dagger_microsteps: int = 16,
    dagger_bc_batch_size: int = 4,
    dagger_history_max_steps: int = 30,
    dagger_chat_format: str | None = None,
    dagger_oracle_mode: str = "soft",
    dagger_oracle_top_k: int = 8,
    dagger_oracle_temperature: float = 1.0,
    dagger_oracle_min_idf: float = 0.0,
) -> None:
    enforce_hf_cache()
    datasets, trl, peft = _import_training_stack()
    similarity_model = TinyPedantixModel.load(tiny_model_path)
    dataset = datasets.load_dataset("json", data_files=str(train_jsonl), split="train")
    dataset = _prepare_grpo_prompts(dataset, model_name_or_path)
    model_arg = _load_peft_model_if_adapter(model_name_or_path)
    peft_config = None if not isinstance(model_arg, str) else _lora_config(peft, lora_rank)
    transformers, _ = _import_transformers()
    tokenizer_name = _tokenizer_name_for_model(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    generation_kwargs = _qwen_no_think_generation_kwargs(tokenizer, tokenizer_name)

    def reward_func(prompts, completions, title, intro, history, **kwargs):
        rewards = []
        for completion, page_title, page_intro, raw_history in zip(completions, title, intro, history):
            parsed_history = json.loads(raw_history) if isinstance(raw_history, str) else raw_history
            rewards.append(
                score_completion(
                    title=page_title,
                    intro=page_intro,
                    history=parsed_history,
                    completion=completion,
                    similarity_model=similarity_model,
                    solve_bonus_scale=solve_bonus_scale,
                ).reward
            )
        return rewards

    config = trl.GRPOConfig(
        **_supported_kwargs(
            trl.GRPOConfig,
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps or max(25, max_steps),
            save_total_limit=2,
            max_prompt_length=512,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            generation_kwargs=generation_kwargs,
            beta=beta,
            report_to=[],
            use_cpu=use_cpu,
            disable_tqdm=True if show_progress else None,
            optim="adamw_torch",
            seed=seed,
            model_init_kwargs={"torch_dtype": "auto"},
        )
    )

    callbacks: list = []
    if show_progress:
        callbacks.append(make_grpo_tqdm_callback(max_steps))
    trainer_holder: dict = {}
    if eval_every_n_steps and eval_pages_path:
        evaluator = _make_held_out_evaluator(
            trainer_holder=trainer_holder,
            output_dir=output_dir,
            eval_pages_path=eval_pages_path,
            similarity_model=similarity_model,
            sample_size=eval_pages,
            max_game_steps=eval_max_game_steps,
            chat_format=eval_chat_format,
            eval_num_generations=eval_num_generations,
            eval_batch_size=eval_batch_size,
            seed=seed,
        )
        callbacks.append(
            _build_periodic_eval_callback(
                every=eval_every_n_steps,
                evaluator=evaluator,
                label="eval",
            )
        )

    if dagger_every and eval_pages_path:
        callbacks.append(
            _build_dagger_callback(
                every=dagger_every,
                trainer_holder=trainer_holder,
                pages_path=eval_pages_path,
                similarity_model=similarity_model,
                chat_format=dagger_chat_format if dagger_chat_format is not None else eval_chat_format,
                bc_pages=dagger_pages,
                bc_rollout_steps=dagger_rollout_steps,
                bc_microsteps=dagger_microsteps,
                bc_batch_size=dagger_bc_batch_size,
                bc_history_max_steps=dagger_history_max_steps,
                rollout_num_return_sequences=eval_num_generations,
                seed=seed,
                output_dir=output_dir,
                oracle_mode=dagger_oracle_mode,
                oracle_top_k=dagger_oracle_top_k,
                oracle_temperature=dagger_oracle_temperature,
                oracle_min_idf=dagger_oracle_min_idf,
            )
        )

    trainer = trl.GRPOTrainer(
        model=model_arg,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks or None,
    )
    trainer_holder["trainer"] = trainer
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model(str(output_dir))
    if log_path:
        _write_trainer_log(trainer.state.log_history, log_path)
    if plot_path:
        _plot_trainer_reward(trainer.state.log_history, plot_path)


def download_hf_model(model_name: str, output_dir: str | Path | None = None) -> str:
    enforce_hf_cache()
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": model_name, "token": True if hf_token_available() else False}
    if output_dir is not None:
        kwargs["local_dir"] = str(output_dir)
    return snapshot_download(**kwargs)


def _feedback_step(
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    history: list[dict],
    guess: str,
) -> dict:
    guessed = {str(step.get("guess", "")) for step in history}
    reward = score_completion(
        title=page.title,
        intro=page.intro,
        history=history,
        completion=f"MOT: {guess}",
        similarity_model=similarity_model,
    )
    return {
        "guess": reward.guess,
        "exact": reward.exact_hits,
        "semantic": reward.semantic_hits,
        "title": reward.title_hits,
        "reward": reward.reward,
        "solved": reward.solved,
        "invalid": reward.invalid and reward.guess not in guessed,
        "repeated": reward.guess in guessed,
    }


def _starter_words(vocabulary: list[str], similarity_model: TinyPedantixModel) -> list[str]:
    words = []
    for word in THEME_SEEDS + similarity_model.starter_words + vocabulary:
        word = normalize_word(word)
        if word not in words and is_valid_guess(word):
            words.append(word)
        if len(words) >= 80:
            break
    return words


def _lora_config(peft, rank: int):
    return peft.LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def _best_device(torch) -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _import_training_stack():
    enforce_hf_cache()
    try:
        import datasets
        import peft
        import trl
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLM training dependencies. Install with: "
            "python3 -m pip install 'datasets>=2.19' 'trl>=0.18' 'peft>=0.11' 'accelerate>=0.30'"
        ) from exc
    return datasets, trl, peft


def _import_transformers():
    enforce_hf_cache()
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Missing transformers/torch dependencies for LLM inference") from exc
    return transformers, torch


def _write_trainer_log(log_history: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in log_history:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _supported_kwargs(cls, **kwargs):
    parameters = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def _qwen_no_think_generation_kwargs(tokenizer, model_name_or_path: str) -> dict:
    if "qwen3" not in str(model_name_or_path).lower():
        return {}
    bad_words = [
        "<think>",
        "</think>",
        "think",
        "Think",
        "thinking",
        "Thinking",
        "reasoning",
        "Reasoning",
        "raisonnement",
        "Raisonnement",
        "réflexion",
        "Réflexion",
    ]
    bad_words_ids = []
    for word in bad_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)
    return {"bad_words_ids": bad_words_ids} if bad_words_ids else {}


def _prepare_grpo_prompts(dataset, model_name_or_path: str):
    model_name = _tokenizer_name_for_model(model_name_or_path).lower()
    if "qwen3" not in model_name:
        return dataset

    def patch_row(row):
        prompt = str(row.get("prompt", ""))
        if "<|im_start|>user\n" in prompt and "<|im_end|>\n<|im_start|>assistant\n" in prompt:
            if "/no_think" not in prompt:
                prompt = prompt.replace(
                    "<|im_end|>\n<|im_start|>assistant\n",
                    "\n/no_think<|im_end|>\n<|im_start|>assistant\n",
                    1,
                )
            prompt = prompt.rstrip()
            if not prompt.endswith("MOT:"):
                prompt += "\nMOT:"
        row["prompt"] = prompt
        return row

    return dataset.map(patch_row, desc="Patch Qwen3 prompts")


def _plot_trainer_reward(log_history: list[dict], path: str | Path) -> None:
    reward_rows = [row for row in log_history if "reward" in row or "rewards/reward_func/mean" in row]
    if not reward_rows:
        return
    import matplotlib.pyplot as plt

    xs = [row.get("step", idx + 1) for idx, row in enumerate(reward_rows)]
    rewards = [row.get("reward", row.get("rewards/reward_func/mean", 0.0)) for row in reward_rows]
    kl = [row.get("kl", 0.0) for row in reward_rows]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_reward = plt.subplots(figsize=(9, 4.8))
    ax_reward.plot(xs, rewards, color="#2563eb", linewidth=2, label="reward")
    ax_reward.set_xlabel("training step")
    ax_reward.set_ylabel("reward")
    ax_reward.grid(alpha=0.25)
    ax_kl = ax_reward.twinx()
    ax_kl.plot(xs, kl, color="#9333ea", linewidth=1.6, label="kl")
    ax_kl.set_ylabel("kl")
    lines = ax_reward.get_lines() + ax_kl.get_lines()
    ax_reward.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def make_grpo_tqdm_callback(max_steps: int):
    from transformers import TrainerCallback

    class GRPOTqdmCallback(TrainerCallback):
        def __init__(self, *, max_steps: int) -> None:
            self.max_steps = max_steps
            self.pbar = None
            self.last_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            from tqdm.auto import tqdm

            total = self.max_steps if self.max_steps and self.max_steps > 0 else None
            self.pbar = tqdm(total=total, desc="GRPO", unit="step")
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.pbar is None:
                return control
            logs = logs or {}
            step = int(state.global_step)
            delta = max(0, step - self.last_step)
            if delta:
                self.pbar.update(delta)
                self.last_step = step
            postfix = {}
            for source, target in [
                ("reward", "reward"),
                ("rewards/reward_func/mean", "reward"),
                ("kl", "kl"),
                ("loss", "loss"),
                ("train_loss", "loss"),
                ("completions/mean_length", "len"),
                ("reward_std", "r_std"),
            ]:
                if source in logs and target not in postfix:
                    value = logs[source]
                    postfix[target] = f"{value:.4g}" if isinstance(value, float) else value
            if postfix:
                self.pbar.set_postfix(postfix)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self.pbar is not None:
                if state.global_step > self.last_step:
                    self.pbar.update(state.global_step - self.last_step)
                self.pbar.close()
                self.pbar = None
            return control

    return GRPOTqdmCallback(max_steps=max_steps)


def _build_periodic_eval_callback(
    *,
    every: int,
    evaluator,
    label: str = "eval",
):
    from transformers import TrainerCallback

    class _PeriodicEvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.every = max(1, int(every))
            self.last_eval = 0
            self.label = label

        def _run(self, state, step: int) -> None:
            if step <= 0 or step <= self.last_eval:
                return
            self.last_eval = step
            try:
                result = evaluator(step)
            except Exception as exc:
                print(f"[{self.label}@step{step}] failed: {exc!r}")
                return
            entry: dict = {"step": step, f"{self.label}/done": True}
            for key, value in result.items():
                if isinstance(value, (int, float, str, bool)):
                    entry[f"{self.label}/{key}"] = value
            state.log_history.append(entry)
            print(f"[{self.label}@step{step}] {json.dumps(entry, ensure_ascii=False)}")

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if self.every > 0 and step > 0 and step % self.every == 0:
                self._run(state, step)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            self._run(state, step)
            return control

    return _PeriodicEvalCallback()


def _dagger_bc_step(
    *,
    model,
    tokenizer,
    optimizer,
    prompts: list[str],
    completions: list[str],
    max_length: int = 512,
) -> float:
    """One supervised cross-entropy update on (prompt, completion) pairs.

    Loss is computed only on completion tokens (prompt tokens get label -100).
    Returns the float loss.
    """
    import torch

    if not prompts:
        return 0.0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = []
    for prompt, completion in zip(prompts, completions):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        c_ids = tokenizer.encode(completion, add_special_tokens=False)
        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids
        if len(ids) > max_length:
            overflow = len(ids) - max_length
            ids = ids[overflow:]
            labels = labels[overflow:]
        encoded.append((ids, labels))
    longest = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), longest), pad_id, dtype=torch.long)
    label_ids = torch.full((len(encoded), longest), -100, dtype=torch.long)
    attn = torch.zeros((len(encoded), longest), dtype=torch.long)
    for i, (ids, lab) in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        label_ids[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        attn[i, : len(ids)] = 1
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    label_ids = label_ids.to(device)
    attn = attn.to(device)

    was_training = model.training
    model.train()
    try:
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn, labels=label_ids)
        loss = out.loss
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
    finally:
        if not was_training:
            model.eval()
    return loss_value


def _dagger_collect_pairs(
    *,
    model,
    tokenizer,
    pages,
    similarity_model: TinyPedantixModel,
    history_max_steps: int,
    rollout_steps: int,
    chat_format: str,
    device: str,
    num_return_sequences: int,
    oracle_mode: str = "soft",
    oracle_top_k: int = 8,
    oracle_temperature: float = 1.0,
    oracle_min_idf: float = 0.0,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    """Roll out the current policy on `pages`; at every visited state, query the
    strong oracle and emit a (prompt_text_for_model, oracle_word + stop_token)
    pair. Returns deduped pairs (by (prompt_hash, oracle_word)).
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[int, str]] = set()
    histories: list[list[dict]] = [[] for _ in pages]
    done = [False for _ in pages]
    for _ in range(rollout_steps):
        active_idx = [i for i, d in enumerate(done) if not d]
        if not active_idx:
            break
        prompts_for_oracle: list[str] = []
        prompts_for_model: list[str] = []
        for i in active_idx:
            raw = make_prompt(
                histories[i],
                max_steps=history_max_steps,
                visible_text=compact_visible_text(
                    replay_game(pages[i], similarity_model, histories[i])
                ),
            )
            prompts_for_oracle.append(raw)
            prompts_for_model.append(format_prompt_for_model(raw, chat_format=chat_format))

        for i, raw_prompt, model_prompt in zip(active_idx, prompts_for_oracle, prompts_for_model):
            if oracle_mode == "soft":
                oracle_word = soft_oracle_guess(
                    pages[i], similarity_model, histories[i],
                    top_k=oracle_top_k, temperature=oracle_temperature,
                    min_idf=oracle_min_idf, rng=rng,
                )
            else:
                oracle_word = strong_oracle_guess(pages[i], similarity_model, histories[i])
            if not oracle_word or not is_valid_guess(oracle_word):
                continue
            key = (i, oracle_word)
            if key in seen:
                continue
            seen.add(key)
            completion = " " + oracle_word + "<|im_end|>"
            pairs.append((model_prompt, completion))

        # Advance histories with the model's actual guesses so the next state is
        # what the student would face on its own trajectory (this is the DAgger
        # property).
        guesses = generate_next_words(
            model,
            tokenizer,
            prompts_for_model,
            histories=[histories[i] for i in active_idx],
            device=device,
            num_return_sequences=num_return_sequences,
        )
        for i, guess in zip(active_idx, guesses):
            step = _feedback_step(pages[i], similarity_model, histories[i], guess)
            histories[i].append(step)
            if step["solved"]:
                done[i] = True
    return pairs


def _build_dagger_callback(
    *,
    every: int,
    trainer_holder: dict,
    pages_path: str | Path,
    similarity_model: TinyPedantixModel,
    chat_format: str,
    bc_pages: int,
    bc_rollout_steps: int,
    bc_microsteps: int,
    bc_batch_size: int,
    bc_history_max_steps: int,
    rollout_num_return_sequences: int,
    seed: int,
    output_dir: str | Path,
    oracle_mode: str = "soft",
    oracle_top_k: int = 8,
    oracle_temperature: float = 1.0,
    oracle_min_idf: float = 0.0,
):
    """Periodic DAgger callback: rolls out the student, labels with a strong
    oracle, and runs supervised BC microsteps on the in-memory trainer model.

    Designed for use alongside `_build_periodic_eval_callback` in train_llm_grpo.
    """
    from transformers import TrainerCallback

    output_root = Path(output_dir)
    log_path = output_root / "dagger_log.jsonl"

    class _DaggerCallback(TrainerCallback):
        def __init__(self) -> None:
            self.every = max(1, int(every))
            self.last_run = 0
            self.rng = random.Random(seed)

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step <= 0 or step <= self.last_run:
                return control
            if step % self.every != 0:
                return control
            self.last_run = step
            try:
                summary = self._cycle(step)
                print(f"[dagger@step{step}] {json.dumps(summary, ensure_ascii=False)}")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": step, **summary}, ensure_ascii=False) + "\n")
            except Exception as exc:  # don't kill training over a DAgger hiccup
                print(f"[dagger@step{step}] failed: {exc!r}")
            return control

        def _cycle(self, step: int) -> dict:
            import torch as _torch
            import gc as _gc

            trainer = trainer_holder.get("trainer")
            if trainer is None:
                raise RuntimeError("trainer holder is empty")
            model = trainer.model
            tokenizer = getattr(trainer, "processing_class", None) or getattr(
                trainer, "tokenizer", None
            )
            if tokenizer is None:
                raise RuntimeError("trainer has no tokenizer")
            optimizer = trainer.optimizer
            if optimizer is None:
                raise RuntimeError("trainer optimizer is not initialised yet")

            # Sample fresh pages each cycle to keep the policy generalising.
            pages = sample_pages(str(pages_path), sample_size=bc_pages, seed=self.rng.randrange(1 << 30))
            device = str(next(model.parameters()).device)

            was_training = model.training
            model.eval()
            try:
                with _torch.inference_mode():
                    pairs = _dagger_collect_pairs(
                        model=model,
                        tokenizer=tokenizer,
                        pages=pages,
                        similarity_model=similarity_model,
                        history_max_steps=bc_history_max_steps,
                        rollout_steps=bc_rollout_steps,
                        chat_format=chat_format,
                        device=device,
                        num_return_sequences=rollout_num_return_sequences,
                        oracle_mode=oracle_mode,
                        oracle_top_k=oracle_top_k,
                        oracle_temperature=oracle_temperature,
                        oracle_min_idf=oracle_min_idf,
                        rng=self.rng,
                    )
            finally:
                if was_training:
                    model.train()
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

            if not pairs:
                return {"pairs": 0, "microsteps": 0, "bc_loss_first": None, "bc_loss_last": None}

            self.rng.shuffle(pairs)
            losses: list[float] = []
            for s in range(bc_microsteps):
                start = (s * bc_batch_size) % max(1, len(pairs))
                batch = pairs[start : start + bc_batch_size]
                if len(batch) < bc_batch_size:
                    batch = (pairs + pairs)[start : start + bc_batch_size]
                prompts = [p for p, _ in batch]
                completions = [c for _, c in batch]
                losses.append(
                    _dagger_bc_step(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        prompts=prompts,
                        completions=completions,
                    )
                )

            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            return {
                "pages": len(pages),
                "pairs": len(pairs),
                "microsteps": len(losses),
                "bc_loss_first": round(losses[0], 4) if losses else None,
                "bc_loss_last": round(losses[-1], 4) if losses else None,
                "bc_loss_mean": round(sum(losses) / len(losses), 4) if losses else None,
            }

    return _DaggerCallback()


def _make_held_out_evaluator(
    *,
    trainer_holder: dict,
    output_dir: str | Path,
    eval_pages_path: str | Path,
    similarity_model: TinyPedantixModel,
    sample_size: int,
    max_game_steps: int,
    chat_format: str,
    eval_num_generations: int,
    eval_batch_size: int,
    seed: int,
):
    """Return a callable(step) -> eval-result dict that runs game simulations
    against the live in-memory trainer model (avoids loading a second copy)."""

    output_root = Path(output_dir)

    def _evaluate(step: int) -> dict:
        trainer = trainer_holder.get("trainer")
        if trainer is None:
            raise RuntimeError("trainer holder is empty when eval callback fired")
        model = trainer.model
        tokenizer = getattr(trainer, "processing_class", None) or getattr(
            trainer, "tokenizer", None
        )
        if tokenizer is None:
            raise RuntimeError("trainer has no tokenizer/processing_class")
        eval_log = output_root / "eval_logs" / f"step_{step}.jsonl"
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        was_training = model.training
        model.eval()
        import gc as _gc
        import torch as _torch
        # Release training's cached-but-unallocated VRAM before generate() needs
        # large contiguous KV-cache buffers. Without this, eval reliably OOMs
        # at ~step 500 on a 24GB GPU even with no_grad active.
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        try:
            device = str(next(model.parameters()).device)
            inference_ctx = getattr(_torch, "inference_mode", _torch.no_grad)
            with inference_ctx():
                result = _run_eval_games(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    pages_path=str(eval_pages_path),
                    similarity_model=similarity_model,
                    sample_size=sample_size,
                    max_steps=max_game_steps,
                    seed=seed,
                    output_path=str(eval_log),
                    chat_format=chat_format,
                    generation_batch_size=eval_batch_size,
                    eval_num_generations=eval_num_generations,
                )
        finally:
            if was_training:
                model.train()
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        return result

    return _evaluate


def _tokenizer_name_for_model(model_name_or_path: str) -> str:
    adapter_config = Path(model_name_or_path) / "adapter_config.json"
    if not adapter_config.exists():
        return model_name_or_path
    data = json.loads(adapter_config.read_text(encoding="utf-8"))
    return data.get("base_model_name_or_path") or model_name_or_path


def _load_peft_model_if_adapter(model_name_or_path: str):
    enforce_hf_cache()
    adapter_config = Path(model_name_or_path) / "adapter_config.json"
    if not adapter_config.exists():
        return model_name_or_path
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    config = PeftConfig.from_pretrained(model_name_or_path)
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype="auto")
    return PeftModel.from_pretrained(base, model_name_or_path, is_trainable=True)


def _load_model_and_tokenizer_for_inference(model_path: str, transformers, torch):
    enforce_hf_cache()
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        base = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype="auto")
        return PeftModel.from_pretrained(base, model_path), tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, token=hf_token_available())
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, token=hf_token_available(), torch_dtype="auto")
    return model, tokenizer
