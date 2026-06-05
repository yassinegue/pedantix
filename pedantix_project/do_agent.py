"""DigitalOcean GenAI inference agent for Pedantix trajectory sampling.

Uses DigitalOcean's serverless inference (OpenAI-compatible API at
https://inference.do-ai.run/v1/) to play Pedantix games and generate
a comprehensive SFT dataset with varied starting states:

  - Full games from scratch (warm_start_steps=0)
  - Mid-game starts: first N steps played by the local oracle, then the
    LLM agent takes over (warm_start_steps=N)

This produces training examples covering both early-game topic discovery
and late-game title narrowing — the full strategic range.
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .dataset import WikiPage
from .llm_policy import (
    _apply_llm_exact_reveals,
    _feedback_step,
    _is_real_french_word,
    compact_visible_text,
    compute_page_max_sim,
    extract_guess,
    is_valid_guess,
    make_prompt,
    score_guess_on_game,
    soft_oracle_guess,
)
from .model import TinyPedantixModel
from .simulator import PedantixGame

_DO_BASE_URL = "https://inference.do-ai.run/v1/"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1/"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/"
_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1/"
_CF_ACCOUNT_ID_ENV = "CF_ACCOUNT_ID"  # env var fallback; base URL built at runtime
_DEFAULT_MODEL = "llama3.3-70b-instruct"

_SYSTEM_PROMPT_NO_COT = (
    "Tu es un expert Wikipedia francais. Tu dois deviner le titre d'un article en proposant des mots un par un. "
    "Reponds UNIQUEMENT avec 'MOT: <mot>' en minuscules. Un seul mot, rien d'autre."
)

_SYSTEM_PROMPT_COT = (
    "Tu es un expert Wikipedia francais. Tu dois deviner le titre d'un article en proposant des mots un par un.\n\n"
    "Avant chaque mot, reflechis brievement a ta strategie entre balises <think>...</think> :\n"
    "- Que revelent les scores des essais precedents sur le domaine ?\n"
    "- Quels mots specifiques pourraient figurer dans le titre ?\n"
    "- Quel mot non encore essaye a le plus de chances ?\n\n"
    "Format de reponse OBLIGATOIRE :\n"
    "<think>\n[ta reflexion strategique ici]\n</think>\n"
    "MOT: <mot>"
)

_SYSTEM_PROMPT = _SYSTEM_PROMPT_NO_COT  # default kept for backward compat


def _build_openai_client(api_key: str, provider: str = "do", cf_account_id: str = "", timeout: float | None = None):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")
    if provider == "groq":
        return OpenAI(base_url=_GROQ_BASE_URL, api_key=api_key, timeout=timeout or 20.0)
    if provider == "cerebras":
        return OpenAI(base_url=_CEREBRAS_BASE_URL, api_key=api_key, timeout=timeout or 30.0)
    if provider == "fireworks":
        return OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key, timeout=timeout or 60.0)
    if provider == "cloudflare":
        if not cf_account_id:
            import os
            cf_account_id = os.environ.get(_CF_ACCOUNT_ID_ENV, "")
        if not cf_account_id:
            raise ValueError("Cloudflare account ID required: --cf-account-id or CF_ACCOUNT_ID env var")
        url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/v1/"
        return OpenAI(base_url=url, api_key=api_key, timeout=timeout or 30.0)
    return OpenAI(base_url=_DO_BASE_URL, api_key=api_key, timeout=timeout or 90.0)


def _ask_llm(
    client,
    model: str,
    prompt: str,
    guessed: set[str],
    retry_delay: float,
    max_retries: int,
    request_delay: float = 1.0,
    max_tokens: int = 50,
    system_prompt: str = _SYSTEM_PROMPT_NO_COT,
) -> tuple[str, str] | None:
    """Call the LLM and return (raw_response, guess) or None if it can't.

    raw_response is the full model output including any <think>...</think> block.
    Returns None when the model repeatedly outputs already-guessed words after
    re-asking — the caller must skip that step (not score or record it).
    """
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        time.sleep(request_delay)
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit:
                try:
                    wait = int(err.split("Retry-After:")[-1].split()[0])
                except Exception:
                    wait = 30
            else:
                wait = retry_delay * (2 ** attempt)
            if attempt < max_retries:
                time.sleep(wait)
                continue
            return None

        guess = extract_guess(raw)
        if not guess:
            return None
        if guess not in guessed:
            return raw, guess
        banned = ", ".join(sorted(guessed))
        current_prompt = (
            prompt
            + f"\n\nATTENTION: '{guess}' est deja dans la liste des mots interdits."
            f"\nMots interdits: {banned}"
            f"\nTu DOIS proposer un mot DIFFERENT."
        )
    return None


def _build_cot_completion(raw: str, guess: str, eos_token: str) -> str:
    """Build a completion string preserving any thinking block from raw.

    Handles two formats from the API:
      - Proper:    <think>...</think>MOT: word
      - Truncated: thinking...</think>MOT: word  (DeepSeek-v4-pro drops the '<')

    Output is always: <think>\n...\n</think>\nMOT: word<eos>
    Falls back to: MOT: word<eos> if no thinking found.
    """
    import re
    # Try proper format first
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if not m:
        # DeepSeek-v4-pro emits "thinking..." without the opening "<"
        m = re.search(r"thinking(.*?)</think>", raw, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        if thinking:
            return f"<think>\n{thinking}\n</think>\nMOT: {guess}{eos_token}"
    return f"MOT: {guess}{eos_token}"


def _build_training_prompt(prompt: str, enable_cot: bool = False, qwen_format: bool = False) -> str:
    """Wrap game-state prompt in the format expected at SFT/GRPO training time.

    CoT mode: Qwen3 chat format with thinking enabled (no /no_think, no MOT: prefix).
      prompt → <|im_start|>user\\n{body}<|im_end|>\\n<|im_start|>assistant\\n

    qwen_format (non-CoT GRPO): adds /no_think + MOT: prefix so model outputs word tokens.
      prompt ��� <|im_start|>user\\n{body}\\n/no_think<|im_end|>\\n<|im_start|>assistant\\nMOT:

    Plain (default): raw prompt unchanged (format_prompt_for_model wraps at eval time).
    """
    if enable_cot:
        body = prompt.rsplit("\nMOT:", 1)[0]
        return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
    if qwen_format:
        body = prompt.rsplit("\nMOT:", 1)[0]
        return f"<|im_start|>user\n{body}\n/no_think<|im_end|>\n<|im_start|>assistant\nMOT:"
    return prompt


def _split_prompt_for_cache(prompt: str) -> tuple[str, str]:
    """Split prompt into (static_rules, dynamic_game_state) at 'Essais restants:'."""
    marker = "\nEssais restants:"
    idx = prompt.find(marker)
    if idx == -1:
        return "", prompt
    return prompt[:idx], prompt[idx:]


def _ask_anthropic(
    client,
    model: str,
    prompt: str,
    guessed: set[str],
    retry_delay: float,
    max_retries: int,
    request_delay: float = 0.1,
) -> str | None:
    """Same contract as _ask_llm but using the Anthropic Messages API with prompt caching."""
    static_part, dynamic_part = _split_prompt_for_cache(prompt)
    system_block = [{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    def _build_content(extra: str = "") -> list:
        blocks = []
        if static_part:
            blocks.append({"type": "text", "text": static_part, "cache_control": {"type": "ephemeral"}})
        blocks.append({"type": "text", "text": dynamic_part + extra})
        return blocks

    extra = ""
    for attempt in range(max_retries + 1):
        time.sleep(request_delay)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=50,
                system=system_block,
                messages=[{"role": "user", "content": _build_content(extra)}],
            )
            raw = (msg.content[0].text if msg.content else "").strip()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "overloaded" in err.lower()
            if is_rate_limit:
                try:
                    wait = int(err.split("retry-after:")[-1].split()[0])
                except Exception:
                    wait = 30
            else:
                wait = retry_delay * (2 ** attempt)
            if attempt < max_retries:
                time.sleep(wait)
                continue
            return None

        guess = extract_guess(raw)
        if not guess:
            return None
        if guess not in guessed:
            return guess
        banned = ", ".join(sorted(guessed))
        extra = (
            f"\n\nATTENTION: '{guess}' est deja dans la liste des mots interdits."
            f"\nMots interdits: {banned}"
            f"\nTu DOIS proposer un mot DIFFERENT."
        )
    return None


def _oracle_warm_start(
    page: WikiPage,
    sim_model: TinyPedantixModel,
    game: PedantixGame,
    history: list[dict],
    guessed: set[str],
    *,
    n_steps: int,
    rng: random.Random,
) -> None:
    """Play n_steps using the soft oracle to reach a non-trivial game state.

    Mutates game, history, and guessed in-place. These steps are NOT written
    as training examples — they are context only.
    """
    for _ in range(n_steps):
        if game.solved:
            break
        word = soft_oracle_guess(
            page,
            sim_model,
            history,
            top_k=8,
            temperature=0.5,
            min_idf=0.2,
            rng=rng,
        )
        if not word or word in guessed or not is_valid_guess(word):
            break
        score_guess_on_game(game, word, history_len=len(history), guessed=guessed)
        _apply_llm_exact_reveals(game, word)
        guessed.add(word)
        history.append(_feedback_step(page, sim_model, history, word))


def play_game_with_do_agent(
    page: WikiPage,
    sim_model: TinyPedantixModel,
    *,
    api_key: str,
    model: str = _DEFAULT_MODEL,
    max_steps: int = 30,
    warm_start_steps: int = 0,
    seed: int | None = None,
    retry_delay: float = 3.0,
    max_retries: int = 3,
    request_delay: float = 0.2,
    provider: str = "do",
    signal_model=None,
    cf_account_id: str = "",
    enable_cot: bool = False,
    eos_token: str = "<|im_end|>",
    grpo_format: bool = False,
) -> dict[str, Any]:
    """Play one game using the DigitalOcean LLM agent.

    warm_start_steps > 0: first run the local oracle for that many steps
    (building a realistic mid-game state), then let the LLM take over.

    Returns: {title, solved, steps, warm_start_steps, history}
    history entries have keys: guess, exact, semantic, title, reward, solved,
    prompt, completion (only for LLM-controlled steps).
    """
    rng = random.Random(seed)

    system_prompt = _SYSTEM_PROMPT_COT if enable_cot else _SYSTEM_PROMPT_NO_COT
    # CoT needs more tokens; thinking models need even more
    _thinking_model_patterns = ("deepseek-r1", "deepseek_r1", "qwq", "kimi", "o1-")
    _is_thinking_model = any(p in model.lower() for p in _thinking_model_patterns)
    _thinking_providers = {"fireworks", "cerebras"}

    use_anthropic = provider == "anthropic"
    if use_anthropic:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        client = _anthropic.Anthropic(api_key=api_key)
    else:
        client = _build_openai_client(api_key, provider=provider, cf_account_id=cf_account_id)
        if provider in _thinking_providers or _is_thinking_model:
            llm_max_tokens = 2000
        elif enable_cot:
            llm_max_tokens = 600  # think block (2-4 sentences) + word
        else:
            llm_max_tokens = 50

    game = PedantixGame(page, similarity_model=sim_model)
    history: list[dict] = []
    guessed: set[str] = set()

    if warm_start_steps > 0:
        _oracle_warm_start(
            page, sim_model, game, history, guessed,
            n_steps=warm_start_steps, rng=rng,
        )

    llm_steps = 0
    skip_streak = 0
    for _ in range(max_steps - len(history)):
        if game.solved:
            break

        visible = compact_visible_text(game)
        prompt = make_prompt(history, max_steps=max_steps, visible_text=visible)

        if use_anthropic:
            result_llm = _ask_anthropic(client, model, prompt, guessed, retry_delay, max_retries, request_delay)
            raw, guess = (None, result_llm) if result_llm is not None else (None, None)
        else:
            result_llm = _ask_llm(client, model, prompt, guessed, retry_delay, max_retries, request_delay, llm_max_tokens, system_prompt)
            if result_llm is None:
                raw, guess = None, None
            else:
                raw, guess = result_llm
        if guess is None:
            skip_streak += 1
            if skip_streak >= 5:
                break  # model is stuck — terminate game
            continue
        skip_streak = 0

        score = compute_page_max_sim(game, guess, signal_model=signal_model)
        step_result = score_guess_on_game(
            game, guess, history_len=len(history), guessed=guessed
        )
        _apply_llm_exact_reveals(game, guess)
        guessed.add(guess)

        completion = (
            _build_cot_completion(raw, guess, eos_token)
            if (enable_cot and raw is not None)
            else f"MOT: {guess}{eos_token}"
        )
        history.append(
            {
                "guess": guess,
                "exact": step_result.exact_hits,
                "score": score,
                "semantic": step_result.semantic_hits,
                "title": step_result.title_hits,
                "reward": round(step_result.reward, 4),
                "solved": step_result.solved,
                "prompt": _build_training_prompt(prompt, enable_cot=enable_cot, qwen_format=grpo_format),
                "completion": completion,
            }
        )
        llm_steps += 1

    return {
        "title": page.title,
        "solved": game.solved,
        "steps": len(history),
        "warm_start_steps": warm_start_steps,
        "llm_steps": llm_steps,
        "history": history,
        "cot": enable_cot,
    }


def generate_do_trajectories(
    pages_path: Path,
    sim_model: TinyPedantixModel,
    *,
    api_key: str,
    model: str = _DEFAULT_MODEL,
    n_pages: int = 1000,
    max_steps: int = 30,
    out_jsonl: Path,
    seed: int = 42,
    eos_token: str = "<|im_end|>",
    warm_start_schedule: list[int] | None = None,
    signal_model=None,
    workers: int = 1,
    request_delay: float = 0.2,
    provider: str = "do",
    verbose: bool = True,
    cf_account_id: str = "",
    enable_cot: bool = False,
    grpo_format: bool = False,
) -> int:
    """Generate SFT training examples from DigitalOcean LLM gameplay.

    warm_start_schedule: list of warm-start step counts to cycle through.
    Default: [0, 0, 10, 10, 20] — mostly from scratch, some mid-game starts.

    Each game step where the LLM was in control becomes one training example:
      {"prompt": "<game state>", "completion": "MOT: word<|im_end|>"}

    Only steps where the LLM guessed a valid French word are kept.
    Returns the total number of training examples written.
    """
    if warm_start_schedule is None:
        warm_start_schedule = [5, 10, 10, 15, 20]

    rng = random.Random(seed)

    pages: list[WikiPage] = []
    with open(pages_path) as f:
        pool = [json.loads(line) for line in f]
    rng.shuffle(pool)
    for p in pool[:n_pages]:
        pages.append(WikiPage(title=p["title"], intro=p["intro"]))

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip pages already in the output file
    already_done: set[str] = set()
    existing_examples = 0
    if out_jsonl.exists():
        with open(out_jsonl) as _f:
            for _line in _f:
                try:
                    _r = json.loads(_line)
                    if "page_title" in _r:
                        already_done.add(_r["page_title"])
                    existing_examples += 1
                except Exception:
                    pass
        if already_done:
            print(f"Resuming: {len(already_done)} pages already done ({existing_examples} examples), skipping.", flush=True)

    n_examples = existing_examples
    n_filtered = 0
    solve_count = 0
    import threading
    write_lock = threading.Lock()

    warm_starts = [warm_start_schedule[i % len(warm_start_schedule)] for i in range(n_pages)]

    def _run_one(args):
        i, page, warm_start = args
        return i, page, warm_start, play_game_with_do_agent(
            page, sim_model,
            api_key=api_key,
            model=model,
            max_steps=max_steps,
            warm_start_steps=warm_start,
            seed=seed + i,
            signal_model=signal_model,
            request_delay=request_delay,
            provider=provider,
            cf_account_id=cf_account_id,
            enable_cot=enable_cot,
            eos_token=eos_token,
            grpo_format=grpo_format,
        )

    with open(out_jsonl, "a") as fout:
        tasks = [
            (i, page, ws)
            for i, (page, ws) in enumerate(zip(pages, warm_starts))
            if page.title not in already_done
        ]
        if len(tasks) < len(pages):
            print(f"Skipping {len(pages) - len(tasks)} already-done pages, {len(tasks)} remaining.", flush=True)

        def _process(i, page, warm_start, game_result):
            nonlocal n_examples, n_filtered, solve_count
            solved = game_result["solved"]
            sim_game = PedantixGame(page, similarity_model=sim_model)
            page_examples = 0
            seen_guesses: set[str] = set()
            rows = []
            full_history = game_result["history"]
            for step_idx, step in enumerate(full_history):
                if "prompt" not in step or not step["prompt"]:
                    seen_guesses.add(step.get("guess", ""))
                    continue
                guess = step["guess"]
                if guess in seen_guesses:
                    continue
                seen_guesses.add(guess)
                if not _is_real_french_word(guess, sim_game):
                    continue
                # CoT mode: discard examples where the model skipped reasoning
                if enable_cot and "<think>" not in step.get("completion", ""):
                    n_filtered += 1
                    continue
                step_score = step.get("score", None)
                row: dict = {
                    "prompt": step["prompt"],
                    "completion": step["completion"],
                    "page_title": page.title,
                }
                if step_score is not None:
                    row["score"] = int(step_score)
                if grpo_format:
                    # GRPO reward function needs these to score completions
                    row["title"] = page.title
                    row["intro"] = page.intro
                    # history = guesses made BEFORE this step
                    prev = [
                        {"guess": s["guess"], "exact": s.get("exact", 0),
                         "semantic": s.get("semantic", 0), "score": s.get("score", 0)}
                        for s in full_history[:step_idx]
                    ]
                    row["history"] = json.dumps(prev, ensure_ascii=False)
                rows.append(json.dumps(row, ensure_ascii=False))
                page_examples += 1
            with write_lock:
                if solved:
                    solve_count += 1
                n_filtered += (len([s for s in game_result["history"] if "prompt" in s and s["prompt"]])
                               - page_examples)
                n_examples += page_examples
                for row in rows:
                    fout.write(row + "\n")
                fout.flush()
                if verbose:
                    status = "SOLVED" if solved else f"step{game_result['steps']}"
                    print(f"[{i+1}/{n_pages}] {page.title!r} warm={warm_start} "
                          f"{status} examples={page_examples} total={n_examples}", flush=True)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_run_one, t): t for t in tasks}
                for fut in as_completed(futs):
                    try:
                        i, page, warm_start, game_result = fut.result()
                        _process(i, page, warm_start, game_result)
                    except Exception as e:
                        print(f"[worker error, skipping page] {e}", flush=True)
                        continue
        else:
            for t in tasks:
                try:
                    i, page, warm_start, game_result = _run_one(t)
                    _process(i, page, warm_start, game_result)
                except Exception as e:
                    print(f"[worker error, skipping page] {e}", flush=True)
                    continue

    if verbose or True:
        print(
            f"\nDone: {n_examples} training examples from {n_pages} games "
            f"(solved={solve_count}, filtered={n_filtered})"
        )
    return n_examples
