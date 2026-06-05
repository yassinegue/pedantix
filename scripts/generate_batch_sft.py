#!/usr/bin/env python3
"""Batch SFT data generation via DigitalOcean batch inference API.

Instead of playing games sequentially (1 LLM call per step = slow), we:

  1. Oracle plays N games locally (instant, no API).
  2. Snapshot every game state (compact visible text + history).
  3. Submit all ~15k prompts as ONE batch to DO /v1/batches.
  4. DO processes async; we poll until done.
  5. Parse completions → filter valid French words → write SFT JSONL.

This replaces sequential LLM play (~20s/call × 30 × 500 = 83 hours) with a
single async batch job that completes in 1-4 hours at 50% cheaper pricing.

Usage:
  python scripts/generate_batch_sft.py \
    --pages data/filtered_pages.jsonl \
    --tiny-model models/tiny_model.json \
    --n-pages 500 \
    --max-steps 30 \
    --model openai-gpt-5-nano \
    --output data/sft_batch_500.jsonl \
    --seed 42

Phases can be run separately:
  --phase generate   # oracle play → saves states to --states-file
  --phase submit     # upload + create batch job → saves batch-id
  --phase poll       # wait for completion
  --phase process    # download results → SFT JSONL
  --phase all        # end to end (default)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pedantix_project.dataset import WikiPage
from pedantix_project.llm_policy import (
    _apply_llm_exact_reveals,
    _feedback_step,
    _is_real_french_word,
    compact_visible_text,
    extract_guess,
    is_valid_guess,
    make_prompt,
    score_guess_on_game,
    soft_oracle_guess,
)
from pedantix_project.model import TinyPedantixModel
from pedantix_project.simulator import PedantixGame

_SYSTEM = (
    "Tu es un expert Wikipedia francais. Tu dois deviner le titre d'un article "
    "en proposant des mots un par un. "
    "Reponds UNIQUEMENT avec 'MOT: <mot>' en minuscules. Un seul mot, rien d'autre."
)


# ---------------------------------------------------------------------------
# Phase 1: oracle play → game states
# ---------------------------------------------------------------------------

def generate_oracle_sft(
    pages: list[WikiPage],
    sim_model: TinyPedantixModel,
    *,
    max_steps: int = 30,
    seed: int = 42,
    eos_token: str = "<|im_end|>",
    verbose: bool = True,
) -> list[dict]:
    """Play all games with the soft oracle, recording (prompt, completion) pairs directly.

    The oracle picks the highest-IDF relevant word at each step from the article text.
    For 'Napoléon Bonaparte' it picks 'Bonaparte', 'Napoléon', 'Corse', etc.
    No API needed — instant generation of high-quality SFT data.

    Returns list of {"prompt": ..., "completion": "MOT: word<eos>"} dicts.
    """
    rng = random.Random(seed)
    examples = []
    solve_count = 0

    for game_idx, page in enumerate(pages):
        game = PedantixGame(page, similarity_model=sim_model)
        history: list[dict] = []
        guessed: set[str] = set()
        page_examples = 0

        for step_idx in range(max_steps):
            if game.solved:
                break

            # Oracle picks next word
            word = soft_oracle_guess(
                page, sim_model, history,
                top_k=8, temperature=0.3, min_idf=0.3, rng=rng,
            )
            if not word or word in guessed or not is_valid_guess(word):
                break

            # Snapshot state BEFORE the oracle's move → (state, oracle_word) = training pair
            visible = compact_visible_text(game)
            prompt = make_prompt(history, max_steps=max_steps, visible_text=visible)

            score_guess_on_game(game, word, history_len=len(history), guessed=guessed)
            _apply_llm_exact_reveals(game, word)
            guessed.add(word)
            history.append(_feedback_step(page, sim_model, history, word))

            if _is_real_french_word(word, game):
                examples.append({
                    "prompt": prompt,
                    "completion": f"MOT: {word}{eos_token}",
                })
                page_examples += 1

        if game.solved:
            solve_count += 1
        if verbose and (game_idx + 1) % 100 == 0:
            print(f"  [{game_idx+1}/{len(pages)}] {len(examples)} examples, {solve_count} solved")

    if verbose:
        print(f"Done: {len(pages)} games → {len(examples)} examples, {solve_count} solved ({100*solve_count/len(pages):.0f}%)")
    return examples


def generate_oracle_states(
    pages: list[WikiPage],
    sim_model: TinyPedantixModel,
    *,
    max_steps: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """Play all games with soft oracle, snapshot each state for LLM annotation."""
    rng = random.Random(seed)
    states = []

    for game_idx, page in enumerate(pages):
        game = PedantixGame(page, similarity_model=sim_model)
        history: list[dict] = []
        guessed: set[str] = set()

        for step_idx in range(max_steps):
            if game.solved:
                break
            visible = compact_visible_text(game)
            prompt = make_prompt(history, max_steps=max_steps, visible_text=visible)
            states.append({"game_idx": game_idx, "step_idx": step_idx, "title": page.title, "prompt": prompt})
            word = soft_oracle_guess(page, sim_model, history, top_k=8, temperature=0.3, min_idf=0.3, rng=rng)
            if not word or word in guessed or not is_valid_guess(word):
                break
            score_guess_on_game(game, word, history_len=len(history), guessed=guessed)
            _apply_llm_exact_reveals(game, word)
            guessed.add(word)
            history.append(_feedback_step(page, sim_model, history, word))

        if verbose and (game_idx + 1) % 50 == 0:
            print(f"  oracle: {game_idx + 1}/{len(pages)} games, {len(states)} states so far")

    if verbose:
        print(f"Oracle done: {len(pages)} games → {len(states)} states")
    return states


# ---------------------------------------------------------------------------
# Phase 2: build batch JSONL + upload + submit
# ---------------------------------------------------------------------------

def build_batch_jsonl(states: list[dict], model: str) -> str:
    """Return batch JSONL string (one request per line)."""
    lines = []
    for s in states:
        lines.append(json.dumps({
            "custom_id": f"g{s['game_idx']}_s{s['step_idx']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": s["prompt"]},
                ],
                "max_tokens": 30,
                "temperature": 0.5,
            },
        }, ensure_ascii=False))
    return "\n".join(lines)


def submit_batch(api_key: str, batch_jsonl: str, model: str, description: str = "pedantix-sft") -> dict:
    """Upload batch file and create batch job. Returns batch dict with 'id'.

    DO Batch Inference API (3-step flow):
      1. POST /v1/batches/files  with {"file_name": "..."} → {file_id, upload_url}
      2. PUT <upload_url>  with raw JSONL bytes
      3. POST /v1/batches  with {file_id, provider, completion_window, request_id, endpoint}
    """
    import uuid
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base = "https://inference.do-ai.run/v1"
    raw = batch_jsonl.encode("utf-8")

    # Step 1: create file intent → get presigned upload URL
    print(f"Step 1: creating file intent ({len(raw)/1024:.0f} KB) …")
    resp = requests.post(
        f"{base}/batches/files",
        headers=headers,
        json={"file_name": "batch_requests.jsonl"},
        timeout=30,
    )
    resp.raise_for_status()
    file_data = resp.json()
    file_id = file_data["file_id"]
    upload_url = file_data["upload_url"]
    print(f"  file_id={file_id}")

    # Step 2: PUT raw JSONL to presigned URL
    print("Step 2: uploading JSONL …")
    put_resp = requests.put(upload_url, data=raw, timeout=300)
    put_resp.raise_for_status()
    print("  upload OK")

    # Step 3: create batch job
    provider = "openai" if model.startswith("openai-") else "anthropic"
    request_id = str(uuid.uuid4())
    print(f"Step 3: creating batch job (provider={provider}, model={model}) …")
    resp = requests.post(
        f"{base}/batches",
        headers=headers,
        json={
            "file_id": file_id,
            "provider": provider,
            "completion_window": "24h",
            "request_id": request_id,
            "endpoint": "/v1/chat/completions",
        },
        timeout=60,
    )
    resp.raise_for_status()
    batch_data = resp.json()
    batch_id = batch_data.get("id")
    status = batch_data.get("status")
    print(f"Batch created: {batch_id}  status={status}")
    return {"id": batch_id, "file_id": file_id, "request_id": request_id, "status": status}


# ---------------------------------------------------------------------------
# Phase 3: poll until done
# ---------------------------------------------------------------------------

def poll_batch(api_key: str, batch_id: str, poll_interval: int = 60) -> str:
    """Poll until batch completes. Returns output_file_id."""
    import requests
    headers = {"Authorization": f"Bearer {api_key}"}
    base = "https://inference.do-ai.run/v1"

    while True:
        resp = requests.get(f"{base}/batches/{batch_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        counts = data.get("request_counts", {})
        print(
            f"[{time.strftime('%H:%M:%S')}] status={status}  "
            f"completed={counts.get('completed', '?')}  "
            f"failed={counts.get('failed', '?')}  "
            f"total={counts.get('total', '?')}"
        )
        if status == "completed":
            return data.get("output_file_id") or data.get("output_file", {}).get("id")
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status={status}\n{data}")
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Phase 4: download results + write SFT JSONL
# ---------------------------------------------------------------------------

def process_results(
    api_key: str,
    output_file_id: str,
    states: list[dict],
    sim_model: TinyPedantixModel,
    out_path: Path,
    eos_token: str = "<|im_end|>",
) -> int:
    """Download batch results and write SFT training examples."""
    import requests
    headers_r = {"Authorization": f"Bearer {api_key}"}
    base = "https://inference.do-ai.run/v1"

    print(f"Downloading results for file {output_file_id} …")
    resp = requests.get(f"{base}/batches/files/{output_file_id}/content", headers=headers_r, timeout=300)
    resp.raise_for_status()
    content = resp.text

    # Index results by custom_id
    results: dict[str, str] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("custom_id", "")
        body = row.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if choices:
            raw = (choices[0].get("message", {}).get("content") or "").strip()
            results[cid] = raw

    # Build a dummy game per page title for _is_real_french_word
    # (needs a PedantixGame with the right similarity_model)
    title_to_dummy: dict[str, PedantixGame] = {}
    state_map = {f"g{s['game_idx']}_s{s['step_idx']}": s for s in states}

    n_written = 0
    n_filtered = 0
    seen_per_game: dict[int, set[str]] = {}

    with open(out_path, "w") as fout:
        for cid, raw in results.items():
            s = state_map.get(cid)
            if s is None:
                continue
            guess = extract_guess(raw)
            if not guess:
                n_filtered += 1
                continue
            # Dedup per game
            game_idx = s["game_idx"]
            seen = seen_per_game.setdefault(game_idx, set())
            if guess in seen:
                n_filtered += 1
                continue
            seen.add(guess)
            # Validate: real French word
            title = s["title"]
            if title not in title_to_dummy:
                page = WikiPage(title=title, intro="")
                title_to_dummy[title] = PedantixGame(page, similarity_model=sim_model)
            if not _is_real_french_word(guess, title_to_dummy[title]):
                n_filtered += 1
                continue
            fout.write(json.dumps({
                "prompt": s["prompt"],
                "completion": f"MOT: {guess}{eos_token}",
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"SFT examples written: {n_written}  filtered: {n_filtered}")
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--tiny-model", required=True)
    ap.add_argument("--n-pages", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--model", default="openai-gpt-5-nano")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--states-file", default=None, help="Cache oracle states (JSON)")
    ap.add_argument("--batch-info-file", default=None, help="Cache batch id/file_id (JSON)")
    ap.add_argument("--phase", choices=["oracle", "generate", "submit", "poll", "process", "all"], default="oracle")
    ap.add_argument("--eos-token", default="<|im_end|>")
    ap.add_argument("--api-key", default=None, help="DO Model Access Key (or set BATCH_INFERENCE_KEY env var)")
    args = ap.parse_args()

    import os
    api_key = args.api_key or os.environ.get("BATCH_INFERENCE_KEY", "") or os.environ.get("DO_INFERENCE_KEY", "")
    if not api_key and args.phase in ("submit", "poll", "process", "all"):
        sys.exit("Set BATCH_INFERENCE_KEY (Model Access Key from DO console) or pass --api-key")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    states_file = Path(args.states_file) if args.states_file else out_path.with_suffix(".states.json")
    batch_info_file = Path(args.batch_info_file) if args.batch_info_file else out_path.with_suffix(".batch.json")

    # ---- Phase: oracle (pure local, no API) ----
    if args.phase == "oracle":
        print(f"Loading model from {args.tiny_model} …")
        sim_model = TinyPedantixModel.load(args.tiny_model)
        print(f"Loading {args.n_pages} pages from {args.pages} …")
        rng = random.Random(args.seed)
        pool = []
        with open(args.pages) as f:
            for line in f:
                pool.append(json.loads(line))
        rng.shuffle(pool)
        pages = [WikiPage(title=p["title"], intro=p["intro"]) for p in pool[:args.n_pages]]
        examples = generate_oracle_sft(pages, sim_model, max_steps=args.max_steps,
                                       seed=args.seed, eos_token=args.eos_token)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{len(examples)} SFT examples → {out_path}")
        return

    # ---- Phase: generate ----
    if args.phase in ("generate", "all"):
        print(f"Loading model from {args.tiny_model} …")
        sim_model = TinyPedantixModel.load(args.tiny_model)

        print(f"Loading {args.n_pages} pages from {args.pages} …")
        rng = random.Random(args.seed)
        pool = []
        with open(args.pages) as f:
            for line in f:
                pool.append(json.loads(line))
        rng.shuffle(pool)
        pages = [WikiPage(title=p["title"], intro=p["intro"]) for p in pool[:args.n_pages]]

        print(f"Playing {len(pages)} games with oracle (max_steps={args.max_steps}) …")
        states = generate_oracle_states(pages, sim_model, max_steps=args.max_steps, seed=args.seed)

        states_file.write_text(json.dumps(states, ensure_ascii=False))
        print(f"States saved to {states_file}  ({len(states)} total)")

    # ---- Phase: submit ----
    if args.phase in ("submit", "all"):
        if args.phase == "submit":
            states = json.loads(states_file.read_text())
            print(f"Loaded {len(states)} states from {states_file}")

        print(f"Building batch JSONL for model={args.model} …")
        batch_jsonl = build_batch_jsonl(states, args.model)
        n_req = batch_jsonl.count("\n") + 1
        print(f"{n_req} requests ({len(batch_jsonl.encode())/1024/1024:.1f} MB)")

        batch_info = submit_batch(api_key, batch_jsonl, args.model)
        batch_info_file.write_text(json.dumps(batch_info))
        print(f"Batch info saved to {batch_info_file}")

    # ---- Phase: poll ----
    if args.phase in ("poll", "all"):
        if args.phase == "poll":
            batch_info = json.loads(batch_info_file.read_text())

        print(f"Polling batch {batch_info['id']} …")
        output_file_id = poll_batch(api_key, batch_info["id"])
        batch_info["output_file_id"] = output_file_id
        batch_info_file.write_text(json.dumps(batch_info))
        print(f"Batch complete. Output file: {output_file_id}")

    # ---- Phase: process ----
    if args.phase in ("process", "all"):
        if args.phase == "process":
            states = json.loads(states_file.read_text())
            batch_info = json.loads(batch_info_file.read_text())
            output_file_id = batch_info["output_file_id"]
            print(f"Loading model from {args.tiny_model} …")
            sim_model = TinyPedantixModel.load(args.tiny_model)

        n = process_results(api_key, output_file_id, states, sim_model, out_path, args.eos_token)
        print(f"\nDone: {n} SFT examples → {out_path}")


if __name__ == "__main__":
    main()
