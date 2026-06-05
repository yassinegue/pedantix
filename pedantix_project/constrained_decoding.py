"""Constrained decoding: restrict generation to valid French words.

Builds a trie over Qwen tokenizer sequences for each word in the vocabulary,
then uses HF's `prefix_allowed_tokens_fn` API to hard-mask invalid tokens at
every generation step.  Zero garbage, zero fallbacks.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Callable

import torch


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def load_french_words(
    dic_path: str | Path | None = "/usr/share/myspell/fr_FR.dic",
    tiny_model_path: str | Path | None = None,
    extra_exclusions: set[str] | None = None,
    allowed_vocab: frozenset[str] | None = None,
) -> list[str]:
    """Return a deduplicated list of lowercase French words.

    If allowed_vocab is provided (accent-stripped), only words whose stripped form
    appears in allowed_vocab are included — this aligns the trie with the game's
    similarity model so the model never generates words the game rejects.
    """
    words: set[str] = set()
    exclusions: set[str] = extra_exclusions or set()

    def _accept(w: str) -> bool:
        stripped = _strip_accents(w)
        if stripped in exclusions or w in exclusions:
            return False
        if allowed_vocab is not None and stripped not in allowed_vocab:
            return False
        return True

    # fr_FR.dic: root words (before the '/' affix separator), lowercase alpha only
    if dic_path and Path(dic_path).exists():
        with open(dic_path, encoding="utf-8", errors="replace") as f:
            next(f)  # skip word-count header
            for line in f:
                w = line.strip().split("/")[0].lower()
                if w and w[0].islower() and all(c.isalpha() for c in w) and len(w) >= 2 and _accept(w):
                    words.add(w)
                    words.add(_strip_accents(w))

    # TinyModel vocabulary (canonical, accent-stripped)
    if tiny_model_path and Path(tiny_model_path).exists():
        import json
        data = json.loads(Path(tiny_model_path).read_text())
        for w in data.get("idf", {}).keys():
            if len(w) >= 2 and all(c.isalpha() for c in w) and _accept(w):
                words.add(w)

    return sorted(words)


def build_trie(tokenizer, words: list[str]) -> dict:
    """Map every French word to its Qwen token-ID sequence in a prefix trie.

    Trie node: dict mapping token_id (int) -> child_node.
    A node with key '__end__' marks a complete word endpoint.
    """
    trie: dict = {}
    skipped = 0
    for word in words:
        # Leading space matches how the tokenizer encodes after "MOT:" in generation context
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not ids:
            skipped += 1
            continue
        node = trie
        for tid in ids:
            node = node.setdefault(tid, {})
        node["__end__"] = True
    return trie


def make_prefix_allowed_fn(
    trie: dict,
    eos_token_id: int,
    prompt_length: int,
) -> Callable[[int, torch.Tensor], list[int]]:
    """Return a prefix_allowed_tokens_fn compatible with model.generate().

    At each step:
    - If no tokens generated yet: allow any trie root token
    - Mid-word: allow valid continuations from trie
    - At a word boundary (__end__): also allow EOS to terminate
    - Off-trie (shouldn't happen): force EOS
    """

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        gen_tokens = input_ids[prompt_length:].tolist()

        node = trie
        for tok in gen_tokens:
            if tok == eos_token_id:
                return [eos_token_id]
            if tok not in node:
                # Generated token not in trie — force stop
                return [eos_token_id]
            node = node[tok]

        allowed = [tok for tok in node if tok != "__end__"]
        if "__end__" in node:
            allowed.append(eos_token_id)

        return allowed if allowed else [eos_token_id]

    return prefix_allowed_tokens_fn


def make_dynamic_prefix_allowed_fn(
    trie: dict,
    eos_token_id: int,
    prompt_terminal_token_id: int,
) -> Callable[[int, torch.Tensor], list[int]]:
    """Variant that auto-detects prompt_length each call.

    Scans input_ids right-to-left for prompt_terminal_token_id (the last token of
    "MOT:" in the prompt). Everything after that position is treated as generated.
    Safe because the trie only contains alphabetic tokens — the terminal token
    (typically ':') can never appear in generated output.
    """

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        ids = input_ids.tolist()
        prompt_end = -1
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == prompt_terminal_token_id:
                prompt_end = i
                break
        if prompt_end == -1:
            return [eos_token_id]
        gen_tokens = ids[prompt_end + 1:]

        node = trie
        for tok in gen_tokens:
            if tok == eos_token_id:
                return [eos_token_id]
            if tok not in node:
                return [eos_token_id]
            node = node[tok]

        allowed = [tok for tok in node if tok != "__end__"]
        if "__end__" in node:
            allowed.append(eos_token_id)
        return allowed if allowed else [eos_token_id]

    return prefix_allowed_tokens_fn


def build_french_constraint(
    tokenizer,
    prompt_length: int | None,
    dic_path: str | Path | None = "/usr/share/myspell/fr_FR.dic",
    tiny_model_path: str | Path | None = None,
    extra_exclusions: set[str] | None = None,
    allowed_vocab: frozenset[str] | None = None,
    dynamic: bool = False,
    _trie_cache: dict = {},
) -> Callable[[int, torch.Tensor], list[int]]:
    """One-call convenience: load words, build trie (cached), return constraint fn.

    If dynamic=True, prompt_length is ignored and the boundary is detected by
    scanning for the last ':' token (end of "MOT:") in each input sequence.
    Use dynamic=True when called from inside a training loop where prompt_length
    varies per batch (e.g., TRL GRPO).
    """
    cache_key = (str(dic_path), str(tiny_model_path),
                 frozenset(extra_exclusions or ()), allowed_vocab)
    if cache_key not in _trie_cache:
        words = load_french_words(dic_path, tiny_model_path,
                                  extra_exclusions=extra_exclusions,
                                  allowed_vocab=allowed_vocab)
        trie = build_trie(tokenizer, words)
        _trie_cache[cache_key] = trie
        print(f"[constrained_decoding] trie built: {len(words)} words, "
              f"{len(trie)} root tokens", flush=True)
    trie = _trie_cache[cache_key]
    if dynamic:
        # Find the last token of "MOT:" — the prompt always ends with this token
        mot_ids = tokenizer.encode("MOT:", add_special_tokens=False)
        prompt_terminal_token_id = mot_ids[-1]
        return make_dynamic_prefix_allowed_fn(trie, tokenizer.eos_token_id, prompt_terminal_token_id)
    return make_prefix_allowed_fn(trie, tokenizer.eos_token_id, prompt_length)
