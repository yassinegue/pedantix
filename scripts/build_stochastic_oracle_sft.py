#!/usr/bin/env python3
"""Generate SFT data from stochastic oracle trajectories.

Uses soft_oracle_guess with temperature sampling (top-K softmax) instead of
deterministic argmax — diverse (state, word) pairs so the model learns a
reward distribution, not a single lookup table.

Words are restored to their proper accented French form using fr_FR.dic.
Non-words (e.g. 'deu', accent-stripped forms not in dictionary) are filtered out.

N_RUNS trajectories per page with different seeds → covers more of the game
tree. CPU-bound (TinyModel numpy), intended to run on cadillac in parallel.

Output: data/sft_stochastic_oracle.jsonl
"""
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pedantix_project.dataset import load_pages
from pedantix_project.llm_policy import _feedback_step, make_prompt, soft_oracle_guess
from pedantix_project.model import TinyPedantixModel
from pedantix_project.text import normalize_word

# ── config ────────────────────────────────────────────────────────────────────
PAGES_PATH  = Path("data/clean_pages.jsonl")
OUTPUT      = Path("data/sft_stochastic_oracle.jsonl")
TINY_MODEL  = Path("models/tiny_model.json")
FR_DIC      = Path("/usr/share/myspell/fr_FR.dic")

N_PAGES     = 5000
N_RUNS      = 6
MAX_STEPS   = 30
MIN_IDF     = 1.5
MAX_ROWS    = 300_000
TEMPERATURE = 5.0
TOP_K       = 10
EOS         = "<|im_end|>"

STOPWORDS = {
    "de","la","le","les","du","des","un","une","en","et","ou",
    "est","au","aux","se","sa","son","ses","ce","qui","que","par",
    "sur","avec","dans","pour","il","elle","ils","elles","on","a","an",
}

# Common words missing from fr_FR.dic root list (derived via .aff rules)
FR_DICT_SUPPLEMENT: dict[str, str] = {
    "francais": "français", "francaise": "française",
    "francaises": "françaises", "francaisen": "français",
    "eglise": "église", "ete": "été", "etes": "été",
    "ele": "élé", "elu": "élu", "elue": "élue",
    "general": "général", "nationale": "nationale",
    "premier": "premier", "premiere": "première",
    "deuxieme": "deuxième", "troisieme": "troisième",
    "superieur": "supérieur", "inferieur": "inférieur",
}

# Proper names and noise words that appear in French Wikipedia but aren't
# useful Pedantix guesses — the oracle tends to pick these because they
# appear as neighbors of common words.
ORACLE_BLOCKLIST = {
    "anne", "inse", "rhone", "seine", "loire",
    "apre",   # âpre = rare adjective (harsh), not a useful Pedantix guess
    "apres",  # après handled by stopwords but just in case
}
# ─────────────────────────────────────────────────────────────────────────────


def _load_fr_dict(dic_path: Path) -> dict[str, str]:
    """Build canonical → accented form mapping from fr_FR.dic.

    For each lowercase dictionary word, map its accent-stripped form to the
    original accented form. When multiple accented forms share a canonical,
    the shortest/first is kept (e.g. 'etat' → 'état').
    """
    mapping: dict[str, str] = {}
    with dic_path.open(encoding="utf-8", errors="replace") as f:
        next(f)  # skip word count header
        for line in f:
            word = line.strip().split("/")[0]
            if not word or not word[0].islower() or not all(c.isalpha() for c in word):
                continue
            canon = normalize_word(word)
            if canon and canon not in mapping:
                mapping[canon] = word
    return mapping


print("Loading French dictionary...", flush=True)
FR_DICT: dict[str, str] = _load_fr_dict(FR_DIC) if FR_DIC.exists() else {}
print(f"  {len(FR_DICT)} canonical → accented entries", flush=True)

print("Loading TinyModel...", flush=True)
sim = TinyPedantixModel.load(str(TINY_MODEL))

print(f"Loading pages from {PAGES_PATH}...", flush=True)
pages = load_pages(PAGES_PATH)[:N_PAGES]
print(f"  {len(pages)} pages loaded", flush=True)

n_written = n_skipped_stop = n_skipped_idf = n_skipped_nonword = 0
seen: Counter = Counter()

with OUTPUT.open("w") as fout:
    for page_idx, page in enumerate(pages):
        if n_written >= MAX_ROWS:
            break

        for run_idx in range(N_RUNS):
            if n_written >= MAX_ROWS:
                break

            rng = random.Random(page_idx * 1000 + run_idx)
            history: list[dict] = []

            for _ in range(MAX_STEPS):
                word = soft_oracle_guess(
                    page, sim, history,
                    rng=rng,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    min_idf=MIN_IDF,
                )
                if not word:
                    break

                # restore accented form via FR_DICT; fall back to canonical
                # form for accent-free words (france, premier, etc.) which
                # are absent from the dic's root list but are valid.
                # Short canonical words not in dic (e.g. 'deu', 3 chars) are
                # noise — filter them out.
                accented = FR_DICT.get(word) or FR_DICT_SUPPLEMENT.get(word)
                if not accented:
                    if len(word) < 4:
                        n_skipped_nonword += 1
                        step = _feedback_step(page, sim, history, word)
                        history.append(step)
                        if step.get("solved"):
                            break
                        continue
                    accented = word  # already correct (no accent to restore)

                if accented in STOPWORDS or word in STOPWORDS or word in ORACLE_BLOCKLIST:
                    n_skipped_stop += 1
                elif sim.idf.get(word, 0.0) < MIN_IDF:
                    n_skipped_idf += 1
                else:
                    prompt_body = make_prompt(history, max_steps=MAX_STEPS)
                    prompt = (
                        f"<|im_start|>user\n{prompt_body}\n"
                        f"/no_think<|im_end|>\n<|im_start|>assistant\nMOT:"
                    )
                    fout.write(json.dumps({
                        "prompt": prompt,
                        "completion": f" {accented}{EOS}",
                        "page_title": page.title,
                    }, ensure_ascii=False) + "\n")
                    n_written += 1
                    seen[accented] += 1

                step = _feedback_step(page, sim, history, word)
                history.append(step)
                if step.get("solved"):
                    break

        if (page_idx + 1) % 500 == 0:
            print(f"  [{page_idx+1}/{len(pages)}] {n_written:,} examples written", flush=True)

print(f"\nDone.")
print(f"  Written:           {n_written:,}")
print(f"  Skipped non-word:  {n_skipped_nonword:,}")
print(f"  Skipped stopword:  {n_skipped_stop:,}")
print(f"  Skipped low-IDF:   {n_skipped_idf:,}")
print(f"  Unique words:      {len(seen):,}")
print(f"  Top 20 words:      {seen.most_common(20)}")
print(f"  Output:            {OUTPUT}")
