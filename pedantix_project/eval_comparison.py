"""
Sharded eval: trained checkpoint, qwen format, 100 steps.
Run with SHARD_ID=0..NUM_SHARDS-1 and NUM_SHARDS env vars.
Each shard processes a slice of eval50_top500.jsonl independently.
"""
import sys, json, os, random
from pathlib import Path

PROJ = Path("/home/yguenn/pedantix")
sys.path.insert(0, str(PROJ))

from pedantix_project.model import FastTextWikiModel
from pedantix_project.llm_policy import (
    _import_transformers,
    _load_model_and_tokenizer_for_inference,
    _run_eval_games,
    _best_device,
)

PAGES_PATH = PROJ / "data/eval50_top500.jsonl"
TINY       = str(PROJ / "models/fasttext_wiki_model.npz")
OUT_DIR    = PROJ / "models/eval_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(PROJ / "models/v16_grpo_dagger_cont/checkpoint-200")
MAX_STEPS  = 100
SEED       = 42

NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "5"))
SHARD_ID   = int(os.environ.get("SHARD_ID", "0"))

# ── load and shard pages ──────────────────────────────────────────────────────
all_pages = []
with open(PAGES_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            all_pages.append(json.loads(line))

rng = random.Random(SEED)
rng.shuffle(all_pages)

shard_size = (len(all_pages) + NUM_SHARDS - 1) // NUM_SHARDS
start = SHARD_ID * shard_size
end   = min(start + shard_size, len(all_pages))
shard_pages = all_pages[start:end]

print(f"[shard {SHARD_ID}/{NUM_SHARDS}] pages {start}-{end-1} ({len(shard_pages)} games)", flush=True)

import tempfile, atexit
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
for p in shard_pages:
    tmp.write(json.dumps(p, ensure_ascii=False) + "\n")
tmp.close()
atexit.register(lambda: os.unlink(tmp.name))

# ── load model ────────────────────────────────────────────────────────────────
print("[eval] Loading FastText...", flush=True)
sim_model = FastTextWikiModel.load(TINY)

transformers, torch = _import_transformers()
print(f"[eval] Loading model from {MODEL_PATH}", flush=True)
model, tokenizer = _load_model_and_tokenizer_for_inference(MODEL_PATH, transformers, torch)
device = _best_device(torch)
model.to(device)
model.eval()

# ── run ───────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / f"shard_{SHARD_ID}.jsonl"
print(f"[shard {SHARD_ID}] running {len(shard_pages)} pages × {MAX_STEPS} steps...", flush=True)
result = _run_eval_games(
    model=model,
    tokenizer=tokenizer,
    device=device,
    pages_path=tmp.name,
    similarity_model=sim_model,
    sample_size=len(shard_pages),
    max_steps=MAX_STEPS,
    seed=SEED + SHARD_ID,
    output_path=str(out_path),
    chat_format="qwen",
    generation_batch_size=1,
    eval_num_generations=1,
    constrained=False,
)

print(f"[shard {SHARD_ID}] solve_rate={result.get('solve_rate',0):.1%}  "
      f"n_solved={result.get('n_solved','?')}/{len(shard_pages)}", flush=True)

games = []
with open(out_path) as f:
    for line in f:
        line = line.strip()
        if line:
            games.append(json.loads(line))

for g in games:
    guesses = [h["guess"] for h in g["history"]]
    scores  = [h["score"] for h in g["history"]]
    status  = "SOLVED" if g["solved"] else f"fail@{g['steps']}"
    pairs   = " → ".join(f"{w}({s})" for w, s in zip(guesses[:8], scores[:8]))
    if len(guesses) > 8:
        pairs += f" ... [{len(guesses)} total]"
    print(f"  [{status}] {g['title']}: {pairs}", flush=True)
