"""
Eval comparison: trained checkpoint vs base model × thinking/no-thinking × 50/100 steps.
Uses 50 pages sampled randomly from the top-500 most popular Wikipedia pages (seed=42).
Loads each model once and runs all 4 conditions. Saves full prediction logs per condition.
"""
import sys, json, time, gc
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

PAGES   = str(PROJ / "data/eval50_top500.jsonl")   # 50 pages sampled from top-500
TINY    = str(PROJ / "models/fasttext_wiki_model.npz")
OUT_DIR = PROJ / "models/eval_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PAGES = 50
SEED    = 42

MODELS = {
    "trained": str(PROJ / "models/v16_grpo_dagger_cont/checkpoint-200"),
}
CONDITIONS = [
    ("qwen", 100),
]

print("[eval] Loading FastText similarity model...", flush=True)
sim_model = FastTextWikiModel.load(TINY)

transformers, torch = _import_transformers()
all_results = {}

for model_key, model_path in MODELS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"[eval] Loading model: {model_key}", flush=True)
    print(f"[eval]   path: {model_path}", flush=True)
    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer_for_inference(model_path, transformers, torch)
    device = _best_device(torch)
    model.to(device)
    model.eval()
    print(f"[eval] Model loaded in {time.time()-t0:.0f}s", flush=True)

    for fmt, steps in CONDITIONS:
        label = f"{model_key}_{fmt}_{steps}steps"
        out_path = OUT_DIR / f"{label}.jsonl"
        print(f"\n[eval] Running: {label} ({N_PAGES} pages)...", flush=True)
        t1 = time.time()
        result = _run_eval_games(
            model=model,
            tokenizer=tokenizer,
            device=device,
            pages_path=PAGES,
            similarity_model=sim_model,
            sample_size=N_PAGES,
            max_steps=steps,
            seed=SEED,
            output_path=str(out_path),
            chat_format=fmt,
            generation_batch_size=1,
            eval_num_generations=1,
            constrained=False,
        )
        elapsed = time.time() - t1
        result["label"] = label
        result["elapsed_min"] = round(elapsed / 60, 1)
        all_results[label] = result

        # Per-game summary
        games = []
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line: games.append(json.loads(line))

        print(f"[eval] {label}: solve_rate={result.get('solve_rate', 0):.1%}  "
              f"elapsed={elapsed/60:.1f}min", flush=True)
        for g in games:
            guesses = [h["guess"] for h in g["history"]]
            scores  = [h["score"] for h in g["history"]]
            status  = "SOLVED" if g["solved"] else f"fail@{g['steps']}"
            pairs   = " → ".join(f"{w}({s})" for w, s in zip(guesses[:8], scores[:8]))
            if len(guesses) > 8:
                pairs += f" ... [{len(guesses)} total]"
            print(f"  [{status}] {g['title']}: {pairs}", flush=True)

    print(f"\n[eval] Unloading {model_key}...", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Final summary table
print("\n" + "="*72)
print(f"{'Condition':<38} {'Solve%':>7} {'Solved':>9} {'Time':>7}")
print("-"*72)
for label, r in all_results.items():
    sr       = r.get("solve_rate", 0)
    n_solved = r.get("n_solved", "?")
    elapsed  = r.get("elapsed_min", "?")
    print(f"{label:<38} {sr:>7.1%} {str(n_solved)+'/50':>9} {str(elapsed)+'m':>7}")

out_json = OUT_DIR / "summary.json"
out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
print(f"\n[eval] Full summary → {out_json}")
