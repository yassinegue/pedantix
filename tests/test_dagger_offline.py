"""Offline DAgger sanity test — no GRPO, just verifies the pieces compose.

Loads a small SFT checkpoint, builds the (prompt, oracle_word) pairs via
_dagger_collect_pairs on a handful of pages, then runs one _dagger_bc_step
to confirm the optimizer/backward path works on the in-memory PEFT model.

Run with:
  python tests/test_dagger_offline.py models/sft_eos_smoke/checkpoint-125
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / "models/hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models/hf_cache/hub"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: test_dagger_offline.py <adapter_dir>")
    adapter_dir = sys.argv[1]

    from pedantix_project.corpus import sample_pages
    from pedantix_project.llm_policy import (
        _dagger_collect_pairs,
        _dagger_bc_step,
        _load_peft_model_if_adapter,
    )
    from pedantix_project.model import TinyPedantixModel

    import transformers
    import torch

    sim = TinyPedantixModel.load(ROOT / "models/tiny_model.json")
    pages = sample_pages(str(ROOT / "data/clean_pages.jsonl"), sample_size=4, seed=1)
    print(f"[dagger-test] sampled {len(pages)} pages: {[p.title for p in pages]}")

    # Use the GRPO load path so the LoRA adapter is trainable (matches the
    # state the DAgger callback will encounter at runtime).
    model = _load_peft_model_if_adapter(adapter_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[dagger-test] loaded model on {device} (trainable={isinstance(model, str) is False})")

    with torch.inference_mode():
        pairs = _dagger_collect_pairs(
            model=model,
            tokenizer=tokenizer,
            pages=pages,
            similarity_model=sim,
            history_max_steps=15,
            rollout_steps=5,
            chat_format="none",
            device=device,
            num_return_sequences=2,
        )
    print(f"[dagger-test] collected {len(pairs)} pairs")
    assert pairs, "expected at least one pair"
    for i, (p, c) in enumerate(pairs[:3]):
        print(f"  pair {i}: completion={c!r}  prompt-tail={p[-100:]!r}")

    # Confirm oracle words look like real French content words (no concatenations).
    completions = [c.replace("<|im_end|>", "").strip() for _, c in pairs]
    assert all(" " not in w and len(w) >= 2 for w in completions), \
        f"oracle returned malformed completion: {completions[:5]}"
    print("[dagger-test] all oracle completions look like single words")

    # Make at least the LoRA parameters require grad so backward works.
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable parameters; cannot BC-step"
    print(f"[dagger-test] trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    opt = torch.optim.AdamW(trainable, lr=1e-5)

    bs = min(4, len(pairs))
    batch = pairs[:bs]
    prompts = [p for p, _ in batch]
    completions = [c for _, c in batch]
    loss = _dagger_bc_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=opt,
        prompts=prompts,
        completions=completions,
    )
    print(f"[dagger-test] BC step loss = {loss:.4f}")
    assert loss > 0, "loss should be positive"
    assert loss < 50, f"loss looks broken: {loss}"
    print("ALL DAGGER OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
