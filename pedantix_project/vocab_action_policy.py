from __future__ import annotations

import json
import random
import re
import types
from collections import Counter
from dataclasses import replace as _dc_replace
from pathlib import Path

from .corpus import sample_pages
from .dataset import WikiPage, load_pages
from .llm_policy import (
    _apply_llm_exact_reveals,
    _best_device,
    _best_rewarding_guess,
    _feedback_step,
    _import_training_stack,
    _import_transformers,
    _load_peft_model_if_adapter,
    _lora_config,
    _title_norm_words,
    _tokenizer_name_for_model,
    compact_visible_text,
    enforce_hf_cache,
    format_prompt_for_model,
    extract_guess,
    is_valid_guess,
    make_prompt,
    replay_game,
    score_completion,
    score_guess_on_game,
)
from .model import TinyPedantixModel
from .rl import build_action_space
from .simulator import PedantixGame


VOCAB_ACTION_CONFIG = "vocab_action_config.json"
ACTION_HEAD_WEIGHTS = "action_head.pt"


def _make_action_head(torch, hidden_size: int, num_actions: int, head_hidden_size: int = 0):
    """Single linear or 2-layer MLP action head."""
    if head_hidden_size > 0:
        return torch.nn.Sequential(
            torch.nn.Linear(hidden_size, head_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(head_hidden_size, num_actions),
        )
    return torch.nn.Linear(hidden_size, num_actions)


def _action_head_logits(action_head, hidden):
    """Forward pass regardless of head type."""
    return action_head(hidden)


def _make_lm_head_module(word_embeds, torch):
    """Construct a cosine-similarity action head from pre-computed L2-normalised embeddings.

    logits[w] = cosine_similarity(hidden, embed(w))

    word_embeds are fixed (buffer, not parameter): LoRA fine-tunes hidden states
    to move toward the correct title word's direction in embedding space.
    Defined lazily so torch is not needed at module-import time.
    """
    class _LMHead(torch.nn.Module):
        def __init__(self, we):
            super().__init__()
            self.register_buffer("word_embeds", we)

        def forward(self, hidden):
            h = torch.nn.functional.normalize(hidden.float(), dim=-1)
            return h @ self.word_embeds.T

        @property
        def bias(self):
            return None

    head = _LMHead(word_embeds)
    head._is_lm_head = True
    return head


def _build_lm_head_action_head(model, tokenizer, actions: list[str], torch):
    """Build a cosine-similarity action head by extracting un-embeddings for each action word."""
    base = model.base_model.model if hasattr(model, "base_model") else model
    lm_weight = base.lm_head.weight.detach().float()  # (vocab_size, hidden_size)

    vecs = []
    for word in actions:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            ids = [tokenizer.unk_token_id or 0]
        vec = lm_weight[ids].mean(dim=0)
        vecs.append(vec)

    word_embeds = torch.stack(vecs)
    word_embeds = torch.nn.functional.normalize(word_embeds, dim=-1)
    print(f"LM-head action head: {len(actions)} word embeddings extracted from lm_head.weight", flush=True)
    return _make_lm_head_module(word_embeds, torch)


def _compact_visible_fast(tokens, revealed: set, max_chars: int = 240) -> str:
    """compact_visible_text without a full PedantixGame object."""
    parts: list[str] = []
    hidden = False
    for idx, tok in enumerate(tokens):
        if not tok.is_word:
            parts.append(tok.text)
            hidden = False
        elif idx in revealed:
            parts.append(tok.text)
            hidden = False
        elif not hidden:
            parts.append("__")
            hidden = True
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    if len(text) <= max_chars:
        return text.strip()
    head = text[: max_chars // 2].rsplit(" ", 1)[0]
    tail = text[-max_chars // 2 :].split(" ", 1)[-1]
    return f"{head}\n[...]\n{tail}".strip()


def _score_with_view(
    base: PedantixGame,
    revealed: set,
    guessed: set,
    word: str,
    hist_len: int,
    *,
    near_solve_shaping_coef: float = 0.0,
) -> tuple:
    """
    Score a guess using a lightweight view of base that carries trajectory-specific
    revealed/guessed sets — avoids deepcopy of the heavy token list.
    Updates revealed and guessed in-place. Returns (reward_obj, hist_step_dict).

    near_solve_shaping_coef: if > 0, adds a bonus equal to
        coef * max_cosine_sim(guess, unrevealed_title_words)
    to the returned reward, creating a dense gradient signal toward title words
    even when exact solves are rare. The hist_step logs the original game reward.
    """
    from .text import normalize_word as _nw
    norm = _nw(word)
    was_repeated = norm in guessed
    view = types.SimpleNamespace(
        revealed=revealed,
        guessed=guessed,
        title_word_indices=base.title_word_indices,
        tokens=base.tokens,
        similarity_model=base.similarity_model,
        semantic_threshold=base.semantic_threshold,
        solved=bool(base.title_word_indices <= revealed),
    )
    r = score_guess_on_game(view, word, history_len=hist_len, guessed=guessed)

    # Near-solve shaping: bonus proportional to max sim to any unrevealed title token.
    # Computed BEFORE updating revealed so we see the correct pre-guess state.
    shaped_bonus = 0.0
    if near_solve_shaping_coef > 0.0 and not r.invalid and base.similarity_model is not None:
        unrevealed_title_norms = {
            base.tokens[idx].norm
            for idx in base.title_word_indices
            if idx not in revealed
        }
        if unrevealed_title_norms:
            max_sim = max(
                base.similarity_model.similarity(norm, tw)
                for tw in unrevealed_title_norms
            )
            if max_sim > 0.0:
                shaped_bonus = near_solve_shaping_coef * max_sim

    # Update revealed and guessed
    for idx, tok in enumerate(base.tokens):
        if tok.is_word and tok.norm == norm:
            revealed.add(idx)
    guessed.add(norm)

    hist_step = {
        "guess": r.guess,
        "exact": r.exact_hits,
        "semantic": r.semantic_hits,
        "title": r.title_hits,
        "reward": r.reward,  # original game reward, not inflated by shaping
        "solved": r.solved,
        "invalid": r.invalid and not was_repeated,
        "repeated": was_repeated,
    }
    if shaped_bonus:
        r = _dc_replace(r, reward=r.reward + shaped_bonus)
    return r, hist_step


def _on_policy_rollout(
    model,
    tokenizer,
    action_head,
    actions: list,
    action_to_idx: dict,
    similarity_model,
    batch: list[dict],
    rollout_steps: int,
    num_generations: int,
    temperature: float,
    max_prompt_length: int,
    device: str,
    torch,
    *,
    title_mask_fn=None,
    near_solve_shaping_coef: float = 0.0,
    dynamic_expand_k: int = 0,
    base_vocab_size: int = 0,
) -> tuple:
    """
    Collect B×G on-policy trajectories of K steps (no gradient).

    Per-trajectory state = (revealed: set[int], guessed: set[str]) — lightweight
    copies that share the heavy token list / model from one base PedantixGame per
    batch element.  No deepcopy of full game objects; no thread pool.

    Returns
    -------
    step0_prompts  : list[str]          – B prompts (shared by all G at step 0)
    step0_actions  : LongTensor (B, G)  – sampled action IDs at step 0
    later_steps    : list[list]         – (K-1) lists, each containing
                                          (prompt_str, action_id, b, g) tuples
    rewards        : FloatTensor (B, G) – cumulative K-step rewards (includes shaping)
    solved_count   : int
    mean_shaped_bonus : float           – mean shaping bonus per trajectory step
    """
    B, G = len(batch), num_generations
    pages = [
        WikiPage(title=str(r.get("title", "")), intro=str(r.get("intro", "")))
        for r in batch
    ]
    start_hists = [
        (json.loads(r["history"]) if isinstance(r["history"], str) else list(r["history"]))
        for r in batch
    ]

    # Build one base game per batch element (contains the heavy token list).
    base_games: list[PedantixGame] = []
    for b in range(B):
        g0 = PedantixGame(pages[b], similarity_model=similarity_model)
        for step in start_hists[b]:
            guess = str(step.get("guess", ""))
            if guess:
                _apply_llm_exact_reveals(g0, guess)
        base_games.append(g0)

    # ── Step 0: one prompt per batch element (shared by all G) ──────────────
    step0_prompts = [
        format_prompt_for_model(
            make_prompt(start_hists[b], max_steps=200,
                        visible_text=_compact_visible_fast(base_games[b].tokens, base_games[b].revealed)),
            chat_format="qwen",
        )
        for b in range(B)
    ]

    with torch.no_grad():
        enc = tokenizer(step0_prompts, padding=True, truncation=True,
                        max_length=max_prompt_length, return_tensors="pt").to(device)
        lh = _forward_last_hidden(model, enc)
        pos = enc["attention_mask"].sum(dim=1) - 1
        h = lh[torch.arange(B, device=device), pos].float()
        lgt0 = action_head(h) / max(0.05, temperature)
        lgt0_masked = _mask_guessed_actions(lgt0, batch, action_to_idx, torch)
        if title_mask_fn is not None:
            lgt0_masked = title_mask_fn(lgt0_masked, batch)
        if dynamic_expand_k > 0:
            ep_mask0 = _episode_expand_mask(batch, actions, action_to_idx, similarity_model, base_vocab_size, dynamic_expand_k, device, torch)
            lgt0_masked = lgt0_masked.masked_fill(~ep_mask0, float('-inf'))
        step0_actions = torch.distributions.Categorical(logits=lgt0_masked).sample((G,)).T  # (B, G)

    # Per-trajectory mutable state: only sets (lightweight copies of base state).
    traj_revealed: list[list[set]] = [
        [set(base_games[b].revealed) for _ in range(G)] for b in range(B)
    ]
    traj_guessed: list[list[set]] = [
        [set(base_games[b].guessed) for _ in range(G)] for b in range(B)
    ]
    traj_hists: list[list[list]] = [[list(start_hists[b]) for _ in range(G)] for b in range(B)]
    cum_rewards = [[0.0] * G for _ in range(B)]
    solved: list[list[bool]] = [[False] * G for _ in range(B)]
    solved_count = 0

    # ── Score step-0 actions ─────────────────────────────────────────────────
    total_shaped_bonus = 0.0
    total_steps_scored = 0
    for b in range(B):
        base = base_games[b]
        for g in range(G):
            word = actions[int(step0_actions[b, g])]
            r, hist_step = _score_with_view(
                base, traj_revealed[b][g], traj_guessed[b][g],
                word, len(traj_hists[b][g]),
                near_solve_shaping_coef=near_solve_shaping_coef,
            )
            cum_rewards[b][g] += r.reward
            traj_hists[b][g].append(hist_step)
            total_shaped_bonus += r.reward - hist_step["reward"]
            total_steps_scored += 1
            if r.solved:
                solved[b][g] = True
                solved_count += 1

    # ── Steps 1..K-1 ────────────────────────────────────────────────────────
    MINI = 64
    later_steps: list[list] = []

    for _ in range(1, rollout_steps):
        prompts_k: list[str] = []
        bg_k: list[tuple[int, int]] = []

        for b in range(B):
            for g in range(G):
                if solved[b][g]:
                    continue
                hist = traj_hists[b][g]
                vis = _compact_visible_fast(base_games[b].tokens, traj_revealed[b][g])
                prompts_k.append(format_prompt_for_model(
                    make_prompt(hist, max_steps=200, visible_text=vis),
                    chat_format="qwen",
                ))
                bg_k.append((b, g))

        if not prompts_k:
            break

        sampled_parts: list = []
        with torch.no_grad():
            for s in range(0, len(prompts_k), MINI):
                mp = prompts_k[s:s + MINI]
                mbg = bg_k[s:s + MINI]
                enc = tokenizer(mp, padding=True, truncation=True,
                                max_length=max_prompt_length, return_tensors="pt").to(device)
                lh = _forward_last_hidden(model, enc)
                pos = enc["attention_mask"].sum(dim=1) - 1
                h = lh[torch.arange(len(mp), device=device), pos].float()
                lgt = action_head(h) / max(0.05, temperature)
                fake = [{"history": json.dumps(traj_hists[b][g]), "title": str(batch[b].get("title", ""))} for b, g in mbg]
                lgt_m = _mask_guessed_actions(lgt, fake, action_to_idx, torch)
                if title_mask_fn is not None:
                    lgt_m = title_mask_fn(lgt_m, fake)
                if dynamic_expand_k > 0:
                    ep_mask_k = _episode_expand_mask(fake, actions, action_to_idx, similarity_model, base_vocab_size, dynamic_expand_k, device, torch)
                    lgt_m = lgt_m.masked_fill(~ep_mask_k, float('-inf'))
                sampled_parts.append(
                    torch.distributions.Categorical(logits=lgt_m).sample().cpu()
                )

        sampled_flat = torch.cat(sampled_parts)
        step_entries = []
        for i, (b, g) in enumerate(bg_k):
            action_id = int(sampled_flat[i])
            word = actions[action_id]
            r, hist_step = _score_with_view(
                base_games[b], traj_revealed[b][g], traj_guessed[b][g],
                word, len(traj_hists[b][g]),
                near_solve_shaping_coef=near_solve_shaping_coef,
            )
            cum_rewards[b][g] += r.reward
            traj_hists[b][g].append(hist_step)
            total_shaped_bonus += r.reward - hist_step["reward"]
            total_steps_scored += 1
            if not solved[b][g] and r.solved:
                solved[b][g] = True
                solved_count += 1
            step_entries.append((prompts_k[i], action_id, b, g))
        later_steps.append(step_entries)

    rewards = torch.tensor(cum_rewards, dtype=torch.float32, device=device)
    mean_shaped_bonus = total_shaped_bonus / max(1, total_steps_scored)
    return step0_prompts, step0_actions.cpu(), later_steps, rewards, solved_count, mean_shaped_bonus


def _dagger_refresh(
    model,
    tokenizer,
    action_head,
    actions: list,
    action_to_idx: dict,
    similarity_model,
    rows: list,
    rng,
    *,
    n_pages: int,
    rollout_steps: int,
    bc_steps: int,
    optimizer,
    device: str,
    max_prompt_length: int,
    torch,
) -> dict:
    """DAgger: simulate policy greedily on n_pages starting states, collect oracle labels, do BC update on action_head."""
    was_training_model = model.training
    model.eval()
    action_head.eval()

    bc_pairs: list[tuple[str, int]] = []  # (formatted_prompt, oracle_action_idx)

    for _ in range(n_pages):
        row = rng.choice(rows)
        page = WikiPage(title=str(row.get("title", "")), intro=str(row.get("intro", "")))
        history: list[dict] = []
        guessed: set[str] = set()

        for _k in range(rollout_steps):
            game = replay_game(page, similarity_model, history)
            if game.title_word_indices and game.title_word_indices <= game.revealed:
                break

            # Oracle label: best word the expert would pick at this game state
            title_words = [w for w in _title_norm_words(page) if w in action_to_idx]
            candidates = title_words + [w for w in actions[:300] if w not in guessed]
            oracle_word = _best_rewarding_guess(page, similarity_model, history, candidates, blocked=guessed)
            if not oracle_word or oracle_word not in action_to_idx:
                break

            # Build prompt for this policy-visited state
            visible = compact_visible_text(game)
            prompt_raw = make_prompt(history, max_steps=200, visible_text=visible)
            prompt = format_prompt_for_model(prompt_raw, chat_format="qwen")
            bc_pairs.append((prompt, action_to_idx[oracle_word]))

            # Advance state with the policy's greedy action (argmax)
            enc = tokenizer([prompt], padding=True, truncation=True,
                            max_length=max_prompt_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                lh = _forward_last_hidden(model, enc)
                pos = enc["attention_mask"].sum(dim=1) - 1
                h = lh[torch.arange(1, device=device), pos].float()
                action_id = int(action_head(h).argmax(dim=-1).item())
            chosen_word = actions[action_id]
            guessed.add(chosen_word)
            history = history + [_feedback_step(page, similarity_model, history, chosen_word)]

    if not bc_pairs:
        return {"dagger_pairs": 0, "dagger_bc_loss": 0.0}

    # BC update: train action_head on (policy-state, oracle-label) pairs
    action_head.train()
    _lmhead_mode = getattr(action_head, "_is_lm_head", False)
    if _lmhead_mode:
        # Gradient must flow through hidden states to reach LoRA params
        model.train()
    total_loss = 0.0
    n_updates = min(bc_steps, len(bc_pairs))
    for _ in range(n_updates):
        idx = rng.randrange(len(bc_pairs))
        prompt, target_id = bc_pairs[idx]
        enc = tokenizer([prompt], padding=True, truncation=True,
                        max_length=max_prompt_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if _lmhead_mode:
            lh = _forward_last_hidden(model, enc)
        else:
            with torch.no_grad():
                lh = _forward_last_hidden(model, enc)
        pos = enc["attention_mask"].sum(dim=1) - 1
        h = lh[torch.arange(1, device=device), pos].float()
        lgt = action_head(h)
        loss = torch.nn.functional.cross_entropy(lgt, torch.tensor([target_id], device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _clip_ps = list(action_head.parameters()) or [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(_clip_ps, 1.0)
        optimizer.step()
        total_loss += float(loss.detach())

    if was_training_model:
        model.train()
        action_head.train()
    return {"dagger_pairs": len(bc_pairs), "dagger_bc_loss": round(total_loss / n_updates, 4)}


def train_vocab_action_grpo(
    *,
    train_jsonl: str | Path,
    pages_path: str | Path,
    model_name_or_path: str,
    tiny_model_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    batch_size: int,
    num_generations: int,
    action_size: int,
    max_prompt_length: int,
    learning_rate: float,
    lora_rank: int,
    temperature: float,
    bc_coef: float,
    seed: int,
    log_path: str | Path | None = None,
    freeze_backbone: bool = False,
    entropy_coef: float = 0.0,
    bc_warmup_steps: int = 0,
    freeze_lora_steps: int = 0,
    plot_path: str | Path | None = None,
    plot_every: int = 50,
    action_head_weight_decay: float = 0.0,
    min_entropy: float = 0.0,
    min_entropy_coef: float = 1.0,
    head_hidden_size: int = 0,
    kl_ref_coef: float = 0.0,
    dynamic_sampling: bool = False,
    rollout_steps: int = 1,
    dagger_every: int = 0,
    dagger_bc_steps: int = 10,
    dagger_pages: int = 4,
    title_bias_init: float = 0.0,
    title_mask_cos_threshold: float = 0.0,
    use_lm_head: bool = False,
    near_solve_shaping_coef: float = 0.0,
    dynamic_expand_k: int = 0,
) -> None:
    enforce_hf_cache()
    torch, transformers, peft = _import_vocab_stack()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    similarity_model = TinyPedantixModel.load(tiny_model_path)
    rows = _load_training_rows(train_jsonl)
    actions, title_word_indices, base_vocab_size = _build_vocab_actions(
        pages_path, max_words=action_size, train_jsonl=train_jsonl,
        dynamic_expand_k=dynamic_expand_k,
        similarity_model=similarity_model if dynamic_expand_k > 0 else None,
    )
    action_to_idx = {word: idx for idx, word in enumerate(actions)}

    tokenizer_name = _tokenizer_name_for_model(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    frozen_cfg_path = Path(model_name_or_path) / VOCAB_ACTION_CONFIG
    is_frozen_checkpoint = frozen_cfg_path.exists() and not (Path(model_name_or_path) / "adapter_config.json").exists()

    # For frozen checkpoints, read the original base model name from the saved config
    if is_frozen_checkpoint:
        frozen_cfg = json.loads(frozen_cfg_path.read_text(encoding="utf-8"))
        hf_base_name = frozen_cfg.get("base_model_name_or_path") or tokenizer_name
    else:
        hf_base_name = tokenizer_name

    if freeze_backbone:
        model = transformers.AutoModelForCausalLM.from_pretrained(hf_base_name, torch_dtype="auto")
        model.requires_grad_(False)
    elif is_frozen_checkpoint:
        # Continuing from a frozen checkpoint: load base model by name, apply fresh LoRA
        peft_config = _lora_config(peft, lora_rank)
        model = transformers.AutoModelForCausalLM.from_pretrained(hf_base_name, torch_dtype="auto")
        model = peft.get_peft_model(model, peft_config)
    else:
        model_arg = _load_peft_model_if_adapter(model_name_or_path)
        peft_config = None if not isinstance(model_arg, str) else _lora_config(peft, lora_rank)
        if isinstance(model_arg, str):
            model = transformers.AutoModelForCausalLM.from_pretrained(model_arg, torch_dtype="auto")
            model = peft.get_peft_model(model, peft_config)
        else:
            model = model_arg

    hidden_size = int(getattr(model.config, "hidden_size"))
    if use_lm_head:
        action_head = _build_lm_head_action_head(model, tokenizer, actions, torch)
    else:
        action_head = _make_action_head(torch, hidden_size, len(actions), head_hidden_size)
        # Small init so initial logits are near-uniform → entropy starts at max
        for module in (action_head.modules() if head_hidden_size > 0 else [action_head]):
            if hasattr(module, "weight"):
                torch.nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        # Boost sampling probability for title-only words at initialization
        if title_bias_init != 0.0 and title_word_indices:
            with torch.no_grad():
                last_linear = list(action_head.children())[-1] if head_hidden_size > 0 else action_head
                if last_linear.bias is not None:
                    last_linear.bias[title_word_indices] += title_bias_init
            print(f"title-word bias init: added {title_bias_init:.1f} to {len(title_word_indices)} title-only indices", flush=True)

    # Frozen reference head for reverse KL: KL(ref || current)
    # Gradient = p_i - ref_i, non-zero for ALL words even at full collapse
    import copy as _copy
    action_head_ref = _copy.deepcopy(action_head)
    for p in action_head_ref.parameters():
        p.requires_grad_(False)

    # If warm-starting from a frozen checkpoint, load its trained action head weights
    # Remap by word identity to handle vocabulary ordering differences between oracle datasets
    # Only supported for simple Linear heads (head_hidden_size=0)
    if is_frozen_checkpoint and head_hidden_size == 0 and not use_lm_head:
        prev_head_path = Path(model_name_or_path) / ACTION_HEAD_WEIGHTS
        if prev_head_path.exists():
            prev_state = torch.load(prev_head_path, map_location="cpu", weights_only=True)
            old_actions = frozen_cfg.get("actions", [])
            if old_actions and len(old_actions) == prev_state["weight"].shape[0]:
                old_action_to_idx = {word: idx for idx, word in enumerate(old_actions)}
                new_weight = action_head.weight.data.clone()
                remapped = 0
                for new_idx, word in enumerate(actions):
                    old_idx = old_action_to_idx.get(word)
                    if old_idx is not None:
                        new_weight[new_idx] = prev_state["weight"][old_idx]
                        remapped += 1
                action_head.weight.data.copy_(new_weight)
                print(f"remapped {remapped}/{len(actions)} action head weights by word identity from {prev_head_path}", flush=True)
            elif prev_state["weight"].shape == action_head.weight.shape:
                action_head.load_state_dict(prev_state)
                print(f"loaded action head from {prev_head_path} (no old actions list, positional load)", flush=True)
            else:
                print(f"action head shape mismatch, starting fresh", flush=True)
    device = _best_device(torch)
    model.to(device)
    action_head.to(device)
    action_head_ref.to(device)
    if freeze_backbone:
        model.eval()
    else:
        model.train()
    action_head.train()
    if not freeze_backbone and hasattr(model, "gradient_checkpointing_enable"):
        # enable_input_require_grads is required for PEFT/LoRA + gradient checkpointing:
        # without it gradients can't flow through the checkpointed inputs to LoRA params,
        # which causes CheckpointError with mismatched tensor shapes during recomputation.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        print("gradient checkpointing enabled", flush=True)

    lora_params = [p for p in model.parameters() if p.requires_grad] if not freeze_backbone else []

    def _make_optimizer(lora_ps, head_ps):
        groups = []
        if lora_ps:
            groups.append({"params": lora_ps, "lr": learning_rate, "weight_decay": 0.0})
        if head_ps:
            groups.append({"params": head_ps, "lr": learning_rate, "weight_decay": action_head_weight_decay})
        if not groups:
            raise ValueError("no trainable parameters: use_lm_head=True requires freeze_backbone=False")
        return torch.optim.AdamW(groups)

    if freeze_lora_steps > 0 and not freeze_backbone and not use_lm_head:
        for p in lora_params:
            p.requires_grad_(False)
        optimizer = _make_optimizer([], list(action_head.parameters()))
        print(f"phase 1: LoRA frozen for first {freeze_lora_steps} steps, training action head only", flush=True)
    else:
        optimizer = _make_optimizer(lora_params, list(action_head.parameters()))
    import time as _time
    from .text import normalize_word as _nw

    # Build a weighted sample pool: oversample rows where the oracle completion IS a
    # title word (near-solve states) at 3× so the model sees more solvable episodes.
    def _is_title_completion(row: dict) -> bool:
        word = extract_guess(str(row.get("completion", "")))
        if not word:
            return False
        page = WikiPage(title=str(row.get("title", "")), intro=str(row.get("intro", "")))
        title_norms = {_nw(w) for w in _title_norm_words(page)}
        return _nw(word) in title_norms

    near_solve = [r for r in rows if _is_title_completion(r)]
    sample_pool = rows + near_solve * 2   # 3× weight for near-solve states
    print(
        f"near-solve rows: {len(near_solve)}/{len(rows)} ({100*len(near_solve)/len(rows):.1f}%) "
        f"— oversampled 3×, pool size={len(sample_pool)}",
        flush=True,
    )

    logs: list[dict] = []
    if rollout_steps > 1:
        print(f"rollout_steps={rollout_steps}: on-policy rollout, {len(rows)} oracle rows available", flush=True)

    for step in range(1, max_steps + 1):
        _t0 = _time.perf_counter()
        if freeze_lora_steps > 0 and not freeze_backbone and step == freeze_lora_steps + 1:
            for p in lora_params:
                p.requires_grad_(True)
            optimizer = _make_optimizer(lora_params, list(action_head.parameters()))
            print(f"phase 2: LoRA unfrozen at step {step}, joint training begins", flush=True)
        batch = [sample_pool[rng.randrange(len(sample_pool))] for _ in range(batch_size)]

        current_trainable = [p for p in lora_params if p.requires_grad] + list(action_head.parameters())

        if rollout_steps > 1:
            # ── On-policy rollout ─────────────────────────────────────────────
            # Collect B×G trajectories of K steps under no_grad.  Each trajectory
            # tracks its own history so the model cannot repeat words — matching
            # the actual game rule and eliminating single-word collapse.
            _title_mask_fn = (
                (lambda lgt, rows: _title_mask_logits(lgt, rows, actions, title_word_indices, similarity_model, title_mask_cos_threshold, torch))
                if title_mask_cos_threshold > 0.0 and title_word_indices else None
            )
            step0_prompts, step0_acts_cpu, later_steps, rewards_tensor, solved_count, mean_shaped_bonus = (
                _on_policy_rollout(
                    model, tokenizer, action_head, actions, action_to_idx,
                    similarity_model, batch, rollout_steps, num_generations,
                    temperature, max_prompt_length, device, torch,
                    title_mask_fn=_title_mask_fn,
                    near_solve_shaping_coef=near_solve_shaping_coef,
                    dynamic_expand_k=dynamic_expand_k,
                    base_vocab_size=base_vocab_size,
                )
            )
            step0_acts = step0_acts_cpu.to(device)  # (B, G)
            if title_word_indices:
                _title_set = set(title_word_indices)
                _n_title = sum(1 for b in range(len(batch)) for g in range(num_generations) if int(step0_acts_cpu[b, g]) in _title_set)
                pct_title = _n_title / max(1, len(batch) * num_generations)
            else:
                pct_title = 0.0

            means = rewards_tensor.mean(dim=1, keepdim=True)
            stds  = rewards_tensor.std(dim=1,  keepdim=True).clamp_min(1.0)
            advantages = ((rewards_tensor - means) / stds).clamp(-3.0, 3.0)
            if dynamic_sampling:
                valid = (rewards_tensor.std(dim=1) > 0.1).float().unsqueeze(1)
                advantages = advantages * valid

            # ── Gradient phase: step 0 ────────────────────────────────────────
            # One forward pass for B prompts; gather log_probs for all G sampled
            # actions at once (no extra forward passes for step 0).
            optimizer.zero_grad(set_to_none=True)

            enc0 = tokenizer(step0_prompts, padding=True, truncation=True,
                             max_length=max_prompt_length, return_tensors="pt").to(device)
            if freeze_backbone:
                with torch.no_grad():
                    lh0 = _forward_last_hidden(model, enc0)
                pos0 = enc0["attention_mask"].sum(dim=1) - 1
                h0   = lh0[torch.arange(batch_size, device=device), pos0].float().detach()
            else:
                lh0  = _forward_last_hidden(model, enc0)
                pos0 = enc0["attention_mask"].sum(dim=1) - 1
                h0   = lh0[torch.arange(batch_size, device=device), pos0].float()

            lgt0        = action_head(h0) / max(0.05, temperature)
            lgt0_masked = _mask_guessed_actions(lgt0, batch, action_to_idx, torch)
            if title_mask_cos_threshold > 0.0 and title_word_indices:
                lgt0_masked = _title_mask_logits(lgt0_masked, batch, actions, title_word_indices, similarity_model, title_mask_cos_threshold, torch)
            lgt0_pre_ep_mask = lgt0_masked  # save before episode expansion (for BC)
            if dynamic_expand_k > 0:
                from .text import normalize_word as _nw
                ep_mask_0 = _episode_expand_mask(batch, actions, action_to_idx, similarity_model, base_vocab_size, dynamic_expand_k, device, torch)
                lgt0_masked = lgt0_pre_ep_mask.masked_fill(~ep_mask_0, float('-inf'))
                # fraction of episodes where every title word is reachable in the expanded mask
                n_reachable = sum(
                    1 for b_i, row in enumerate(batch)
                    if all(
                        action_to_idx.get(tw) is not None and ep_mask_0[b_i, action_to_idx[tw]]
                        for tw in (_nw(w) for w in str(row.get("title", "")).split())
                        if tw
                    )
                )
                pct_title_reachable = n_reachable / max(1, len(batch))
            else:
                ep_mask_0 = None
                pct_title_reachable = 0.0
            lp0         = torch.log_softmax(lgt0_masked, dim=-1)   # (B, |A|)
            sampled_lp0 = lp0.gather(1, step0_acts)                # (B, G)

            grpo_weight  = 0.0 if step <= bc_warmup_steps else 1.0
            grpo_loss_0  = -(advantages.detach() * sampled_lp0).mean() / rollout_steps

            distribution    = torch.distributions.Categorical(logits=lgt0_masked)
            logits          = lgt0_masked
            entropy_bonus   = distribution.entropy().mean()
            entropy_deficit = torch.clamp(float(min_entropy) - entropy_bonus, min=0.0)

            tgt0    = _teacher_action_ids(batch, action_to_idx, torch, device)
            if (tgt0 != -100).any():
                if ep_mask_0 is not None:
                    # Ensure oracle target is never masked to -inf in BC: force-unmask it
                    ep_mask_for_bc = ep_mask_0.clone()
                    valid = tgt0 != -100
                    ep_mask_for_bc[valid, tgt0.clamp(min=0)[valid]] = True
                    bc_logits = lgt0_pre_ep_mask.masked_fill(~ep_mask_for_bc, float('-inf'))
                else:
                    bc_logits = lgt0_masked
                bc_loss = torch.nn.functional.cross_entropy(bc_logits, tgt0, ignore_index=-100)
            else:
                bc_loss = lgt0.sum() * 0.0

            if kl_ref_coef > 0.0:
                with torch.no_grad():
                    # Use BASE model hidden states (LoRA disabled) so the reference
                    # distribution stays stable as the LoRA trains.  Without this,
                    # action_head_ref(h0_lora) drifts with the LoRA and the KL
                    # silently collapses to ~0 even when the policy has collapsed.
                    if not freeze_backbone and hasattr(model, "disable_adapter"):
                        with model.disable_adapter():
                            lh0_ref = _forward_last_hidden(model, enc0)
                        h0_ref = lh0_ref[torch.arange(batch_size, device=device), pos0].float()
                    else:
                        h0_ref = h0.detach()  # already frozen base model
                    ref_lgt0 = action_head_ref(h0_ref) / max(0.05, temperature)
                    ref_lgt0 = _mask_guessed_actions(ref_lgt0, batch, action_to_idx, torch)
                    if dynamic_expand_k > 0:
                        ref_lgt0 = ref_lgt0.masked_fill(~ep_mask_0, float('-inf'))
                    ref_lp0  = torch.log_softmax(ref_lgt0, dim=-1)
                    ref_p0   = ref_lp0.exp()
                log_ratio0  = (ref_lp0 - lp0).clamp(max=50.0)
                kl_ref_loss = torch.nan_to_num(
                    ref_p0 * log_ratio0, nan=0.0
                ).sum(-1).mean()
            else:
                kl_ref_loss = lgt0.sum() * 0.0

            loss_0 = (
                grpo_weight      * grpo_loss_0
                + float(bc_coef)          * bc_loss
                - float(entropy_coef)     * entropy_bonus
                + float(min_entropy_coef) * entropy_deficit
                + float(kl_ref_coef)      * kl_ref_loss
            )
            loss_0.backward()

            # ── Gradient phase: steps 1..K-1 (mini-batch accumulation) ────────
            # Each mini-batch calls backward() immediately so only one graph
            # lives in memory at a time.
            MINI = 64
            grpo_later_val = 0.0
            for step_entries in later_steps:
                n = len(step_entries)
                for s in range(0, n, MINI):
                    mini = step_entries[s:s + MINI]
                    mp  = [e[0] for e in mini]
                    ma  = torch.tensor([e[1] for e in mini], device=device)
                    mb  = [e[2] for e in mini]
                    mg  = [e[3] for e in mini]

                    adv_mini = advantages[mb, mg]

                    enc_k = tokenizer(mp, padding=True, truncation=True,
                                      max_length=max_prompt_length, return_tensors="pt").to(device)
                    if freeze_backbone:
                        with torch.no_grad():
                            lh_k = _forward_last_hidden(model, enc_k)
                        pos_k = enc_k["attention_mask"].sum(dim=1) - 1
                        h_k   = lh_k[torch.arange(len(mp), device=device), pos_k].float().detach()
                    else:
                        lh_k  = _forward_last_hidden(model, enc_k)
                        pos_k = enc_k["attention_mask"].sum(dim=1) - 1
                        h_k   = lh_k[torch.arange(len(mp), device=device), pos_k].float()

                    lgt_k  = action_head(h_k) / max(0.05, temperature)
                    if dynamic_expand_k > 0:
                        fake_k = [{"title": str(batch[b].get("title", ""))} for (_, _, b, _) in mini]
                        ep_mask_k = _episode_expand_mask(fake_k, actions, action_to_idx, similarity_model, base_vocab_size, dynamic_expand_k, device, torch)
                        lgt_k = lgt_k.masked_fill(~ep_mask_k, float('-inf'))
                    lp_k   = torch.log_softmax(lgt_k, dim=-1)
                    slp_k  = lp_k.gather(1, ma.unsqueeze(1)).squeeze(1)

                    mini_grpo = (
                        -(adv_mini.detach() * slp_k).sum()
                        / (batch_size * num_generations * rollout_steps)
                    )
                    (grpo_weight * mini_grpo).backward()
                    grpo_later_val += float(mini_grpo.detach())

            grpo_loss = grpo_loss_0.detach() + grpo_later_val
            loss      = loss_0.detach() + grpo_later_val * grpo_weight
            grad_norm = torch.nn.utils.clip_grad_norm_(current_trainable, 1.0)
            optimizer.step()

        else:
            # ── Single-step path ──────────────────────────────────────────────
            mean_shaped_bonus = 0.0
            pct_title_reachable = 0.0
            prompts = [str(row["prompt"]) for row in batch]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            ).to(device)
            if freeze_backbone:
                with torch.no_grad():
                    last_hidden = _forward_last_hidden(model, encoded)
            else:
                last_hidden = _forward_last_hidden(model, encoded)
            positions = encoded["attention_mask"].sum(dim=1) - 1
            hidden = (
                last_hidden[torch.arange(len(batch), device=device), positions].float().detach()
                if freeze_backbone else
                last_hidden[torch.arange(len(batch), device=device), positions].float()
            )
            logits = action_head(hidden) / max(0.05, temperature)
            logits = _mask_guessed_actions(logits, batch, action_to_idx, torch)
            if title_mask_cos_threshold > 0.0 and title_word_indices:
                logits = _title_mask_logits(logits, batch, actions, title_word_indices, similarity_model, title_mask_cos_threshold, torch)
            distribution = torch.distributions.Categorical(logits=logits)
            sampled = distribution.sample((num_generations,)).transpose(0, 1)
            if title_word_indices:
                _title_set = set(title_word_indices)
                _sampled_list = sampled.detach().cpu().tolist()
                _n_title = sum(1 for row_acts in _sampled_list for aid in row_acts if aid in _title_set)
                pct_title = _n_title / max(1, batch_size * num_generations)
            else:
                pct_title = 0.0
            log_probs = torch.log_softmax(logits, dim=-1).gather(1, sampled)
            target_ids = _teacher_action_ids(batch, action_to_idx, torch, device)
            if (target_ids != -100).any():
                bc_loss = torch.nn.functional.cross_entropy(logits, target_ids, ignore_index=-100)
            else:
                bc_loss = logits.sum() * 0.0

            reward_values: list[list[float]] = []
            solved_count = 0
            for row, action_ids in zip(batch, sampled.detach().cpu().tolist()):
                rewards_row = []
                history = json.loads(row["history"]) if isinstance(row["history"], str) else row["history"]
                for action_id in action_ids:
                    word = actions[int(action_id)]
                    reward = score_completion(
                        title=str(row["title"]),
                        intro=str(row["intro"]),
                        history=history,
                        completion=f"MOT: {word}",
                        similarity_model=similarity_model,
                    )
                    rewards_row.append(float(reward.reward))
                    solved_count += int(reward.solved)
                reward_values.append(rewards_row)
            rewards_tensor = torch.tensor(reward_values, dtype=torch.float32, device=device)

            entropy_bonus = distribution.entropy().mean()
            if kl_ref_coef > 0.0:
                with torch.no_grad():
                    if not freeze_backbone and hasattr(model, "disable_adapter"):
                        with model.disable_adapter():
                            lh_ref = _forward_last_hidden(model, encoded)
                        h_ref = lh_ref[torch.arange(len(batch), device=device), positions].float()
                    else:
                        h_ref = hidden.detach()
                    ref_logits = action_head_ref(h_ref) / max(0.05, temperature)
                    ref_logits = _mask_guessed_actions(ref_logits, batch, action_to_idx, torch)
                    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                    ref_probs = ref_log_probs.exp()
                current_log_probs_all = torch.log_softmax(logits, dim=-1)
                log_ratio = (ref_log_probs - current_log_probs_all).clamp(max=50.0)
                kl_ref_loss = torch.nan_to_num(
                    ref_probs * log_ratio, nan=0.0
                ).sum(dim=-1).mean()
            else:
                kl_ref_loss = logits.sum() * 0.0

            # GRPO advantage + loss + optimizer step
            means = rewards_tensor.mean(dim=1, keepdim=True)
            stds  = rewards_tensor.std(dim=1, keepdim=True).clamp_min(1.0)
            advantages = ((rewards_tensor - means) / stds).clamp(-3.0, 3.0)
            if dynamic_sampling:
                valid = (rewards_tensor.std(dim=1) > 0.1).float().unsqueeze(1)
                advantages = advantages * valid
            grpo_loss       = -(advantages.detach() * log_probs).mean()
            entropy_deficit = torch.clamp(float(min_entropy) - entropy_bonus, min=0.0)
            grpo_weight     = 0.0 if step <= bc_warmup_steps else 1.0
            loss = (
                grpo_weight      * grpo_loss
                + float(bc_coef)          * bc_loss
                - float(entropy_coef)     * entropy_bonus
                + float(min_entropy_coef) * entropy_deficit
                + float(kl_ref_coef)      * kl_ref_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(current_trainable, 1.0)
            optimizer.step()

        total_evals = batch_size * num_generations * rollout_steps
        log_row = {
            "step": step,
            "step_time": round(_time.perf_counter() - _t0, 1),
            "loss": round(float(loss.detach().cpu()), 6),
            "grpo_loss": round(float(grpo_loss.detach().cpu()), 6),
            "bc_loss": round(float(bc_loss.detach().cpu()), 6),
            "kl_ref": round(float(kl_ref_loss.detach().cpu()), 4),
            "reward": round(float(rewards_tensor.mean().detach().cpu()), 4),
            "reward_std": round(float(rewards_tensor.std().detach().cpu()), 4),
            "solve_rate": round(solved_count / max(1, total_evals), 4),
            "grad_norm": round(float(grad_norm), 4),
            "entropy": round(float(distribution.entropy().mean().detach().cpu()), 4),
            "pct_title_word": round(pct_title, 4),
            "shaped_bonus": round(mean_shaped_bonus, 4),
            "title_reachable": round(pct_title_reachable, 4),
        }

        if dagger_every > 0 and step % dagger_every == 0:
            if device == "cuda":
                torch.cuda.empty_cache()
            dagger_info = _dagger_refresh(
                model, tokenizer, action_head, actions, action_to_idx, similarity_model, rows, rng,
                n_pages=dagger_pages, rollout_steps=rollout_steps, bc_steps=dagger_bc_steps,
                optimizer=optimizer, device=device, max_prompt_length=max_prompt_length, torch=torch,
            )
            log_row.update(dagger_info)
            print(f"[dagger step={step}] {dagger_info}", flush=True)

        logs.append(log_row)
        print(log_row, flush=True)
        if plot_path and step % plot_every == 0:
            _plot_vocab_reward(logs, plot_path)
        if device == "cuda":
            torch.cuda.empty_cache()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not freeze_backbone:
        model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    torch.save(action_head.state_dict(), output / ACTION_HEAD_WEIGHTS)
    (output / VOCAB_ACTION_CONFIG).write_text(
        json.dumps(
            {
                "base_model_name_or_path": tokenizer_name,
                "actions": actions,
                "hidden_size": hidden_size,
                "head_hidden_size": head_hidden_size,
                "chat_format": "qwen",
                "frozen_backbone": freeze_backbone,
                "use_lm_head": use_lm_head,
                "near_solve_shaping_coef": near_solve_shaping_coef,
                "base_vocab_size": base_vocab_size,
                "dynamic_expand_k": dynamic_expand_k,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if plot_path:
        _plot_vocab_reward(logs, plot_path)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("w", encoding="utf-8") as handle:
            for row in logs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_vocab_action_policy(
    *,
    pages_path: str | Path,
    model_path: str | Path,
    similarity_model: TinyPedantixModel,
    sample_size: int,
    max_steps: int,
    seed: int,
    output_path: str | Path,
    batch_size: int = 16,
    max_prompt_length: int = 384,
    dynamic_expand_k: int = 0,
) -> dict:
    """Evaluate the vocab-action policy by playing full games from empty history.

    dynamic_expand_k > 0 enables test-time dynamic expansion: after each step,
    newly revealed words' TinyModel neighbors are added to that game's action set.
    This mirrors the training ep_mask expansion without leaking the title.
    """
    enforce_hf_cache()
    torch, transformers, _ = _import_vocab_stack()
    model_path = Path(model_path)
    config = json.loads((model_path / VOCAB_ACTION_CONFIG).read_text(encoding="utf-8"))
    actions = [str(word) for word in config["actions"]]
    action_to_idx = {word: idx for idx, word in enumerate(actions)}
    tokenizer_name = str(config["base_model_name_or_path"])
    use_lm_head = bool(config.get("use_lm_head", False))
    base_vocab_size = int(config.get("base_vocab_size") or len(actions))
    if dynamic_expand_k == 0:
        dynamic_expand_k = int(config.get("dynamic_expand_k") or 0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if config.get("frozen_backbone"):
        model = transformers.AutoModelForCausalLM.from_pretrained(tokenizer_name, torch_dtype="auto")
        model.requires_grad_(False)
    else:
        model = _load_peft_model_if_adapter(str(model_path))
    if use_lm_head:
        saved_state = torch.load(model_path / ACTION_HEAD_WEIGHTS, map_location="cpu", weights_only=True)
        action_head = _make_lm_head_module(saved_state["word_embeds"], torch)
    else:
        action_head = _make_action_head(torch, int(config["hidden_size"]), len(actions), int(config.get("head_hidden_size", 0)))
        action_head.load_state_dict(torch.load(model_path / ACTION_HEAD_WEIGHTS, map_location="cpu", weights_only=True))
    device = _best_device(torch)
    model.to(device)
    action_head.to(device)
    model.eval()
    action_head.eval()

    pages = sample_pages(pages_path, sample_size=sample_size, seed=seed)
    histories: list[list[dict]] = [[] for _ in pages]
    done = [False for _ in pages]

    # Test-time dynamic expansion state: per-game set of enabled action indices.
    # Starts at base vocab only; grows as revealed words' neighbors are added.
    if dynamic_expand_k > 0:
        per_game_enabled: list[set[int]] = [set(range(base_vocab_size)) for _ in pages]
        per_game_revealed: list[set[int]] = [set() for _ in pages]  # revealed token indices
    else:
        per_game_enabled = None
        per_game_revealed = None

    for _ in range(max_steps):
        active = [idx for idx, is_done in enumerate(done) if not is_done]
        if not active:
            break
        for start in range(0, len(active), batch_size):
            batch_indices = active[start : start + batch_size]
            prompts = []
            batch_rows = []
            games_for_batch = []
            for idx in batch_indices:
                history = histories[idx]
                game = replay_game(pages[idx], similarity_model, history)
                games_for_batch.append(game)
                prompt = make_prompt(
                    history,
                    max_steps=max_steps,
                    visible_text=compact_visible_text(game),
                )
                prompts.append(format_prompt_for_model(prompt, chat_format="qwen"))
                batch_rows.append({"history": json.dumps(history, ensure_ascii=False)})
            with torch.no_grad():
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length,
                    return_tensors="pt",
                ).to(device)
                last_hidden = _forward_last_hidden(model, encoded)
                positions = encoded["attention_mask"].sum(dim=1) - 1
                hidden = last_hidden[torch.arange(len(batch_indices), device=device), positions].float()
                logits = action_head(hidden)
                logits = _mask_guessed_actions(logits, batch_rows, action_to_idx, torch)
                if dynamic_expand_k > 0:
                    # Apply per-game mask: disable actions outside each game's enabled set
                    for bi, idx in enumerate(batch_indices):
                        enabled = per_game_enabled[idx]
                        mask = torch.zeros(len(actions), dtype=torch.bool, device=device)
                        if enabled:
                            idx_tensor = torch.tensor(sorted(enabled), dtype=torch.long, device=device)
                            mask[idx_tensor] = True
                        logits[bi] = logits[bi].masked_fill(~mask, float('-inf'))
                action_ids = logits.argmax(dim=1).detach().cpu().tolist()
            for i, (idx, action_id, game) in enumerate(zip(batch_indices, action_ids, games_for_batch)):
                word = actions[int(action_id)]
                step = _feedback_step(pages[idx], similarity_model, histories[idx], word)
                histories[idx].append(step)
                if step["solved"]:
                    done[idx] = True
                # Expand per-game action set based on newly revealed tokens
                if dynamic_expand_k > 0 and not step.get("invalid") and not step.get("repeated"):
                    prev_revealed = per_game_revealed[idx]
                    new_game = replay_game(pages[idx], similarity_model, histories[idx])
                    newly_revealed = new_game.revealed - prev_revealed
                    per_game_revealed[idx] = new_game.revealed
                    from .text import normalize_word as _nw_eval
                    for tok_idx in newly_revealed:
                        norm = new_game.tokens[tok_idx].norm
                        if not norm:
                            continue
                        nbr_dict = similarity_model.neighbors.get(norm, {})
                        for nbr in list(nbr_dict.keys())[:dynamic_expand_k]:
                            nbr_idx = action_to_idx.get(nbr)
                            if nbr_idx is not None:
                                per_game_enabled[idx].add(nbr_idx)

    rows = []
    solved = 0
    total_steps = 0
    first_words: Counter[str] = Counter()
    for page, history in zip(pages, histories):
        solved += int(bool(history and history[-1]["solved"]))
        total_steps += len(history)
        for step in history[:10]:
            first_words[str(step["guess"])] += 1
        rows.append({"title": page.title, "solved": bool(history and history[-1]["solved"]), "history": history})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "pages": len(pages),
        "solve_rate": solved / max(1, len(pages)),
        "mean_steps": total_steps / max(1, len(pages)),
        "action_vocab": len(actions),
        "common_first_10_words": first_words.most_common(25),
        "output": str(output),
    }
    return result


def _import_vocab_stack():
    transformers, torch = _import_transformers()
    _, _, peft = _import_training_stack()
    return torch, transformers, peft


def _forward_last_hidden(model, encoded):
    causal_model = getattr(getattr(model, "base_model", None), "model", model)
    transformer = getattr(causal_model, "model", None)
    if transformer is not None:
        outputs = transformer(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        )
        return outputs.last_hidden_state
    outputs = model(**encoded, output_hidden_states=True, logits_to_keep=1, use_cache=False)
    return outputs.hidden_states[-1]


def _load_training_rows(train_jsonl: str | Path) -> list[dict]:
    rows = []
    with Path(train_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty train file: {train_jsonl}")
    return rows


def _build_vocab_actions(
    pages_path: str | Path,
    *,
    max_words: int,
    max_vocab_pages: int = 50000,
    train_jsonl: str | Path | None = None,
    dynamic_expand_k: int = 0,
    similarity_model=None,
) -> tuple[list[str], list[int], int]:
    """Build the action vocab.

    Returns (actions, title_only_indices) where title_only_indices are the indices
    in actions of words appearing only in page titles (not in top completions).

    If train_jsonl is given, actions are taken directly from oracle completion words
    in that file (words the teacher actually guesses).  Falls back to the
    page-frequency approach when train_jsonl is None.
    """
    if train_jsonl is not None:
        from collections import Counter
        from .text import normalize_word as _nw
        counts: Counter[str] = Counter()
        title_words: set[str] = set()
        with Path(train_jsonl).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    word = extract_guess(str(row.get("completion", "")))
                    if word and is_valid_guess(word):
                        counts[word] += 1
                    for raw_word in str(row.get("title", "")).split():
                        w = _nw(raw_word)
                        if w and is_valid_guess(w):
                            title_words.add(w)
        top_words = [w for w, _ in counts.most_common(max_words)]
        top_words_set = set(top_words)
        title_only_set = title_words - top_words_set
        # Merge: base vocab + title-only words + dynamic neighbors of title words
        merged = top_words + [w for w in sorted(title_words) if w not in top_words_set]
        base_vocab_size = len(top_words)
        if dynamic_expand_k > 0 and similarity_model is not None:
            merged_set = set(merged)
            neighbor_words: set[str] = set()
            for tw in title_words:
                top_k = sorted(
                    similarity_model.neighbors.get(tw, {}).items(),
                    key=lambda x: -x[1],
                )[:dynamic_expand_k]
                for nw, _ in top_k:
                    if nw not in merged_set:
                        neighbor_words.add(nw)
            merged = merged + sorted(neighbor_words)
        actions = [w for w in merged if extract_guess(f"MOT: {w}") == w]
        if actions:
            n_title_extra = len(title_only_set)
            n_neighbors = len(actions) - base_vocab_size - n_title_extra
            print(
                f"action vocab: {len(actions)} words "
                f"({base_vocab_size} base + {n_title_extra} title-only + {max(0, n_neighbors)} neighbors)",
                flush=True,
            )
            title_only_indices = [i for i, w in enumerate(actions) if w in title_only_set]
            return actions, title_only_indices, base_vocab_size

    pages = sample_pages(pages_path, sample_size=max_vocab_pages, seed=0)
    vocabulary, _ = build_action_space(pages, max_words=max_words)
    actions = []
    for word in vocabulary:
        parsed = extract_guess(f"MOT: {word}")
        if parsed == word and word not in actions and is_valid_guess(word):
            actions.append(word)
    if not actions:
        raise ValueError("empty action vocabulary")
    return actions, [], len(actions)


def _mask_guessed_actions(logits, rows: list[dict], action_to_idx: dict[str, int], torch):
    masked = logits.clone()
    for row_idx, row in enumerate(rows):
        raw_history = row.get("history", "[]")
        history = json.loads(raw_history) if isinstance(raw_history, str) else raw_history
        for step in history:
            action_idx = action_to_idx.get(str(step.get("guess", "")))
            if action_idx is not None:
                masked[row_idx, action_idx] = -torch.inf
    return masked


def _episode_expand_mask(
    batch: list[dict],
    actions: list[str],
    action_to_idx: dict[str, int],
    similarity_model,
    base_vocab_size: int,
    expand_k: int,
    device: str,
    torch,
):
    """Return (B, N) bool mask: True = allowed for this episode.

    Base vocab (first base_vocab_size words) is always allowed.
    For each episode, also allow: its title words + their top-expand_k TinyModel
    neighbors.  This lets the model actually guess (and solve) episodes whose
    title word is not in the base frequency vocab.
    """
    from .text import normalize_word as _nw
    B, N = len(batch), len(actions)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask[:, :base_vocab_size] = True
    for b, row in enumerate(batch):
        title = str(row.get("title", ""))
        for raw_word in title.split():
            tw = _nw(raw_word)
            if not tw:
                continue
            if tw in action_to_idx:
                mask[b, action_to_idx[tw]] = True
            if similarity_model is not None and expand_k > 0:
                top_k = sorted(
                    similarity_model.neighbors.get(tw, {}).items(),
                    key=lambda x: -x[1],
                )[:expand_k]
                for nw, _ in top_k:
                    if nw in action_to_idx:
                        mask[b, action_to_idx[nw]] = True
    return mask


def _title_mask_logits(
    logits,
    batch: list[dict],
    actions: list[str],
    title_word_indices: list[int],
    similarity_model,
    threshold: float,
    torch,
):
    """Per-row -inf bias for title words whose max similarity to revealed content is below threshold."""
    biased = logits.clone()
    for b, row in enumerate(batch):
        history = json.loads(row["history"]) if isinstance(row["history"], str) else row["history"]
        revealed = {
            str(step["guess"])
            for step in history
            if step.get("guess") and not step.get("invalid") and not step.get("repeated")
        }
        if not revealed:
            continue
        max_sim: dict[str, float] = {}
        for r in revealed:
            for neighbor, score in similarity_model.neighbors.get(r, {}).items():
                if score > max_sim.get(neighbor, -1.0):
                    max_sim[neighbor] = score
        for idx in title_word_indices:
            word = actions[idx]
            if max_sim.get(word, 0.0) < threshold:
                biased[b, idx] = -torch.inf
    return biased


def _teacher_action_ids(rows: list[dict], action_to_idx: dict[str, int], torch, device):
    ids = []
    for row in rows:
        word = extract_guess(str(row.get("completion", "")))
        ids.append(action_to_idx.get(word, -100))
    return torch.tensor(ids, dtype=torch.long, device=device)


def _plot_vocab_reward(logs: list[dict], plot_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import math
    except ImportError:
        return
    if not logs:
        return
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in logs]
    rewards = [r["reward"] for r in logs]
    entropies = [r["entropy"] for r in logs]
    bc_losses = [r["bc_loss"] for r in logs]

    # EMA
    alpha = 0.05
    ema_reward: list[float] = []
    ema_val = rewards[0]
    for v in rewards:
        ema_val = alpha * v + (1 - alpha) * ema_val
        ema_reward.append(ema_val)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(steps, rewards, color="#93c5fd", linewidth=0.8, alpha=0.5, label="reward (raw)")
    ax1.plot(steps, ema_reward, color="#1d4ed8", linewidth=2.0, label="reward (EMA)")
    ax1.set_ylabel("reward")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.25)

    ax2.plot(steps, entropies, color="#f97316", linewidth=1.2, label="entropy")
    ax2_bc = ax2.twinx()
    ax2_bc.plot(steps, bc_losses, color="#9ca3af", linewidth=1.0, linestyle="--", label="bc_loss")
    ax2.set_ylabel("entropy")
    ax2_bc.set_ylabel("bc_loss")
    ax2.set_xlabel("step")
    lines = ax2.get_lines() + ax2_bc.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=100)
    plt.close(fig)
