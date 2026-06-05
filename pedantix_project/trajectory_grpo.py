"""Trajectory GRPO: model plays full games from turn 0; updates from its own rollouts.

The bandit GRPO failure mode was a state distribution mismatch — training on
frozen oracle snapshots (median turn 18, lots already revealed), eval starts at
turn 0 on fresh pages. The model never learned cold-start behaviour.

This implementation keeps a pool of N live games. Each training step:
  1. Sample B games from the pool, build prompts from their current state.
  2. Generate G constrained completions per prompt (full fr_FR.dic trie).
  3. Score each completion with the standard reward function.
  4. Group-normalize advantages within each prompt (the "GR" in GRPO).
  5. Compute KL to the frozen ref policy (LoRA-disabled forward).
  6. Loss = -E[adv * logprob_completion] + beta * KL, single on-policy step.
  7. Advance each of the B games by playing one of its G sampled candidates.
  8. Reset games that solved or hit max_steps.

Single on-policy update per rollout (no PPO inner-epoch loop) ⇒ ratio = 1
exactly, so we drop the clip. This is the cleanest form of GRPO and matches
TRL's num_iterations=1 default.

Masking invariant: loss + KL include ONLY completion tokens, up to and
including the first EOS. Prompt tokens contribute 0. The constrained trie is
plugged into model.generate via prefix_allowed_tokens_fn — same trie used at
eval, so train/test distributions match.
"""
from __future__ import annotations

import gc
import json
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .dataset import WikiPage
from .llm_policy import (
    INVALID_GUESSES,
    _feedback_step,
    _load_similarity_model,
    _qwen_no_think_generation_kwargs,
    compact_visible_text,
    enforce_hf_cache,
    extract_guess,
    format_prompt_for_model,
    make_prompt,
    replay_game,
    score_completion,
)
from .rl import STOPWORDS


# ---------------------------------------------------------------------------
# OnlineGameBuffer
# ---------------------------------------------------------------------------

@dataclass
class LiveGame:
    page: WikiPage
    history: list[dict] = field(default_factory=list)
    plays: int = 0  # how many times the model has played in this game (==len(history))


class OnlineGameBuffer:
    """Pool of N live games. The model generates its own state distribution."""

    # Curriculum EMA config: pages with reward EMA in [lo, hi] are "learnable".
    # Pages outside this window are de-weighted (not excluded) so the pool stays
    # diverse. Alpha controls how fast the EMA reacts to new reward observations.
    _CURRICULUM_LO = -80.0
    _CURRICULUM_HI = 20.0
    _CURRICULUM_ALPHA = 0.2   # EMA smoothing (lower = slower to update)
    _CURRICULUM_WEIGHT = 4.0  # learnable pages are this many times more likely

    def __init__(
        self,
        pages: list[dict],
        sim_model,
        *,
        pool_size: int = 64,
        max_steps: int = 30,
        seed: int = 42,
        seed_turns: int = 0,
        max_seed_turns: int | None = None,
        curriculum: bool = False,
    ) -> None:
        self.pages = pages  # list of {"title": ..., "intro": ...} dicts
        self.sim_model = sim_model
        self.pool_size = pool_size
        self.max_steps = max_steps
        self.seed_turns = seed_turns
        # E: variable seeding — sample uniformly in [0, max_seed_turns] per game.
        # If max_seed_turns is None, use fixed seed_turns (original behaviour).
        self.max_seed_turns = max_seed_turns
        # D: curriculum sampling — weight pages by recent reward EMA.
        self.curriculum = curriculum
        self._page_ema: dict[str, float] = {}  # title → reward EMA
        self.rng = random.Random(seed)
        self.games: list[LiveGame] = []
        self.games_completed = 0
        self.games_solved = 0
        self.completed_trajectories: deque[dict] = deque()
        self._fill()

    # ── D: curriculum page sampling ──────────────────────────────────────────

    def _page_weight(self, page: dict) -> float:
        """Return sampling weight for a page based on its reward EMA.

        Pages with no history get weight 1.0 (neutral — we don't know yet).
        Pages in the learnable zone get _CURRICULUM_WEIGHT.
        Pages outside get 1.0 (still sampled, just less often).
        """
        if not self.curriculum:
            return 1.0
        ema = self._page_ema.get(page["title"])
        if ema is None:
            return 1.0
        if self._CURRICULUM_LO <= ema <= self._CURRICULUM_HI:
            return self._CURRICULUM_WEIGHT
        return 1.0

    def _sample_page(self) -> dict:
        if not self.curriculum:
            return self.rng.choice(self.pages)
        weights = [self._page_weight(p) for p in self.pages]
        return self.rng.choices(self.pages, weights=weights, k=1)[0]

    def _update_ema(self, title: str, reward: float) -> None:
        prev = self._page_ema.get(title)
        if prev is None:
            self._page_ema[title] = reward
        else:
            self._page_ema[title] = (
                self._CURRICULUM_ALPHA * reward + (1 - self._CURRICULUM_ALPHA) * prev
            )

    # ── E: variable oracle seeding ───────────────────────────────────────────

    def _seed_game(self, game: LiveGame, n_turns: int) -> None:
        """Pre-play n_turns turns with intro-word oracle to build context."""
        if n_turns == 0:
            return
        title_words = set(re.sub(r'[^a-zA-ZÀ-ÿ ]', '', game.page.title.lower()).split())
        intro_words = re.findall(r'\b[a-zA-ZÀ-ÿ]{5,}\b', game.page.intro.lower())
        candidates = [w for w in dict.fromkeys(intro_words) if w not in title_words]
        self.rng.shuffle(candidates)
        for guess in candidates[:n_turns]:
            step = _feedback_step(game.page, self.sim_model, game.history, guess)
            if step.get("solved"):
                game.history = []
                game.plays = 0
                return
            game.history.append(step)
            game.plays = len(game.history)

    def _draw_seed_turns(self) -> int:
        if self.max_seed_turns is not None:
            # E: uniform in [0, max_seed_turns]
            return self.rng.randint(0, self.max_seed_turns)
        return self.seed_turns

    def _new_game(self) -> LiveGame:
        p = self._sample_page()
        game = LiveGame(page=WikiPage(title=p["title"], intro=p["intro"]))
        n = self._draw_seed_turns()
        if n > 0:
            self._seed_game(game, n)
        return game

    def _fill(self) -> None:
        while len(self.games) < self.pool_size:
            self.games.append(self._new_game())

    def sample_batch(self, batch_size: int) -> list[tuple[int, LiveGame]]:
        """Return [(buffer_idx, game), ...] of size min(batch_size, pool_size)."""
        n = min(batch_size, len(self.games))
        idxs = self.rng.sample(range(len(self.games)), n)
        return [(i, self.games[i]) for i in idxs]

    def advance(self, idx: int, guess: str) -> dict:
        """Apply `guess` to game at idx. Reset if solved or out of turns.

        Returns the feedback dict so the caller can log it.
        """
        g = self.games[idx]
        step = _feedback_step(g.page, self.sim_model, g.history, guess)
        g.history.append(step)
        g.plays = len(g.history)
        solved = bool(step.get("solved"))
        reward = step.get("reward", 0.0)
        if solved or g.plays >= self.max_steps:
            self.games_completed += 1
            if solved:
                self.games_solved += 1
            # D: update EMA for curriculum before replacing the game
            self._update_ema(g.page.title, reward)
            self.completed_trajectories.append({
                "title": g.page.title,
                "history": list(g.history),
                "solved": solved,
                "n_turns": len(g.history),
            })
            self.games[idx] = self._new_game()
        return step


# ---------------------------------------------------------------------------
# DAgger rescue: oracle hint when the policy is spiraling
# ---------------------------------------------------------------------------

def _oracle_rescue_guess(meta_dict: dict, similarity_model) -> str | None:
    """Pick a recovery word for a spiraling game.

    Strategy: from the intro vocabulary (5+ chars, not function/title/guessed),
    return the word with highest FastText similarity to ANY title canon. This
    is a "hot/cold" hint — the model sees a guess that scores moderately well,
    breaking the cycle of wrong-domain associations.

    Returns None if no candidate words remain (e.g. intro fully exhausted).
    """
    title = meta_dict["title"]
    intro = meta_dict["intro"]
    history = meta_dict["history"]

    guessed = {str(h.get("guess", "")).lower() for h in history}
    title_canons = [w for w in re.sub(r'[^a-zA-ZÀ-ÿ ]', ' ', title.lower()).split() if w]
    title_canon_set = set(title_canons)

    intro_words = re.findall(r'\b[a-zA-ZÀ-ÿ]{5,}\b', intro.lower())
    candidates = [
        w for w in dict.fromkeys(intro_words)
        if w not in guessed and w not in title_canon_set and w not in STOPWORDS
    ]
    if not candidates:
        return None

    if title_canons and hasattr(similarity_model, "similarity"):
        best_word, best_score = None, -1.0
        for w in candidates[:200]:
            try:
                s = max(float(similarity_model.similarity(w, t)) for t in title_canons)
            except Exception:
                continue
            if s > best_score:
                best_word, best_score = w, s
        if best_word is not None:
            return best_word

    return candidates[0]


# ---------------------------------------------------------------------------
# Trajectory GRPO training loop
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryGRPOArgs:
    pages_jsonl: str
    model_name_or_path: str
    tiny_model_path: str
    output_dir: str

    max_steps: int = 3000
    batch_size: int = 4
    num_generations: int = 4
    # Chunk the policy+ref forward/backward across micro-batches of this size.
    # Gradients accumulate (no zero_grad between chunks) and the optimizer
    # steps once per training step. None = run the full B*G in a single
    # forward (original behavior). Use a small value (1 or 2) when activation
    # memory blows up under eager attention or long completions.
    micro_batch_size: int | None = None
    max_completion_length: int = 8
    learning_rate: float = 1e-6
    beta: float = 0.1  # raised from 0.05 — KL constraint must resist drift
    pool_size: int = 64
    game_max_steps: int = 30
    seed: int = 42

    temperature: float = 0.9
    top_p: float = 0.9
    lora_rank: int = 32
    seed_turns: int = 0  # pre-play N turns with intro-word oracle before GRPO takes over
    # E: if set, each game draws seed turns uniformly from [0, max_seed_turns]
    # instead of the fixed seed_turns value.
    max_seed_turns: int | None = None
    # D: curriculum sampling — bias page selection toward learnable pages.
    curriculum: bool = False
    # DAgger rescue: when a game has this many consecutive negative-reward
    # turns, replace the worst of the G generations with an oracle hint
    # (intro word with highest FastText similarity to the title). The model
    # gets positive advantage on that slot → learns to escape spirals.
    # None or 0 = disabled.
    dagger_rescue_threshold: int | None = None
    solve_bonus_scale: float = 1.0

    # Pool of allowed pages (None = all)
    top_k_pages: int | None = 20000
    min_intro_chars: int = 300

    # Logging / eval
    log_path: str | None = None
    plot_path: str | None = None
    logging_steps: int = 1
    save_steps: int = 100
    eval_pages_path: str | None = None
    eval_every_n_steps: int = 100
    eval_pages: int = 20
    eval_max_game_steps: int = 30
    eval_chat_format: str = "qwen"
    eval_num_generations: int = 2
    eval_batch_size: int = 2

    chat_format: str = "qwen"  # how prompts are formatted during training

    # Kept for backward compat but should be 0.0 — contrastive backfired.
    contrastive_weight: float = 0.0

    # Fluency floor: penalize guesses whose log-prob under the reference model
    # is below this threshold. Directly closes the obscure-word loophole —
    # gibberish or extremely rare tokens that the base model would never emit
    # get a bounded penalty proportional to how surprising they are.
    # Set to None to disable. Typical value: -15.0 (log-prob).
    fluency_floor_logprob: float | None = None
    fluency_floor_weight: float = 2.0  # penalty = weight * max(0, floor - logprob)

    # KL alarm: if KL exceeds kl_alarm_threshold in a single step, roll back to
    # the last saved checkpoint and cut learning_rate by kl_lr_cut_factor.
    # This catches the step-1000-style catastrophic drift before it locks in.
    kl_alarm_threshold: float = 50.0
    kl_lr_cut_factor: float = 0.5


def _load_filtered_pages(args: TrajectoryGRPOArgs) -> list[dict]:
    rows: list[dict] = []
    with open(args.pages_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    if args.top_k_pages is not None:
        rows = rows[: args.top_k_pages]
    rows = [r for r in rows if len(r.get("intro", "")) >= args.min_intro_chars]
    return rows


def _build_constrained_trie(tokenizer):
    """Build the French trie once; return (trie, eos_id) for per-step constraint creation."""
    from .constrained_decoding import load_french_words, build_trie
    words = load_french_words(extra_exclusions=STOPWORDS | INVALID_GUESSES)
    trie = build_trie(tokenizer, words)
    print(f"[constrained_decoding] trie built: {len(words)} words, {len(trie)} root tokens",
          flush=True)
    return trie


def train_llm_grpo_trajectory(args: TrajectoryGRPOArgs) -> None:
    enforce_hf_cache()
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else out_dir / "training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    # ---- similarity model + pages ----
    similarity_model = _load_similarity_model(args.tiny_model_path)
    pages = _load_filtered_pages(args)
    print(f"[traj-GRPO] page pool: {len(pages)} pages", flush=True)
    buffer = OnlineGameBuffer(
        pages=pages,
        sim_model=similarity_model,
        pool_size=args.pool_size,
        max_steps=args.game_max_steps,
        seed=args.seed,
        seed_turns=args.seed_turns,
        max_seed_turns=args.max_seed_turns,
        curriculum=args.curriculum,
    )
    if args.max_seed_turns is not None:
        print(f"[traj-GRPO] variable oracle seeding: uniform [0, {args.max_seed_turns}] turns per game", flush=True)
    elif args.seed_turns > 0:
        print(f"[traj-GRPO] oracle pre-seeding: {args.seed_turns} turns per game", flush=True)
    if args.curriculum:
        print("[traj-GRPO] curriculum sampling enabled: learnable pages weighted 4x", flush=True)

    # ---- model + tokenizer ----
    adapter_cfg = Path(args.model_name_or_path) / "adapter_config.json"
    is_lora_checkpoint = adapter_cfg.exists()
    if is_lora_checkpoint:
        from peft import PeftConfig
        peft_cfg = PeftConfig.from_pretrained(args.model_name_or_path)
        tokenizer_name = peft_cfg.base_model_name_or_path
    else:
        tokenizer_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build bad_words_ids for French function words (articles, prepositions, aux verbs).
    # These always score near zero similarity and waste turns — ban them at generation
    # time so the model never spends a turn on them.
    _FW = {
        "le", "la", "les", "l", "un", "une", "des", "du", "de", "d",
        "en", "et", "ou", "au", "aux", "dans", "par", "sur", "avec",
        "pour", "sans", "est", "sont", "a", "ont", "qui", "que", "se",
        "ne", "ce", "il", "elle", "ils", "elles",
    }
    fw_banned_ids: list[list[int]] = []
    for _w in _FW:
        for _pfx in ("", " "):
            _ids = tokenizer.encode(_pfx + _w, add_special_tokens=False)
            if _ids:
                fw_banned_ids.append(_ids)
    print(f"[traj-GRPO] {len(fw_banned_ids)} function-word token seqs banned from generation", flush=True)

    print(f"[traj-GRPO] loading base model from {tokenizer_name}", flush=True)
    # FA2 handles padding via varlen kernels — no NaN on fully-masked rows.
    # SDPA is NOT a safe fallback: NaN grads confirmed with left-padded batches
    # on PyTorch 2.8 (pytorch#103749, #109517, #125674).
    try:
        import flash_attn as _fa
        attn_impl = "flash_attention_2"
        print(f"[traj-GRPO] flash_attn {_fa.__version__} found → flash_attention_2", flush=True)
    except ImportError:
        attn_impl = "eager"
        print("[traj-GRPO] flash_attn not found → eager (not sdpa)", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        tokenizer_name, torch_dtype="auto", attn_implementation=attn_impl
    )
    # Enable gradient checkpointing BEFORE wrapping with PEFT, so PEFT's
    # _prepare_model_for_gradient_checkpointing() wires up enable_input_require_grads
    # during __init__. Calling it after get_peft_model() leaves the input embeds
    # without a working requires_grad hook (transformers#42947, peft#2398).
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if is_lora_checkpoint:
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path, is_trainable=True)
        print(f"[traj-GRPO] resumed LoRA adapter from {args.model_name_or_path}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base_model, lora_cfg)
        print(f"[traj-GRPO] fresh LoRA (rank={args.lora_rank}) on base", flush=True)

    # Belt-and-suspenders: idempotent re-call on the PeftModel ensures the
    # hook is on the right module even if PEFT's __init__ path missed it.
    model.enable_input_require_grads()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    use_think = args.chat_format == "qwen-think"
    gen_extra_kwargs = _qwen_no_think_generation_kwargs(tokenizer, tokenizer_name) if not use_think else {}
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # ---- optimizer ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    # ---- training loop ----
    print(f"[traj-GRPO] starting {args.max_steps} steps "
          f"(B={args.batch_size}, G={args.num_generations}, beta={args.beta}, lr={args.learning_rate})",
          flush=True)
    t_start = time.time()
    running_reward = 0.0
    running_kl = 0.0
    running_solve = 0.0
    EMA = 0.05
    seen = 0
    rng = random.Random(args.seed + 1)
    last_good_ckpt: str | None = None  # path to last checkpoint before a KL alarm
    _last_preds_key: tuple = (None, -1)  # (page_title, turn) — skip PREDS if unchanged

    for step in range(1, args.max_steps + 1):
        # --- sample live games ---
        sampled = buffer.sample_batch(args.batch_size)
        if not sampled:
            break
        idxs = [i for i, _ in sampled]
        games = [g for _, g in sampled]

        # --- build prompts ---
        prompts: list[str] = []
        meta: list[dict] = []  # (title, intro, history) for reward
        for g in games:
            visible = compact_visible_text(replay_game(g.page, similarity_model, g.history))
            raw_prompt = make_prompt(g.history, max_steps=args.game_max_steps, visible_text=visible)
            prompts.append(format_prompt_for_model(raw_prompt, chat_format=args.chat_format))
            meta.append({
                "title": g.page.title,
                "intro": g.page.intro,
                "history": list(g.history),
            })

        # --- tokenize (left-padded) ---
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048, truncation_side="left",
        ).to(device)
        prompt_ids = enc["input_ids"]            # [B, P]
        prompt_attn = enc["attention_mask"]      # [B, P]
        P = prompt_ids.shape[1]

        # --- generate G samples per prompt (batched) ---
        # Batched for speed; function words banned globally via bad_words_ids;
        # temperature=1.3 provides within-group diversity without sequential overhead.
        G = args.num_generations
        model.eval()
        model.config.use_cache = True
        generate_kwargs: dict = dict(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=args.max_completion_length,
            min_new_tokens=3,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=G,
            pad_token_id=eos_id,
        )
        if fw_banned_ids:
            generate_kwargs["bad_words_ids"] = fw_banned_ids
        generate_kwargs.update(gen_extra_kwargs)
        with torch.no_grad():
            gen_out = model.generate(**generate_kwargs)
        model.config.use_cache = False
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # gen_out: [B*G, P+L]  row k → prompt k // G
        BG = gen_out.shape[0]
        L = gen_out.shape[1] - P
        completion_ids = gen_out[:, P:]  # [B*G, L]

        # --- mask: 1 on tokens up to AND including first EOS ---
        eos_mask = (completion_ids == eos_id).int()
        cum_eos = eos_mask.cumsum(dim=1)
        # valid where no EOS seen yet, OR exactly at first EOS
        completion_mask = ((cum_eos == 0) | ((cum_eos == 1) & eos_mask.bool())).float()
        # length of each completion (incl. EOS)
        comp_lens = completion_mask.sum(dim=1).long()

        # --- score each completion ---
        rewards_list: list[float] = []
        guesses_list: list[str] = []
        solved_list: list[bool] = []
        for k in range(BG):
            comp_text = tokenizer.decode(completion_ids[k], skip_special_tokens=True)
            m = meta[k // G]
            guess = extract_guess(comp_text)
            r = score_completion(
                title=m["title"], intro=m["intro"], history=m["history"],
                completion=f"MOT: {guess}",
                similarity_model=similarity_model,
                solve_bonus_scale=args.solve_bonus_scale,
            )
            rewards_list.append(r.reward)
            guesses_list.append(guess)
            solved_list.append(bool(r.solved))

        # Fluency floor (disabled by default): penalize guesses with very low
        # log-prob under the reference model. If obscure-word exploit recurs,
        # this is the right shape of fix — bounded penalty tied to reference,
        # not a corpus statistic the model can chase.
        # See TrajectoryGRPOArgs.fluency_floor_logprob to enable.

        # --- DAgger rescue: oracle hint when a game is spiraling ---
        rescued_log: list[tuple[str, str, float]] = []
        if args.dagger_rescue_threshold and args.dagger_rescue_threshold > 0:
            thr = args.dagger_rescue_threshold
            for b in range(args.batch_size):
                m = meta[b]
                hist = m["history"]
                if len(hist) < thr:
                    continue
                recent = hist[-thr:]
                if not all(float(h.get("reward", 0.0)) < 0 for h in recent):
                    continue
                oracle_word = _oracle_rescue_guess(m, similarity_model)
                if not oracle_word:
                    continue
                # Find the worst slot in this game's G generations
                slot0 = b * G
                worst_off = min(range(G), key=lambda i: rewards_list[slot0 + i])
                k = slot0 + worst_off
                # Tokenize the oracle word + EOS, pad to L with EOS (mask zeros out the rest)
                tok_ids = tokenizer.encode(oracle_word, add_special_tokens=False)
                tok_ids = tok_ids[: max(1, L - 1)]
                tok_ids.append(eos_id)
                pad_len = L - len(tok_ids)
                if pad_len > 0:
                    tok_ids = tok_ids + [eos_id] * pad_len
                oracle_tensor = torch.tensor(tok_ids[:L], dtype=gen_out.dtype, device=device)
                # gen_out[:, P:] is a view of completion_ids → writing via gen_out keeps both in sync
                gen_out[k, P:] = oracle_tensor
                # Recompute mask for this row only (1 up to and including first EOS)
                eos_mask_k = (oracle_tensor == eos_id).int()
                cum_k = eos_mask_k.cumsum(dim=0)
                completion_mask[k] = ((cum_k == 0) | ((cum_k == 1) & eos_mask_k.bool())).float()
                # Rescore with oracle guess
                r_oracle = score_completion(
                    title=m["title"], intro=m["intro"], history=m["history"],
                    completion=f"MOT: {oracle_word}",
                    similarity_model=similarity_model,
                    solve_bonus_scale=args.solve_bonus_scale,
                )
                rewards_list[k] = r_oracle.reward
                guesses_list[k] = oracle_word
                solved_list[k] = bool(r_oracle.solved)
                rescued_log.append((m["title"], oracle_word, r_oracle.reward))
            # Recompute completion lengths after any rescue mutations
            comp_lens = completion_mask.sum(dim=1).long()
            if rescued_log and step % 5 == 0:
                rescued_str = " | ".join(
                    f"{t!r}→{w}({r:+.1f})" for t, w, r in rescued_log
                )
                print(f"[dagger-rescue step={step}] {rescued_str}", flush=True)

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)  # [B*G]

        # --- print predictions every 5 steps (first game in batch, skip if not progressed) ---
        if step % 5 == 0:
            preds_key = (meta[0]["title"], len(meta[0]["history"]))
            if preds_key != _last_preds_key:
                _last_preds_key = preds_key
                b0_history = meta[0]["history"]  # full history
                hist_str = " → ".join(
                    f"{h['guess']}(s={h.get('semantic', 0)},e={h.get('exact', 0)},r={h.get('reward', 0):+.1f})"
                    for h in b0_history
                ) or "(no history yet)"
                best_g = max(range(G), key=lambda gi: rewards_list[gi])
                guesses_str = " | ".join(
                    f"{guesses_list[gi]}({rewards_list[gi]:+.1f}){'*' if gi == best_g else ''}"
                    for gi in range(G)
                )
                print(
                    f"[PREDS step={step}] page={meta[0]['title']!r} turn={len(meta[0]['history']) + 1}\n"
                    f"  hist: {hist_str}\n"
                    f"  guesses: {guesses_str}",
                    flush=True,
                )

        # --- group-relative advantages (GRPO) ---
        rewards_grp = rewards.view(args.batch_size, G)  # [B, G]
        mean_grp = rewards_grp.mean(dim=1, keepdim=True)
        std_grp = rewards_grp.std(dim=1, keepdim=True).clamp(min=1e-4)
        adv_grp = (rewards_grp - mean_grp) / std_grp
        advantages = adv_grp.view(-1)  # [B*G]

        # --- attention mask over full sequence (prompt + completion-up-to-EOS) ---
        prompt_attn_g = prompt_attn.repeat_interleave(G, dim=0)  # [B*G, P]
        completion_attn = completion_mask.long()                  # [B*G, L]
        full_attn = torch.cat([prompt_attn_g, completion_attn], dim=1)  # [B*G, P+L]

        # --- chunked policy + ref forward/backward ---
        # Compute the global mask denominator once so per-chunk losses scale
        # such that summing all chunk_losses == the original full-batch loss.
        # Gradients accumulate across chunks; optimizer.step() runs once.
        denom = completion_mask.sum().clamp(min=1.0)
        micro_bs = args.micro_batch_size if args.micro_batch_size else BG
        micro_bs = max(1, min(micro_bs, BG))

        optimizer.zero_grad(set_to_none=True)

        total_policy_loss_val = 0.0
        total_kl_val = 0.0
        any_nan_loss = False
        # Save logits of last chunk for entropy logging (cheap; small slice).
        last_chunk_logits_for_entropy: "torch.Tensor | None" = None
        last_chunk_mask_for_entropy: "torch.Tensor | None" = None

        for start in range(0, BG, micro_bs):
            end = min(start + micro_bs, BG)
            chunk_ids = gen_out[start:end]                      # [m, P+L]
            chunk_attn = full_attn[start:end]                   # [m, P+L]
            chunk_comp_ids = completion_ids[start:end]          # [m, L]
            chunk_mask = completion_mask[start:end]             # [m, L]
            chunk_adv = advantages[start:end]                   # [m]

            # Policy forward (gradient on)
            out = model(input_ids=chunk_ids, attention_mask=chunk_attn)
            chunk_logits = out.logits[:, P - 1 : P - 1 + L, :]  # [m, L, V]
            if chunk_logits.isnan().any() or chunk_logits.isinf().any():
                nan_frac = chunk_logits.isnan().float().mean().item()
                print(f"[NaN-DIAG] step={step} chunk={start}:{end} comp_logits NaN={nan_frac:.3f}", flush=True)
                chunk_logits = torch.nan_to_num(chunk_logits, nan=0.0, posinf=65504.0, neginf=-65504.0)
            chunk_log_probs = F.log_softmax(chunk_logits.float(), dim=-1)
            policy_token_logp = chunk_log_probs.gather(
                2, chunk_comp_ids.unsqueeze(-1)
            ).squeeze(-1)
            policy_token_logp = policy_token_logp * chunk_mask

            # Reference forward (LoRA disabled, no grad)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_out = model(input_ids=chunk_ids, attention_mask=chunk_attn)
                    ref_logits = ref_out.logits[:, P - 1 : P - 1 + L, :]
                    ref_logits = torch.nan_to_num(ref_logits, nan=0.0, posinf=65504.0, neginf=-65504.0)
                    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
                    ref_token_logp = ref_log_probs.gather(
                        2, chunk_comp_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    ref_token_logp = ref_token_logp * chunk_mask

            # GRPO KL estimator (low variance), normalized by global denom
            log_diff = (ref_token_logp - policy_token_logp).clamp(min=-10.0, max=10.0)
            kl_per_token = torch.exp(log_diff) - log_diff - 1.0
            chunk_kl = (kl_per_token * chunk_mask).sum() / denom

            # Policy loss contribution, normalized by global denom
            chunk_policy = -(
                (chunk_adv.unsqueeze(1) * policy_token_logp) * chunk_mask
            ).sum() / denom

            chunk_loss = chunk_policy + args.beta * chunk_kl

            if chunk_loss.isnan() or chunk_loss.isinf():
                print(f"[NaN-DIAG] step={step} chunk={start}:{end} loss=nan/inf — skipping chunk", flush=True)
                any_nan_loss = True
                # Free graph for this chunk
                del chunk_logits, chunk_log_probs, policy_token_logp
                del ref_logits, ref_log_probs, ref_token_logp
                del chunk_kl, chunk_policy, chunk_loss
                continue

            chunk_loss.backward()
            total_policy_loss_val += float(chunk_policy.detach().item())
            total_kl_val += float(chunk_kl.detach().item())

            # Keep last chunk's logits-detached for entropy logging.
            last_chunk_logits_for_entropy = chunk_logits.detach()
            last_chunk_mask_for_entropy = chunk_mask

            # Free graph between chunks
            del chunk_logits, chunk_log_probs, policy_token_logp
            del ref_logits, ref_log_probs, ref_token_logp
            del chunk_kl, chunk_policy, chunk_loss, out, ref_out

        if any_nan_loss and total_policy_loss_val == 0.0 and total_kl_val == 0.0:
            print(f"[NaN-DIAG] step={step} all chunks NaN — skipping optimizer step", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        # Synthesize summary scalars for logging (sums of per-chunk contributions
        # = the original full-batch loss values).
        policy_loss = torch.tensor(total_policy_loss_val, device=device)
        kl_loss = torch.tensor(total_kl_val, device=device)
        loss = policy_loss + args.beta * kl_loss

        # Zero NaN/inf gradients before clipping — gradient checkpointing with
        # bf16 can produce NaN grads in deep layers; zeroing lets the rest update.
        n_nan_grads = 0
        for p in trainable:
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                p.grad.zero_()
                n_nan_grads += 1
        if n_nan_grads:
            print(f"[GRAD-DIAG] step={step} zeroed {n_nan_grads} NaN/inf grad tensors", flush=True)

        grad_norm_pre = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

        optimizer.step()

        # NaN guard: if the optimizer step corrupted weights, revert and skip.
        if any(torch.isnan(p.data).any() or torch.isinf(p.data).any() for p in trainable):
            print(f"[NaN ALARM] step={step} — NaN/inf in LoRA weights after optimizer step", flush=True)
            if last_good_ckpt is not None:
                model.load_adapter(last_good_ckpt, adapter_name="default")
                print(f"[NaN ALARM] reverted to {last_good_ckpt}", flush=True)
            else:
                for p in trainable:
                    torch.nan_to_num_(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                print("[NaN ALARM] no checkpoint to revert to — zeroed NaN params", flush=True)
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()  # reset Adam m1/m2 buffers — they also contain NaN
            continue

        # --- advance buffer: sample one of G candidates per game ---
        n_solved_in_batch = 0
        for b, idx in enumerate(idxs):
            # pick a random one of the G samples (matches model's stochastic play)
            pick = rng.randrange(G)
            k = b * G + pick
            guess = guesses_list[k]
            step_info = buffer.advance(idx, guess)
            if step_info.get("solved"):
                n_solved_in_batch += 1

        # --- log completed game trajectories ---
        while buffer.completed_trajectories:
            traj = buffer.completed_trajectories.popleft()
            outcome = "SOLVED" if traj["solved"] else f"failed({traj['n_turns']} turns)"
            turns_str = " → ".join(
                f"{h['guess']}(s={h.get('semantic', 0)},e={h.get('exact', 0)},r={h.get('reward', 0):+.1f})"
                for h in traj["history"]
            )
            print(
                f"[TRAJ step={step}] {traj['title']!r} [{outcome}]\n"
                f"  {turns_str}",
                flush=True,
            )
            log_handle.write(json.dumps({"step": step, "trajectory": traj}, ensure_ascii=False) + "\n")
            log_handle.flush()

        # --- logging ---
        mean_r = float(rewards.mean().item())
        max_r = float(rewards.max().item())
        std_r = float(rewards.std().item())
        kl_val = float(kl_loss.item())
        # Policy entropy over completion tokens — computed on the LAST chunk only
        # (cheap proxy; full-batch entropy would need re-running the forward).
        with torch.no_grad():
            if last_chunk_logits_for_entropy is not None and last_chunk_mask_for_entropy is not None:
                policy_probs = torch.softmax(last_chunk_logits_for_entropy.float(), dim=-1)
                log_probs_for_entropy = torch.log(policy_probs.clamp(min=1e-10))
                token_entropy = -(policy_probs * log_probs_for_entropy).sum(dim=-1)
                ent_denom = last_chunk_mask_for_entropy.sum().clamp(min=1.0)
                entropy_val = float((token_entropy * last_chunk_mask_for_entropy).sum() / ent_denom)
            else:
                entropy_val = float("nan")
        seen += 1
        running_reward = (1 - EMA) * running_reward + EMA * mean_r if seen > 1 else mean_r
        running_kl = (1 - EMA) * running_kl + EMA * kl_val if seen > 1 else kl_val
        running_solve = (1 - EMA) * running_solve + EMA * float(any(solved_list)) if seen > 1 else float(any(solved_list))

        log_row = {
            "step": step,
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl": kl_val,
            "entropy": entropy_val,
            "reward_mean": mean_r,
            "reward_max": max_r,
            "reward_std": std_r,
            "comp_len_mean": float(completion_mask.sum(dim=1).float().mean().item()),
            "advantage_std": float(advantages.std().item()),
            "n_solved_in_batch": n_solved_in_batch,
            "games_completed": buffer.games_completed,
            "games_solved": buffer.games_solved,
            "pool_solve_rate": (
                buffer.games_solved / max(1, buffer.games_completed)
            ),
            "step_time": time.time() - t_start,
        }
        log_handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        log_handle.flush()

        if step % args.logging_steps == 0:
            elapsed = time.time() - t_start
            eta = elapsed / step * (args.max_steps - step)
            print(
                f"[step {step:5d}/{args.max_steps}] "
                f"reward={mean_r:+.2f} (ema {running_reward:+.2f}, max {max_r:+.1f}) "
                f"kl={kl_val:.3f} entropy={entropy_val:.2f} loss={float(loss.item()):+.4f} "
                f"len={float(completion_mask.sum(dim=1).float().mean().item()):.1f} "
                f"solved={buffer.games_solved}/{buffer.games_completed} "
                f"pool_solve={log_row['pool_solve_rate']*100:.1f}% "
                f"eta={eta/60:.1f}min",
                flush=True,
            )

        # --- KL alarm: catch catastrophic drift before it locks in ---
        # A sudden KL spike almost always precedes degenerate output (e.g. obscure
        # token exploit, gibberish collapse). On alarm: roll back to last good
        # checkpoint and cut learning rate. Better to lose a few steps than lose
        # the whole run to a locked-in degenerate regime.
        if kl_val > args.kl_alarm_threshold:
            print(
                f"\n[KL ALARM] step={step} kl={kl_val:.1f} > threshold={args.kl_alarm_threshold}",
                flush=True,
            )
            if last_good_ckpt is not None:
                print(f"  rolling back to {last_good_ckpt}", flush=True)
                from peft import PeftModel as _PeftModel
                # Reload LoRA weights in-place
                model.load_adapter(last_good_ckpt, adapter_name="default")
                old_lr = optimizer.param_groups[0]["lr"]
                new_lr = old_lr * args.kl_lr_cut_factor
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                print(f"  lr cut: {old_lr:.2e} → {new_lr:.2e}", flush=True)
            else:
                print("  no checkpoint yet to roll back to — cutting LR only", flush=True)
                old_lr = optimizer.param_groups[0]["lr"]
                new_lr = old_lr * args.kl_lr_cut_factor
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                print(f"  lr cut: {old_lr:.2e} → {new_lr:.2e}", flush=True)
            log_row["kl_alarm"] = True
            log_handle.write(json.dumps({"step": step, "event": "kl_alarm", "kl": kl_val, "new_lr": new_lr}) + "\n")
            log_handle.flush()

        # --- periodic save ---
        if args.save_steps and step % args.save_steps == 0:
            ckpt_dir = out_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            last_good_ckpt = str(ckpt_dir)
            print(f"  [save] {ckpt_dir}", flush=True)

        # --- periodic eval ---
        if (
            args.eval_every_n_steps
            and step % args.eval_every_n_steps == 0
            and args.eval_pages_path
        ):
            _run_periodic_eval(
                model=model,
                tokenizer=tokenizer,
                args=args,
                similarity_model=similarity_model,
                step=step,
                out_dir=out_dir,
                log_handle=log_handle,
            )

        # release fragments periodically
        if step % 10 == 0 and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    # final save
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log_handle.close()
    print(f"[traj-GRPO] done. final model → {out_dir}", flush=True)

    if args.plot_path:
        _plot(log_path, args.plot_path)


def _run_periodic_eval(
    *,
    model,
    tokenizer,
    args: TrajectoryGRPOArgs,
    similarity_model,
    step: int,
    out_dir: Path,
    log_handle,
) -> None:
    import gc as _gc
    import torch as _torch
    from .llm_policy import _run_eval_games

    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    was_training = model.training
    model.eval()
    eval_log = out_dir / "eval_logs" / f"step_{step}.jsonl"
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        device = str(next(model.parameters()).device)
        ctx = getattr(_torch, "inference_mode", _torch.no_grad)
        with ctx():
            result = _run_eval_games(
                model=model,
                tokenizer=tokenizer,
                device=device,
                pages_path=str(args.eval_pages_path),
                similarity_model=similarity_model,
                sample_size=args.eval_pages,
                max_steps=args.eval_max_game_steps,
                seed=args.seed,
                output_path=str(eval_log),
                chat_format=args.eval_chat_format,
                generation_batch_size=args.eval_batch_size,
                eval_num_generations=args.eval_num_generations,
                constrained=True,
            )
        log_handle.write(json.dumps({"step": step, "eval": result}, ensure_ascii=False) + "\n")
        log_handle.flush()
        print(
            f"[step {step:5d}] EVAL solve_rate={result['solve_rate']:.2%} "
            f"mean_steps={result['mean_steps']:.1f}",
            flush=True,
        )
    finally:
        if was_training:
            model.train()
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()


def _plot(log_path: Path, plot_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps, rewards, kls, solves = [], [], [], []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "reward_mean" not in row:
                continue
            steps.append(row["step"])
            rewards.append(row["reward_mean"])
            kls.append(row["kl"])
            solves.append(row["pool_solve_rate"])
    if not steps:
        return
    fig, axL = plt.subplots(figsize=(10, 5))
    axL.plot(steps, rewards, color="#2563eb", label="reward (mean)")
    axL.set_xlabel("step"); axL.set_ylabel("reward"); axL.grid(alpha=0.25)
    axR = axL.twinx()
    axR.plot(steps, kls, color="#9333ea", label="kl", alpha=0.6)
    axR.plot(steps, [s * 100 for s in solves], color="#16a34a", label="pool_solve % (live)")
    axR.set_ylabel("kl / solve%")
    lines = axL.get_lines() + axR.get_lines()
    axL.legend(lines, [l.get_label() for l in lines], loc="best")
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
