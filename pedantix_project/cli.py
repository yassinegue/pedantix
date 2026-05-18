from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

from .corpus import audit_corpus, filter_corpus, sample_pages
from .dataset import (
    fetch_popular_titles,
    fetch_wikipedia_summaries,
    fetch_wikipedia_summaries_batched,
    ingest_pages_articles_dump,
    load_pages,
    save_pages,
)
from .claude_agent import generate_claude_sft_data, run_claude_eval
from .llm_policy import (
    DEFAULT_LLM_MODEL,
    build_llm_curriculum,
    download_hf_model,
    evaluate_llm_policy,
    hf_token_available,
    train_llm_grpo,
    train_llm_sft,
)
from .model import TinyPedantixModel, train_tiny_model
from .rl import GRPOLogEntry, RLPolicy, solve_with_policy, train_grpo_policy, train_rl_policy
from .simulator import PedantixGame
from .solver import solve_page
from .vocab_action_policy import evaluate_vocab_action_policy, train_vocab_action_grpo


def cmd_fetch(args: argparse.Namespace) -> None:
    pages = fetch_wikipedia_summaries(args.titles, language=args.language)
    save_pages(pages, args.output)
    print(f"saved {len(pages)} pages to {args.output}")


def cmd_fetch_popular(args: argparse.Namespace) -> None:
    start = date.fromisoformat(args.start)
    titles = fetch_popular_titles(
        language=args.language,
        start=start,
        days=args.days,
        per_day=args.per_day,
    )
    pages = fetch_wikipedia_summaries_batched(
        titles,
        language=args.language,
        limit=args.limit,
        batch_size=args.batch_size,
        min_intro_chars=args.min_intro_chars,
        sleep_seconds=args.sleep_seconds,
        checkpoint_path=args.output,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_pages(pages, args.output)
    print(f"saved {len(pages)} popular pages to {args.output} from {len(titles)} candidate titles")


def cmd_ingest_dump(args: argparse.Namespace) -> None:
    count = ingest_pages_articles_dump(
        args.dump,
        args.output,
        limit=args.limit,
        min_intro_chars=args.min_intro_chars,
    )
    print(f"saved {count} pages to {args.output}")


def cmd_audit_corpus(args: argparse.Namespace) -> None:
    report = audit_corpus(args.pages, min_words=args.min_words, min_chars=args.min_chars, shortest=args.shortest)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={report['rows']}")
    print(f"chars_quantiles={report['chars_quantiles']}")
    print(f"words_quantiles={report['words_quantiles']}")
    print(f"issue_counts={report['issue_counts']}")
    print(f"rejected_if_filtered={report['rejected_if_filtered']}")
    print(f"top_title_prefixes={report['top_title_prefixes'][:10]}")
    print("shortest:")
    for chars, words, title, intro, issues in report["shortest"][: args.shortest]:
        issue_text = ",".join(issues) if issues else "-"
        print(f"- {chars} chars / {words} words / {issue_text} / {title}: {intro[:180]}")


def cmd_filter_corpus(args: argparse.Namespace) -> None:
    total, kept, issue_counts = filter_corpus(
        args.pages,
        args.output,
        min_words=args.min_words,
        min_chars=args.min_chars,
    )
    print(f"filtered {args.pages} -> {args.output}")
    print(f"total={total} kept={kept} rejected={total - kept}")
    print(f"issue_counts={dict(issue_counts.most_common())}")


def cmd_train(args: argparse.Namespace) -> None:
    pages = load_pages(args.pages)
    model = train_tiny_model(pages, max_vocab=args.max_vocab, max_neighbors=args.max_neighbors)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"trained tiny model on {len(pages)} pages, vocab={len(model.vocabulary)} -> {args.output}")


def cmd_play(args: argparse.Namespace) -> None:
    pages = load_pages(args.pages)
    model = TinyPedantixModel.load(args.model)
    page = random.choice(pages) if args.title is None else next(p for p in pages if p.title == args.title)
    game = PedantixGame(page, similarity_model=model)
    print(game.masked_text())
    while not game.solved:
        guess = input("mot> ").strip()
        result = game.guess(guess)
        print(f"exact={len(result.exact)} proche={len(result.semantic)} solved={result.solved}")
        print(game.masked_text(reveal_semantic=result.semantic))
    print(f"trouvé: {page.title}")


def cmd_solve(args: argparse.Namespace) -> None:
    pages = load_pages(args.pages)
    model = TinyPedantixModel.load(args.model)
    target = random.choice(pages) if args.title is None else next(p for p in pages if p.title == args.title)
    steps = solve_page(target, pages, model, max_steps=args.max_steps)
    for idx, step in enumerate(steps, 1):
        print(
            f"{idx:03d} {step.guess:<18} exact={step.exact_hits:<2} "
            f"proche={step.semantic_hits:<3} candidats={step.candidates:<4} solved={step.solved}"
        )
    print(f"target={target.title} solved={bool(steps and steps[-1].solved)} tries={len(steps)}")


def cmd_rl_train(args: argparse.Namespace) -> None:
    pages = load_pages(args.pages)
    model = TinyPedantixModel.load(args.model)
    policy = train_rl_policy(
        pages,
        model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        action_size=args.action_size,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    print(
        f"trained RL policy on {len(pages)} pages, states={len(policy.q)}, "
        f"actions={len(policy.vocabulary)} -> {args.output}"
    )


def cmd_grpo_train(args: argparse.Namespace) -> None:
    pages = sample_pages(args.pages, sample_size=args.sample_pages, seed=args.seed)
    model = TinyPedantixModel.load(args.model)
    policy, logs = train_grpo_policy(
        pages,
        model,
        updates=args.updates,
        group_size=args.group_size,
        max_steps=args.max_steps,
        action_size=args.action_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_k=args.top_k,
        gradient_top_k=args.gradient_top_k,
        max_policy_states=args.max_policy_states,
        max_actions_per_state=args.max_actions_per_state,
        max_state_items=args.max_state_items,
        curriculum_max_title_words=args.curriculum_max_title_words,
        seed=args.seed,
        log_every=args.log_every,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    if args.log:
        _write_grpo_log(logs, args.log)
    if args.plot:
        _plot_grpo_reward(logs, args.plot)
    last = logs[-1] if logs else GRPOLogEntry(update=0, mean_reward=0.0, solve_rate=0.0, mean_steps=0.0)
    print(
        f"trained GRPO policy on sample={len(pages)} pages, updates={args.updates}, "
        f"states={len(policy.q)}, actions={len(policy.vocabulary)} -> {args.output}"
    )
    print(
        f"last mean_reward={last.mean_reward} solve_rate={last.solve_rate} "
        f"mean_steps={last.mean_steps}"
    )


def _write_grpo_log(logs: list[GRPOLogEntry], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for entry in logs:
            handle.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")


def _plot_grpo_reward(logs: list[GRPOLogEntry], path: str | Path) -> None:
    if not logs:
        return
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    xs = [entry.update for entry in logs]
    rewards = [entry.mean_reward for entry in logs]
    solve_rates = [entry.solve_rate for entry in logs]
    fig, ax_reward = plt.subplots(figsize=(9, 4.8))
    ax_reward.plot(xs, rewards, color="#2563eb", linewidth=2, label="mean reward")
    ax_reward.set_xlabel("GRPO update")
    ax_reward.set_ylabel("mean reward")
    ax_reward.grid(alpha=0.25)
    ax_solve = ax_reward.twinx()
    ax_solve.plot(xs, solve_rates, color="#16a34a", linewidth=1.8, label="solve rate")
    ax_solve.set_ylabel("solve rate")
    lines = ax_reward.get_lines() + ax_solve.get_lines()
    ax_reward.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_rl_solve(args: argparse.Namespace) -> None:
    pages = load_pages(args.pages)
    model = TinyPedantixModel.load(args.model)
    policy = RLPolicy.load(args.policy)
    target = random.choice(pages) if args.title is None else next(p for p in pages if p.title == args.title)
    steps = solve_with_policy(target, model, policy, max_steps=args.max_steps)
    for idx, step in enumerate(steps, 1):
        print(
            f"{idx:03d} {step.guess:<18} exact={step.exact_hits:<2} "
            f"proche={step.semantic_hits:<3} titre={step.title_hits:<2} solved={step.solved}"
        )
    print(f"target={target.title} solved={bool(steps and steps[-1].solved)} tries={len(steps)}")


def cmd_llm_prepare(args: argparse.Namespace) -> None:
    model = TinyPedantixModel.load(args.tiny_model)
    count = build_llm_curriculum(
        args.pages,
        args.output,
        model,
        sample_size=args.sample_pages,
        states_per_page=args.states_per_page,
        max_steps=args.max_game_steps,
        max_title_words=args.curriculum_max_title_words,
        action_size=args.action_size,
        chat_format=args.chat_format,
        seed=args.seed,
        trajectory_mode=args.trajectory_mode,
        min_intro_words=args.min_intro_words,
        min_history_len=args.min_history_len,
    )
    print(f"wrote {count} LLM curriculum rows -> {args.output}")


def cmd_llm_download(args: argparse.Namespace) -> None:
    if not hf_token_available():
        print("warning: no HF token env var detected (expected HF_TOKEN or HUGGINGFACE_HUB_TOKEN)")
    path = download_hf_model(args.model, args.output_dir)
    print(f"downloaded {args.model} -> {path}")


def cmd_llm_sft(args: argparse.Namespace) -> None:
    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in {"auto", "latest", "true", "1"}:
        resume = True
    elif isinstance(resume, str) and resume.lower() in {"", "none", "false", "0"}:
        resume = None
    # Auto-resume is best-effort: if nothing has been saved yet, fall back to a
    # fresh run instead of letting the trainer raise.
    if resume is True:
        out = Path(args.output_dir)
        has_ckpt = out.is_dir() and any(p.name.startswith("checkpoint-") for p in out.iterdir())
        if not has_ckpt:
            resume = None
    args.resume_from_checkpoint = resume
    train_llm_sft(
        train_jsonl=args.train,
        model_name=args.model,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        use_cpu=args.cpu,
        seed=args.seed,
        log_path=args.log,
        eval_every_n_steps=args.eval_every,
        eval_pages=args.eval_pages,
        eval_pages_path=args.eval_corpus,
        eval_max_game_steps=args.eval_max_game_steps,
        eval_chat_format=args.eval_chat_format,
        eval_num_generations=args.eval_num_generations,
        eval_batch_size=args.eval_batch_size,
        tiny_model_path=args.tiny_model,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"saved SFT adapter/model -> {args.output_dir}")


def cmd_llm_grpo(args: argparse.Namespace) -> None:
    train_llm_grpo(
        train_jsonl=args.train,
        model_name_or_path=args.model,
        tiny_model_path=args.tiny_model,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        use_cpu=args.cpu,
        seed=args.seed,
        log_path=args.log,
        plot_path=args.plot,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        show_progress=not args.no_progress,
        resume_from_checkpoint=args.resume_from_checkpoint,
        solve_bonus_scale=args.solve_bonus_scale,
        beta=args.beta,
        eval_every_n_steps=args.eval_every,
        eval_pages=args.eval_pages,
        eval_pages_path=args.eval_corpus,
        eval_max_game_steps=args.eval_max_game_steps,
        eval_chat_format=args.eval_chat_format,
        eval_num_generations=args.eval_num_generations,
        eval_batch_size=args.eval_batch_size,
        dagger_every=args.dagger_every,
        dagger_pages=args.dagger_pages,
        dagger_rollout_steps=args.dagger_rollout_steps,
        dagger_microsteps=args.dagger_microsteps,
        dagger_bc_batch_size=args.dagger_bc_batch_size,
        dagger_history_max_steps=args.dagger_history_max_steps,
        dagger_oracle_mode=args.dagger_oracle_mode,
        dagger_oracle_top_k=args.dagger_oracle_top_k,
        dagger_oracle_temperature=args.dagger_oracle_temperature,
        dagger_oracle_min_idf=args.dagger_oracle_min_idf,
    )
    print(f"saved GRPO adapter/model -> {args.output_dir}")


def cmd_claude_eval(args: argparse.Namespace) -> None:
    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("Provide --api-key or set ANTHROPIC_API_KEY env var")
    sim = TinyPedantixModel.load(args.tiny_model)
    summary = run_claude_eval(
        pages_path=args.pages,
        sim_model=sim,
        api_key=api_key,
        model=args.claude_model,
        n_pages=args.n_pages,
        max_steps=args.max_steps,
        out_jsonl=args.output,
        seed=args.seed,
        verbose=True,
    )
    print(json.dumps(summary, indent=2))


def cmd_claude_sft_gen(args: argparse.Namespace) -> None:
    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("Provide --api-key or set ANTHROPIC_API_KEY env var")
    sim = TinyPedantixModel.load(args.tiny_model)
    n = generate_claude_sft_data(
        pages_path=args.pages,
        sim_model=sim,
        api_key=api_key,
        model=args.claude_model,
        n_pages=args.n_pages,
        max_steps=args.max_steps,
        out_jsonl=args.output,
        seed=args.seed,
        verbose=True,
    )
    print(f"wrote {n} training examples to {args.output}")


def cmd_llm_eval(args: argparse.Namespace) -> None:
    model = TinyPedantixModel.load(args.tiny_model)
    result = evaluate_llm_policy(
        pages_path=args.pages,
        model_path=args.model,
        similarity_model=model,
        sample_size=args.sample_pages,
        max_steps=args.max_game_steps,
        seed=args.seed,
        output_path=args.output,
        chat_format=args.chat_format,
        generation_batch_size=args.eval_batch_size,
        eval_num_generations=args.eval_num_generations,
    )
    print(json.dumps(result, ensure_ascii=False))


def cmd_llm_vocab_grpo(args: argparse.Namespace) -> None:
    train_vocab_action_grpo(
        train_jsonl=args.train,
        pages_path=args.pages,
        model_name_or_path=args.model,
        tiny_model_path=args.tiny_model,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        action_size=args.action_size,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        temperature=args.temperature,
        bc_coef=args.bc_coef,
        seed=args.seed,
        log_path=args.log,
        freeze_backbone=args.freeze_backbone,
        entropy_coef=args.entropy_coef,
        bc_warmup_steps=args.bc_warmup_steps,
        freeze_lora_steps=args.freeze_lora_steps,
        plot_path=args.plot,
        plot_every=args.plot_every,
        action_head_weight_decay=args.action_head_weight_decay,
        min_entropy=args.min_entropy,
        min_entropy_coef=args.min_entropy_coef,
        head_hidden_size=args.head_hidden_size,
        kl_ref_coef=args.kl_ref_coef,
        dynamic_sampling=args.dynamic_sampling,
        rollout_steps=args.rollout_steps,
        dagger_every=args.dagger_every,
        dagger_bc_steps=args.dagger_bc_steps,
        dagger_pages=args.dagger_pages,
        title_bias_init=args.title_bias_init,
        title_mask_cos_threshold=args.title_mask_cos_threshold,
        use_lm_head=args.use_lm_head,
        near_solve_shaping_coef=args.near_solve_shaping_coef,
        dynamic_expand_k=args.dynamic_expand_k,
    )
    print(f"saved vocab-action GRPO adapter/model -> {args.output_dir}")


def cmd_llm_vocab_eval(args: argparse.Namespace) -> None:
    model = TinyPedantixModel.load(args.tiny_model)
    result = evaluate_vocab_action_policy(
        pages_path=args.pages,
        model_path=args.model,
        similarity_model=model,
        sample_size=args.sample_pages,
        max_steps=args.max_game_steps,
        seed=args.seed,
        output_path=args.output,
        batch_size=args.eval_batch_size,
        max_prompt_length=args.max_prompt_length,
        dynamic_expand_k=args.dynamic_expand_k,
    )
    print(json.dumps(result, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pedantix-project")
    sub = parser.add_subparsers(required=True)

    fetch = sub.add_parser("fetch", help="fetch Wikipedia summaries into JSONL")
    fetch.add_argument("titles", nargs="+")
    fetch.add_argument("--language", default="fr")
    fetch.add_argument("--output", default="data/pages.jsonl")
    fetch.set_defaults(func=cmd_fetch)

    fetch_popular = sub.add_parser("fetch-popular", help="fetch popular French Wikipedia pages")
    fetch_popular.add_argument("--language", default="fr")
    fetch_popular.add_argument("--start", default="2025-01-01")
    fetch_popular.add_argument("--days", type=int, default=45)
    fetch_popular.add_argument("--per-day", type=int, default=250)
    fetch_popular.add_argument("--limit", type=int, default=3000)
    fetch_popular.add_argument("--batch-size", type=int, default=10)
    fetch_popular.add_argument("--min-intro-chars", type=int, default=180)
    fetch_popular.add_argument("--sleep-seconds", type=float, default=1.0)
    fetch_popular.add_argument("--output", default="data/popular_pages.jsonl")
    fetch_popular.set_defaults(func=cmd_fetch_popular)

    ingest_dump = sub.add_parser("ingest-dump", help="stream a Wikimedia pages-articles .bz2 dump into JSONL")
    ingest_dump.add_argument("--dump", required=True)
    ingest_dump.add_argument("--output", default="data/bulk_pages.jsonl")
    ingest_dump.add_argument("--limit", type=int, default=100000)
    ingest_dump.add_argument("--min-intro-chars", type=int, default=220)
    ingest_dump.set_defaults(func=cmd_ingest_dump)

    audit = sub.add_parser("audit-corpus", help="inspect corpus quality and shortest pages")
    audit.add_argument("--pages", default="data/bulk_pages.jsonl")
    audit.add_argument("--min-words", type=int, default=40)
    audit.add_argument("--min-chars", type=int, default=220)
    audit.add_argument("--shortest", type=int, default=40)
    audit.add_argument("--output")
    audit.set_defaults(func=cmd_audit_corpus)

    filter_cmd = sub.add_parser("filter-corpus", help="remove low-quality pages from a JSONL corpus")
    filter_cmd.add_argument("--pages", default="data/bulk_pages.jsonl")
    filter_cmd.add_argument("--output", default="data/clean_pages.jsonl")
    filter_cmd.add_argument("--min-words", type=int, default=40)
    filter_cmd.add_argument("--min-chars", type=int, default=220)
    filter_cmd.set_defaults(func=cmd_filter_corpus)

    train = sub.add_parser("train", help="train tiny local model")
    train.add_argument("--pages", default="data/sample_pages.jsonl")
    train.add_argument("--output", default="models/tiny_model.json")
    train.add_argument("--max-vocab", type=int, default=5000)
    train.add_argument("--max-neighbors", type=int, default=25)
    train.set_defaults(func=cmd_train)

    play = sub.add_parser("play", help="play a local game")
    play.add_argument("--pages", default="data/sample_pages.jsonl")
    play.add_argument("--model", default="models/tiny_model.json")
    play.add_argument("--title")
    play.set_defaults(func=cmd_play)

    solve = sub.add_parser("solve", help="let the tiny solver play")
    solve.add_argument("--pages", default="data/sample_pages.jsonl")
    solve.add_argument("--model", default="models/tiny_model.json")
    solve.add_argument("--title")
    solve.add_argument("--max-steps", type=int, default=80)
    solve.set_defaults(func=cmd_solve)

    rl_train = sub.add_parser("rl-train", help="train candidate-free RL policy")
    rl_train.add_argument("--pages", default="data/popular_pages.jsonl")
    rl_train.add_argument("--model", default="models/tiny_model.json")
    rl_train.add_argument("--output", default="models/rl_policy.json")
    rl_train.add_argument("--episodes", type=int, default=6000)
    rl_train.add_argument("--max-steps", type=int, default=80)
    rl_train.add_argument("--action-size", type=int, default=900)
    rl_train.add_argument("--seed", type=int, default=7)
    rl_train.set_defaults(func=cmd_rl_train)

    grpo_train = sub.add_parser("grpo-train", help="train candidate-free GRPO-style tabular policy")
    grpo_train.add_argument("--pages", default="data/clean_pages.jsonl")
    grpo_train.add_argument("--model", default="models/tiny_model.json")
    grpo_train.add_argument("--output", default="models/grpo_policy.json")
    grpo_train.add_argument("--log", default="models/grpo_training.jsonl")
    grpo_train.add_argument("--plot", default="models/grpo_reward.png")
    grpo_train.add_argument("--sample-pages", type=int, default=20000)
    grpo_train.add_argument("--updates", type=int, default=1000)
    grpo_train.add_argument("--group-size", type=int, default=4)
    grpo_train.add_argument("--max-steps", type=int, default=50)
    grpo_train.add_argument("--action-size", type=int, default=900)
    grpo_train.add_argument("--learning-rate", type=float, default=0.05)
    grpo_train.add_argument("--temperature", type=float, default=0.9)
    grpo_train.add_argument("--top-k", type=int, default=180)
    grpo_train.add_argument("--gradient-top-k", type=int, default=12)
    grpo_train.add_argument("--max-policy-states", type=int, default=50000)
    grpo_train.add_argument("--max-actions-per-state", type=int, default=16)
    grpo_train.add_argument("--max-state-items", type=int, default=5)
    grpo_train.add_argument("--curriculum-max-title-words", type=int)
    grpo_train.add_argument("--log-every", type=int, default=20)
    grpo_train.add_argument("--seed", type=int, default=7)
    grpo_train.set_defaults(func=cmd_grpo_train)

    rl_solve = sub.add_parser("rl-solve", help="solve with RL policy without candidate filtering")
    rl_solve.add_argument("--pages", default="data/popular_pages.jsonl")
    rl_solve.add_argument("--model", default="models/tiny_model.json")
    rl_solve.add_argument("--policy", default="models/rl_policy.json")
    rl_solve.add_argument("--title")
    rl_solve.add_argument("--max-steps", type=int, default=80)
    rl_solve.set_defaults(func=cmd_rl_solve)

    llm_prepare = sub.add_parser("llm-prepare", help="build prompt/state rows for LLM SFT and GRPO")
    llm_prepare.add_argument("--pages", default="data/clean_pages.jsonl")
    llm_prepare.add_argument("--tiny-model", default="models/tiny_model.json")
    llm_prepare.add_argument("--output", default="data/llm_curriculum.jsonl")
    llm_prepare.add_argument("--sample-pages", type=int, default=5000)
    llm_prepare.add_argument("--states-per-page", type=int, default=2)
    llm_prepare.add_argument("--max-game-steps", type=int, default=100)
    llm_prepare.add_argument("--curriculum-max-title-words", type=int, default=1)
    llm_prepare.add_argument("--action-size", type=int, default=2000)
    llm_prepare.add_argument("--chat-format", choices=["none", "qwen"], default="qwen")
    llm_prepare.add_argument("--trajectory-mode", choices=["teacher", "oracle"], default="teacher")
    llm_prepare.add_argument("--min-intro-words", type=int, default=0,
                             help="skip pages with fewer than this many intro words (0=no filter)")
    llm_prepare.add_argument("--min-history-len", type=int, default=0,
                             help="run this many oracle steps as warm-up before writing training rows (trains on later game states)")
    llm_prepare.add_argument("--seed", type=int, default=7)
    llm_prepare.set_defaults(func=cmd_llm_prepare)

    llm_download = sub.add_parser("llm-download", help="download/cache the HF base model using the env HF token")
    llm_download.add_argument("--model", default=DEFAULT_LLM_MODEL)
    llm_download.add_argument("--output-dir")
    llm_download.set_defaults(func=cmd_llm_download)

    llm_sft = sub.add_parser("llm-sft", help="LoRA SFT warmup for one-word Pedantix policy")
    llm_sft.add_argument("--train", default="data/llm_curriculum.jsonl")
    llm_sft.add_argument("--model", default=DEFAULT_LLM_MODEL)
    llm_sft.add_argument("--output-dir", default="models/llm_sft")
    llm_sft.add_argument("--max-steps", type=int, default=100)
    llm_sft.add_argument("--batch-size", type=int, default=1)
    llm_sft.add_argument("--gradient-accumulation-steps", type=int, default=8)
    llm_sft.add_argument("--learning-rate", type=float, default=2e-5)
    llm_sft.add_argument("--lora-rank", type=int, default=8)
    llm_sft.add_argument("--cpu", action="store_true")
    llm_sft.add_argument("--seed", type=int, default=7)
    llm_sft.add_argument("--log", default=None,
                         help="path to write trainer log_history JSONL")
    llm_sft.add_argument("--tiny-model", default="models/tiny_model.json",
                         help="TinyPedantixModel path for held-out eval (required if --eval-every>0)")
    llm_sft.add_argument("--eval-every", type=int, default=0,
                         help="run held-out eval every N optimizer steps (0=disabled)")
    llm_sft.add_argument("--eval-pages", type=int, default=50,
                         help="number of fresh pages to evaluate on")
    llm_sft.add_argument("--eval-corpus", default=None,
                         help="path to held-out pages JSONL; required when --eval-every>0")
    llm_sft.add_argument("--eval-max-game-steps", type=int, default=30,
                         help="max guesses per eval game")
    llm_sft.add_argument("--eval-chat-format", choices=["none", "qwen"], default="qwen")
    llm_sft.add_argument("--eval-num-generations", type=int, default=4)
    llm_sft.add_argument("--eval-batch-size", type=int, default=8)
    llm_sft.add_argument("--save-steps", type=int, default=None,
                         help="checkpoint every N optimizer steps; defaults to --eval-every (or end-of-run if disabled)")
    llm_sft.add_argument("--save-total-limit", type=int, default=2,
                         help="max number of checkpoints kept on disk")
    llm_sft.add_argument("--resume-from-checkpoint", default=None,
                         help="path to a checkpoint dir, or pass 'auto' / 'latest' to auto-resume from --output-dir")
    llm_sft.set_defaults(func=cmd_llm_sft)

    llm_grpo = sub.add_parser("llm-grpo", help="LoRA GRPO/RLVR training with simulator rewards")
    llm_grpo.add_argument("--train", default="data/llm_curriculum.jsonl")
    llm_grpo.add_argument("--model", default=DEFAULT_LLM_MODEL)
    llm_grpo.add_argument("--tiny-model", default="models/tiny_model.json")
    llm_grpo.add_argument("--output-dir", default="models/llm_grpo")
    llm_grpo.add_argument("--log", default="models/llm_grpo_training.jsonl")
    llm_grpo.add_argument("--plot", default="models/llm_grpo_reward.png")
    llm_grpo.add_argument("--max-steps", type=int, default=100)
    llm_grpo.add_argument("--batch-size", type=int, default=16)
    llm_grpo.add_argument("--gradient-accumulation-steps", type=int, default=1)
    llm_grpo.add_argument("--num-generations", type=int, default=8)
    llm_grpo.add_argument("--max-completion-length", type=int, default=10)
    llm_grpo.add_argument("--learning-rate", type=float, default=1e-6)
    llm_grpo.add_argument("--lora-rank", type=int, default=32)
    llm_grpo.add_argument("--logging-steps", type=int, default=1)
    llm_grpo.add_argument("--save-steps", type=int)
    llm_grpo.add_argument("--temperature", type=float, default=0.8)
    llm_grpo.add_argument("--top-p", type=float, default=0.9)
    llm_grpo.add_argument("--resume-from-checkpoint")
    llm_grpo.add_argument("--no-progress", action="store_true")
    llm_grpo.add_argument("--cpu", action="store_true")
    llm_grpo.add_argument("--seed", type=int, default=7)
    llm_grpo.add_argument("--solve-bonus-scale", type=float, default=1.0,
                          help="scale on +1000 SOLVED_TITLE_REWARD; v2 plan uses 0.05 to clip to +50")
    llm_grpo.add_argument("--beta", type=float, default=0.02,
                          help="KL coefficient against the reference policy (TRL GRPOConfig.beta)")
    llm_grpo.add_argument("--eval-every", type=int, default=0,
                          help="run held-out eval every N optimizer steps (0=disabled)")
    llm_grpo.add_argument("--eval-pages", type=int, default=50,
                          help="number of fresh pages to evaluate on")
    llm_grpo.add_argument("--eval-corpus", default=None,
                          help="path to held-out pages JSONL; required when --eval-every>0")
    llm_grpo.add_argument("--eval-max-game-steps", type=int, default=30,
                          help="max guesses per eval game")
    llm_grpo.add_argument("--eval-chat-format", choices=["none", "qwen"], default="qwen")
    llm_grpo.add_argument("--eval-num-generations", type=int, default=4)
    llm_grpo.add_argument("--eval-batch-size", type=int, default=8)
    llm_grpo.add_argument("--dagger-every", type=int, default=0,
                          help="run a DAgger cycle every N GRPO steps (0=disabled)")
    llm_grpo.add_argument("--dagger-pages", type=int, default=32,
                          help="number of fresh pages to roll out per DAgger cycle")
    llm_grpo.add_argument("--dagger-rollout-steps", type=int, default=12,
                          help="how many turns to roll out per page during DAgger")
    llm_grpo.add_argument("--dagger-microsteps", type=int, default=16,
                          help="number of BC SGD microsteps per DAgger cycle")
    llm_grpo.add_argument("--dagger-bc-batch-size", type=int, default=4,
                          help="batch size used for each BC microstep")
    llm_grpo.add_argument("--dagger-history-max-steps", type=int, default=30,
                          help="game length cap to pass into make_prompt during DAgger rollouts")
    llm_grpo.add_argument("--dagger-oracle-mode", choices=["strong", "soft"], default="soft",
                          help="strong=argmax over all candidates incl. titles; soft=excludes titles + top-K sampling")
    llm_grpo.add_argument("--dagger-oracle-top-k", type=int, default=8,
                          help="for soft oracle: sample from K best-scored non-title candidates")
    llm_grpo.add_argument("--dagger-oracle-temperature", type=float, default=1.0,
                          help="for soft oracle: softmax temperature over top-K rewards (0 = argmax)")
    llm_grpo.add_argument("--dagger-oracle-min-idf", type=float, default=0.0,
                          help="for soft oracle: drop candidate words with IDF below this floor")
    llm_grpo.set_defaults(func=cmd_llm_grpo)

    llm_eval = sub.add_parser("llm-eval", help="play local Pedantix games with a trained LLM policy")
    llm_eval.add_argument("--pages", default="data/clean_pages.jsonl")
    llm_eval.add_argument("--model", default="models/llm_grpo")
    llm_eval.add_argument("--tiny-model", default="models/tiny_model.json")
    llm_eval.add_argument("--sample-pages", type=int, default=20)
    llm_eval.add_argument("--max-game-steps", type=int, default=100)
    llm_eval.add_argument("--output", default="models/llm_eval.jsonl")
    llm_eval.add_argument("--chat-format", choices=["none", "qwen"], default="qwen")
    llm_eval.add_argument("--eval-batch-size", type=int, default=16)
    llm_eval.add_argument("--eval-num-generations", type=int, default=8)
    llm_eval.add_argument("--seed", type=int, default=7)
    llm_eval.set_defaults(func=cmd_llm_eval)

    llm_vocab_grpo = sub.add_parser("llm-vocab-grpo", help="LoRA GRPO over a fixed French word action vocabulary")
    llm_vocab_grpo.add_argument("--train", default="data/llm_curriculum.jsonl")
    llm_vocab_grpo.add_argument("--pages", default="data/clean_pages.jsonl")
    llm_vocab_grpo.add_argument("--model", default=DEFAULT_LLM_MODEL)
    llm_vocab_grpo.add_argument("--tiny-model", default="models/tiny_model.json")
    llm_vocab_grpo.add_argument("--output-dir", default="models/llm_vocab_grpo")
    llm_vocab_grpo.add_argument("--log", default="models/llm_vocab_grpo_training.jsonl")
    llm_vocab_grpo.add_argument("--plot", default=None,
                                help="path to save reward/entropy plot (updated every --plot-every steps)")
    llm_vocab_grpo.add_argument("--plot-every", type=int, default=50,
                                help="save plot every N steps")
    llm_vocab_grpo.add_argument("--max-steps", type=int, default=100)
    llm_vocab_grpo.add_argument("--batch-size", type=int, default=8)
    llm_vocab_grpo.add_argument("--num-generations", type=int, default=8)
    llm_vocab_grpo.add_argument("--action-size", type=int, default=5000)
    llm_vocab_grpo.add_argument("--max-prompt-length", type=int, default=384)
    llm_vocab_grpo.add_argument("--learning-rate", type=float, default=2e-5)
    llm_vocab_grpo.add_argument("--lora-rank", type=int, default=16)
    llm_vocab_grpo.add_argument("--temperature", type=float, default=1.0)
    llm_vocab_grpo.add_argument("--bc-coef", type=float, default=0.0)
    llm_vocab_grpo.add_argument("--freeze-backbone", action="store_true",
                                help="freeze Qwen weights, train only the action head (much faster)")
    llm_vocab_grpo.add_argument("--entropy-coef", type=float, default=0.0,
                                help="entropy bonus coefficient to prevent distribution collapse")
    llm_vocab_grpo.add_argument("--bc-warmup-steps", type=int, default=0,
                                help="steps of pure BC (no GRPO) before enabling RL signal")
    llm_vocab_grpo.add_argument("--freeze-lora-steps", type=int, default=0,
                                help="steps to freeze LoRA (phase 1: action head only), then unfreeze for joint training")
    llm_vocab_grpo.add_argument("--action-head-weight-decay", type=float, default=0.0,
                                help="L2 weight decay on action head to prevent logit explosion and entropy collapse")
    llm_vocab_grpo.add_argument("--min-entropy", type=float, default=0.0,
                                help="minimum entropy target: penalizes when entropy drops below this value")
    llm_vocab_grpo.add_argument("--min-entropy-coef", type=float, default=1.0,
                                help="coefficient for the minimum entropy penalty")
    llm_vocab_grpo.add_argument("--head-hidden-size", type=int, default=0,
                                help="if >0, use a 2-layer MLP action head with this hidden dim instead of a single Linear")
    llm_vocab_grpo.add_argument("--kl-ref-coef", type=float, default=0.0,
                                help="coefficient for reverse KL(ref||current): provides non-zero gradient to all actions even at collapse")
    llm_vocab_grpo.add_argument("--dynamic-sampling", action="store_true", default=False,
                                help="skip GRPO update for groups where all generations share the same reward (zero variance)")
    llm_vocab_grpo.add_argument("--rollout-steps", type=int, default=1,
                                help="number of consecutive oracle states per GRPO step (multi-step rollout)")
    llm_vocab_grpo.add_argument("--dagger-every", type=int, default=0,
                                help="run DAgger BC refresh every N GRPO steps (0 = disabled)")
    llm_vocab_grpo.add_argument("--dagger-bc-steps", type=int, default=10,
                                help="number of BC gradient steps per DAgger refresh")
    llm_vocab_grpo.add_argument("--dagger-pages", type=int, default=4,
                                help="number of pages to simulate per DAgger refresh")
    llm_vocab_grpo.add_argument("--title-bias-init", type=float, default=0.0,
                                help="add constant to action-head bias for title-only word indices at init")
    llm_vocab_grpo.add_argument("--title-mask-cos-threshold", type=float, default=0.0,
                                help="mask title words with max page-similarity below threshold (0=disabled)")
    llm_vocab_grpo.add_argument("--near-solve-shaping-coef", type=float, default=0.0,
                                help="bonus = coef * max_sim(guess, unrevealed_title_words); creates dense gradient toward title words (0 = disabled)")
    llm_vocab_grpo.add_argument("--use-lm-head", action="store_true", default=False,
                                help="use LM un-embedding matrix as fixed action head; LoRA trains hidden states toward word embedding directions")
    llm_vocab_grpo.add_argument("--dynamic-expand-k", type=int, default=0,
                                help="if >0, per-episode expand action space with title words + their top-K TinyModel neighbors")
    llm_vocab_grpo.add_argument("--seed", type=int, default=7)
    llm_vocab_grpo.set_defaults(func=cmd_llm_vocab_grpo)

    llm_vocab_eval = sub.add_parser("llm-vocab-eval", help="play Pedantix with a fixed-vocabulary LLM action policy")
    llm_vocab_eval.add_argument("--pages", default="data/clean_pages.jsonl")
    llm_vocab_eval.add_argument("--model", default="models/llm_vocab_grpo")
    llm_vocab_eval.add_argument("--tiny-model", default="models/tiny_model.json")
    llm_vocab_eval.add_argument("--sample-pages", type=int, default=20)
    llm_vocab_eval.add_argument("--max-game-steps", type=int, default=100)
    llm_vocab_eval.add_argument("--output", default="models/llm_vocab_eval.jsonl")
    llm_vocab_eval.add_argument("--eval-batch-size", type=int, default=16)
    llm_vocab_eval.add_argument("--max-prompt-length", type=int, default=384)
    llm_vocab_eval.add_argument("--seed", type=int, default=7)
    llm_vocab_eval.add_argument("--dynamic-expand-k", type=int, default=0,
                                help="test-time expansion: after each reveal, add k TinyModel neighbors to action set (0=disabled)")
    llm_vocab_eval.set_defaults(func=cmd_llm_vocab_eval)

    claude_eval = sub.add_parser("claude-eval", help="baseline: play Pedantix games with Claude API and measure solve rate")
    claude_eval.add_argument("--pages", default="data/clean_pages.jsonl")
    claude_eval.add_argument("--tiny-model", default="models/tiny_model.json")
    claude_eval.add_argument("--n-pages", type=int, default=100)
    claude_eval.add_argument("--max-steps", type=int, default=30)
    claude_eval.add_argument("--output", default="models/claude_baseline.jsonl")
    claude_eval.add_argument("--claude-model", default="claude-haiku-4-5-20251001")
    claude_eval.add_argument("--api-key", default="", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    claude_eval.add_argument("--seed", type=int, default=42)
    claude_eval.set_defaults(func=cmd_claude_eval)

    claude_sft_gen = sub.add_parser("claude-sft-gen", help="generate SFT training data from Claude gameplay")
    claude_sft_gen.add_argument("--pages", default="data/clean_pages.jsonl")
    claude_sft_gen.add_argument("--tiny-model", default="models/tiny_model.json")
    claude_sft_gen.add_argument("--n-pages", type=int, default=1000)
    claude_sft_gen.add_argument("--max-steps", type=int, default=30)
    claude_sft_gen.add_argument("--output", default="data/claude_sft.jsonl")
    claude_sft_gen.add_argument("--claude-model", default="claude-haiku-4-5-20251001")
    claude_sft_gen.add_argument("--api-key", default="", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    claude_sft_gen.add_argument("--seed", type=int, default=42)
    claude_sft_gen.set_defaults(func=cmd_claude_sft_gen)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
