from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import WikiPage
from .model import TinyPedantixModel
from .simulator import GuessResult, PedantixGame
from .text import canonical_word, content_words


THEME_SEEDS = [
    "personne",
    "ville",
    "pays",
    "france",
    "monde",
    "histoire",
    "politique",
    "guerre",
    "science",
    "art",
    "sport",
    "musique",
    "cinema",
    "europe",
    "etat",
    "groupe",
    "langue",
    "culture",
    "naissance",
    "mort",
    "siecle",
    "annee",
    "club",
    "universite",
    "religion",
]

STOPWORDS = {
    "afin",
    "ainsi",
    "alors",
    "apres",
    "aucun",
    "aussi",
    "autre",
    "aux",
    "avec",
    "avoir",
    "car",
    "ceci",
    "cela",
    "ces",
    "cet",
    "cette",
    "chez",
    "comme",
    "dans",
    "dan",
    "des",
    "deux",
    "donc",
    "dont",
    "elle",
    "elles",
    "entre",
    "est",
    "etre",
    "eux",
    "fait",
    "font",
    "hors",
    "ils",
    "les",
    "leur",
    "leurs",
    "lors",
    "mais",
    "meme",
    "mes",
    "ont",
    "par",
    "pas",
    "peu",
    "plus",
    "plu",
    "pour",
    "puis",
    "quand",
    "que",
    "quel",
    "qui",
    "quoi",
    "sans",
    "ses",
    "son",
    "sont",
    "sous",
    "sur",
    "tres",
    "une",
    "vers",
    "via",
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
    "pendant",
}


@dataclass(frozen=True)
class RLStep:
    guess: str
    exact_hits: int
    semantic_hits: int
    title_hits: int
    solved: bool


@dataclass
class RLPolicy:
    q: dict[str, dict[str, float]]
    priors: dict[str, float]
    vocabulary: list[str]
    max_state_items: int = 8

    def save(self, path: str | Path) -> None:
        payload = {
            "q": self.q,
            "priors": self.priors,
            "vocabulary": self.vocabulary,
            "max_state_items": self.max_state_items,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RLPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            q={str(k): {str(a): float(v) for a, v in actions.items()} for k, actions in payload["q"].items()},
            priors={str(k): float(v) for k, v in payload["priors"].items()},
            vocabulary=list(payload["vocabulary"]),
            max_state_items=int(payload.get("max_state_items", 8)),
        )

    def choose(self, history: list[RLStep], guessed: set[str]) -> str | None:
        state = state_key(history, self.max_state_items)
        candidates = self.q.get(state, {})
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        for word, _ in ranked:
            if word not in guessed:
                return word
        for word, _ in sorted(self.priors.items(), key=lambda item: item[1], reverse=True):
            if word not in guessed:
                return word
        for word in self.vocabulary:
            if word not in guessed:
                return word
        return None


@dataclass(frozen=True)
class GRPOLogEntry:
    update: int
    mean_reward: float
    solve_rate: float
    mean_steps: float


def train_rl_policy(
    pages: list[WikiPage],
    similarity_model: TinyPedantixModel,
    *,
    episodes: int = 6000,
    max_steps: int = 80,
    action_size: int = 900,
    alpha: float = 0.25,
    gamma: float = 0.88,
    epsilon_start: float = 0.85,
    epsilon_end: float = 0.08,
    seed: int = 7,
) -> RLPolicy:
    rng = random.Random(seed)
    vocabulary, priors = build_action_space(pages, max_words=action_size)
    q: dict[str, dict[str, float]] = defaultdict(dict)

    if not pages or not vocabulary:
        return RLPolicy(q={}, priors=priors, vocabulary=vocabulary)

    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / max(1, episodes / 4))
        page = rng.choice(pages)
        game = PedantixGame(page, similarity_model=similarity_model)
        history: list[RLStep] = []
        guessed: set[str] = set()

        for _ in range(max_steps):
            state = state_key(history)
            action = _choose_training_action(q, priors, vocabulary, state, guessed, epsilon, rng)
            if action is None:
                break
            before_title_hits = _revealed_title_hits(game)
            result = game.guess(action)
            guessed.add(action)
            title_hits = _revealed_title_hits(game) - before_title_hits
            step = _step_from_result(action, result, title_hits)
            next_history = history + [step]
            next_state = state_key(next_history)
            reward = _reward(step)
            old = q[state].get(action, priors.get(action, 0.0))
            future = max(q.get(next_state, {}).values(), default=0.0)
            q[state][action] = old + alpha * (reward + gamma * future - old)
            history = next_history
            if result.solved:
                break

    compact_q = {
        state: dict(sorted(actions.items(), key=lambda item: item[1], reverse=True)[:40])
        for state, actions in q.items()
    }
    return RLPolicy(q=compact_q, priors=priors, vocabulary=vocabulary)


def train_grpo_policy(
    pages: list[WikiPage],
    similarity_model: TinyPedantixModel,
    *,
    updates: int = 1000,
    group_size: int = 4,
    max_steps: int = 50,
    action_size: int = 900,
    learning_rate: float = 0.05,
    temperature: float = 0.9,
    top_k: int = 180,
    gradient_top_k: int = 12,
    max_policy_states: int = 50000,
    max_actions_per_state: int = 16,
    max_state_items: int = 5,
    curriculum_max_title_words: int | None = None,
    seed: int = 7,
    log_every: int = 20,
) -> tuple[RLPolicy, list[GRPOLogEntry]]:
    """Train a small tabular policy with group-relative policy updates.

    This is intentionally not a page-candidate solver. The policy state only
    contains the compact game feedback history, and the action space is a fixed
    vocabulary of words learned before play. For each target page, several
    rollouts are sampled; their returns are centered within the group, then the
    sampled action logits are nudged by the relative advantage.
    """
    rng = random.Random(seed)
    vocabulary, priors = build_action_space(pages, max_words=action_size)
    logits: dict[str, dict[str, float]] = defaultdict(dict)
    logs: list[GRPOLogEntry] = []

    if not pages or not vocabulary:
        return RLPolicy(q={}, priors=priors, vocabulary=vocabulary), logs
    if curriculum_max_title_words is not None:
        vocab_set = set(vocabulary)
        curriculum_pages = [
            page
            for page in pages
            if page.title_words
            and len(page.title_words) <= curriculum_max_title_words
            and page.title_words <= vocab_set
        ]
        if curriculum_pages:
            pages = curriculum_pages

    recent_rewards: list[float] = []
    recent_steps: list[int] = []
    recent_solved: list[bool] = []

    for update in range(1, updates + 1):
        page = rng.choice(pages)
        rollouts = [
            _sample_rollout(
                logits,
                priors,
                vocabulary,
                page,
                similarity_model,
                max_steps,
                temperature,
                top_k,
                max_state_items,
                rng,
            )
            for _ in range(group_size)
        ]
        rewards = [rollout["reward"] for rollout in rollouts]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / max(1, len(rewards) - 1)
        std = math.sqrt(variance) or 1.0

        for rollout, reward in zip(rollouts, rewards):
            advantage = max(-3.0, min(3.0, (reward - mean_reward) / std))
            for decision in rollout["decisions"]:
                state = decision["state"]
                action = decision["action"]
                actions = decision["actions"]
                probs = decision["probs"]
                state_logits = logits[state]
                alternatives = list(zip(actions[:gradient_top_k], probs[:gradient_top_k]))
                if action not in {alt for alt, _ in alternatives}:
                    alternatives.append((action, 0.0))
                for alt_action, prob in alternatives:
                    state_logits[alt_action] = state_logits.get(alt_action, 0.0) - learning_rate * advantage * prob
                state_logits[action] = state_logits.get(action, 0.0) + learning_rate * advantage
                _clip_state_logits(state_logits, max_actions=max_actions_per_state)

        recent_rewards.extend(rewards)
        recent_steps.extend(int(rollout["steps"]) for rollout in rollouts)
        recent_solved.extend(bool(rollout["solved"]) for rollout in rollouts)
        if update % log_every == 0 or update == updates:
            logs.append(
                GRPOLogEntry(
                    update=update,
                    mean_reward=round(sum(recent_rewards) / len(recent_rewards), 4),
                    solve_rate=round(sum(1 for solved in recent_solved if solved) / len(recent_solved), 4),
                    mean_steps=round(sum(recent_steps) / len(recent_steps), 4),
                )
            )
            recent_rewards.clear()
            recent_steps.clear()
            recent_solved.clear()

    compact_q = _compact_logits(logits, max_states=max_policy_states, max_actions=max_actions_per_state)
    return RLPolicy(q=compact_q, priors=priors, vocabulary=vocabulary, max_state_items=max_state_items), logs


def solve_with_policy(
    target: WikiPage,
    similarity_model: TinyPedantixModel,
    policy: RLPolicy,
    *,
    max_steps: int = 80,
) -> list[RLStep]:
    game = PedantixGame(target, similarity_model=similarity_model)
    history: list[RLStep] = []
    guessed: set[str] = set()
    for _ in range(max_steps):
        action = policy.choose(history, guessed)
        if action is None:
            break
        before_title_hits = _revealed_title_hits(game)
        result = game.guess(action)
        guessed.add(action)
        title_hits = _revealed_title_hits(game) - before_title_hits
        step = _step_from_result(action, result, title_hits)
        history.append(step)
        if result.solved:
            break
    return history


def state_key(history: list[RLStep], max_items: int = 8) -> str:
    if not history:
        return "START"
    parts = []
    for step in history[-max_items:]:
        exact_bin = _bin(step.exact_hits, [0, 1, 2, 5])
        semantic_bin = _bin(step.semantic_hits, [0, 3, 10, 25])
        title_bin = _bin(step.title_hits, [0, 1, 2])
        parts.append(f"{step.guess}:{exact_bin}:{semantic_bin}:{title_bin}")
    return "|".join(parts)


def build_action_space(pages: list[WikiPage], *, max_words: int) -> tuple[list[str], dict[str, float]]:
    doc_freq: Counter[str] = Counter()
    title_freq: Counter[str] = Counter()
    for page in pages:
        words = set(content_words(page.full_text))
        doc_freq.update(words)
        title_freq.update(page.title_words)
    n_pages = max(1, len(pages))
    priors: dict[str, float] = {}
    for word, count in doc_freq.items():
        if len(word) < 3 or word.isdigit() or word in STOPWORDS:
            continue
        p = count / n_pages
        entropy = p * (1 - p)
        theme_bonus = 1.0 if word in THEME_SEEDS else 0.0
        title_bonus = title_freq[word] / n_pages
        priors[word] = 4.0 * entropy + 1.2 * title_bonus + theme_bonus

    for seed in THEME_SEEDS:
        priors.setdefault(seed, 0.5)

    ranked_prior = [word for word, _ in sorted(priors.items(), key=lambda item: item[1], reverse=True)]
    ranked_titles = [
        word
        for word, _ in title_freq.most_common()
        if len(word) >= 3 and not word.isdigit() and word not in STOPWORDS and word in priors
    ]
    vocabulary: list[str] = []
    for word in THEME_SEEDS + ranked_titles[: max(50, max_words // 3)] + ranked_prior:
        if word not in vocabulary:
            vocabulary.append(word)
        if len(vocabulary) >= max_words:
            break
    return vocabulary, {word: priors[word] for word in vocabulary}


def _choose_training_action(
    q: dict[str, dict[str, float]],
    priors: dict[str, float],
    vocabulary: list[str],
    state: str,
    guessed: set[str],
    epsilon: float,
    rng: random.Random,
) -> str | None:
    available = [word for word in vocabulary if word not in guessed]
    if not available:
        return None
    if rng.random() < epsilon:
        sample = available[: min(180, len(available))]
        weights = [max(0.01, priors.get(word, 0.01)) for word in sample]
        return rng.choices(sample, weights=weights, k=1)[0]
    ranked = sorted(q.get(state, {}).items(), key=lambda item: item[1], reverse=True)
    for word, _ in ranked:
        if word not in guessed:
            return word
    return max(available[: min(180, len(available))], key=lambda word: priors.get(word, 0.0))


def _sample_rollout(
    logits: dict[str, dict[str, float]],
    priors: dict[str, float],
    vocabulary: list[str],
    page: WikiPage,
    similarity_model: TinyPedantixModel,
    max_steps: int,
    temperature: float,
    top_k: int,
    max_state_items: int,
    rng: random.Random,
) -> dict:
    game = PedantixGame(page, similarity_model=similarity_model)
    history: list[RLStep] = []
    guessed: set[str] = set()
    decisions: list[dict] = []
    total_reward = 0.0

    for _ in range(max_steps):
        state = state_key(history, max_state_items)
        sampled = _sample_policy_action(logits, priors, vocabulary, state, guessed, temperature, top_k, rng)
        if sampled is None:
            break
        action, actions, probs = sampled
        before_title_hits = _revealed_title_hits(game)
        result = game.guess(action)
        guessed.add(action)
        title_hits = _revealed_title_hits(game) - before_title_hits
        step = _step_from_result(action, result, title_hits)
        history.append(step)
        decisions.append({"state": state, "action": action, "actions": actions, "probs": probs})
        total_reward += _reward(step)
        if result.solved:
            break

    if not history or not history[-1].solved:
        total_reward -= 8.0
    return {"reward": total_reward, "steps": len(history), "solved": bool(history and history[-1].solved), "decisions": decisions}


def _sample_policy_action(
    logits: dict[str, dict[str, float]],
    priors: dict[str, float],
    vocabulary: list[str],
    state: str,
    guessed: set[str],
    temperature: float,
    top_k: int,
    rng: random.Random,
) -> tuple[str, list[str], list[float]] | None:
    available = [word for word in vocabulary if word not in guessed]
    if not available:
        return None
    state_logits = logits.get(state, {})
    ranked = sorted(
        available[: min(max(top_k * 3, top_k), len(available))],
        key=lambda word: state_logits.get(word, 0.0) + 0.15 * priors.get(word, 0.0),
        reverse=True,
    )[:top_k]
    scores = [(state_logits.get(word, 0.0) + 0.15 * priors.get(word, 0.0)) / max(0.05, temperature) for word in ranked]
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    total = sum(weights)
    probs = [weight / total for weight in weights]
    action = rng.choices(ranked, weights=probs, k=1)[0]
    return action, ranked, probs


def _clip_state_logits(state_logits: dict[str, float], *, max_actions: int = 80) -> None:
    for action, value in list(state_logits.items()):
        state_logits[action] = max(-8.0, min(8.0, value))
    if len(state_logits) <= max_actions:
        return
    keep = {
        action
        for action, _ in sorted(state_logits.items(), key=lambda item: abs(item[1]), reverse=True)[:max_actions]
    }
    for action in list(state_logits):
        if action not in keep:
            del state_logits[action]


def _compact_logits(
    logits: dict[str, dict[str, float]],
    *,
    max_states: int,
    max_actions: int,
) -> dict[str, dict[str, float]]:
    ranked_states = sorted(
        logits.items(),
        key=lambda item: max((abs(value) for value in item[1].values()), default=0.0),
        reverse=True,
    )[:max_states]
    return {
        state: dict(sorted(actions.items(), key=lambda item: item[1], reverse=True)[:max_actions])
        for state, actions in ranked_states
        if actions
    }


def _step_from_result(action: str, result: GuessResult, title_hits: int) -> RLStep:
    return RLStep(
        guess=canonical_word(action),
        exact_hits=len(result.exact),
        semantic_hits=len(result.semantic),
        title_hits=title_hits,
        solved=result.solved,
    )


def _reward(step: RLStep) -> float:
    if step.solved:
        return 120.0
    reward = -1.0
    reward += min(8, step.exact_hits) * 1.2
    reward += min(4, step.title_hits) * 10.0
    reward += min(25, step.semantic_hits) * 0.04
    return reward


def _revealed_title_hits(game: PedantixGame) -> int:
    return len(game.title_word_indices & game.revealed)


def _bin(value: int, cuts: list[int]) -> str:
    for idx, cut in enumerate(cuts):
        if value <= cut:
            return str(idx)
    return str(len(cuts))
