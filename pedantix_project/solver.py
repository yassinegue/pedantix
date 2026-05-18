from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import WikiPage
from .model import TinyPedantixModel
from .simulator import GuessResult, PedantixGame
from .text import canonical_word


@dataclass
class SolveStep:
    guess: str
    exact_hits: int
    semantic_hits: int
    candidates: int
    solved: bool


@dataclass
class TinySolver:
    pages: list[WikiPage]
    model: TinyPedantixModel
    guessed: set[str] = field(default_factory=set)
    present: set[str] = field(default_factory=set)
    absent: set[str] = field(default_factory=set)
    candidates: list[WikiPage] = field(init=False)

    def __post_init__(self) -> None:
        self.candidates = list(self.pages)

    def observe(self, guess: str, result: GuessResult) -> None:
        canon = canonical_word(guess)
        self.guessed.add(canon)
        if result.exact:
            self.present.add(canon)
        else:
            self.absent.add(canon)
        self._filter_candidates()

    def next_guess(self) -> str | None:
        for word in self.model.starter_words:
            canon = canonical_word(word)
            if canon and canon not in self.guessed:
                return canon

        if len(self.candidates) <= 3 and self.candidates:
            for page in self._rank_candidates()[:3]:
                for word in sorted(page.title_words, key=lambda w: -self.model.idf.get(w, 1.0)):
                    if word not in self.guessed:
                        return word

        return self._best_information_word()

    def _filter_candidates(self) -> None:
        filtered = []
        for page in self.candidates:
            words = page.words
            if not self.present <= words:
                continue
            if self.absent & words:
                continue
            filtered.append(page)
        self.candidates = filtered or self.candidates

    def _rank_candidates(self) -> list[WikiPage]:
        def score(page: WikiPage) -> float:
            words = page.words
            title_words = page.title_words
            present_score = sum(self.model.idf.get(w, 1.0) for w in self.present if w in words)
            title_score = sum(self.model.idf.get(w, 1.0) for w in self.present if w in title_words)
            return present_score + 2.5 * title_score - 0.01 * len(words)

        return sorted(self.candidates, key=score, reverse=True)

    def _best_information_word(self) -> str | None:
        if not self.candidates:
            pool = self.pages
        else:
            pool = self._rank_candidates()[: min(250, len(self.candidates))]

        counts: dict[str, int] = {}
        for page in pool:
            for word in page.words:
                if word not in self.guessed and len(word) >= 3 and not word.isdigit():
                    counts[word] = counts.get(word, 0) + 1

        if not counts:
            for word in self.model.vocabulary:
                if word not in self.guessed:
                    return word
            return None

        total = len(pool)

        def word_score(item: tuple[str, int]) -> float:
            word, count = item
            p = count / total
            uncertainty = p * (1.0 - p)
            idf = self.model.idf.get(word, 1.0)
            title_bonus = sum(1 for page in pool if word in page.title_words) / total
            return uncertainty + 0.18 * p * idf + 0.8 * title_bonus

        return max(counts.items(), key=word_score)[0]


def solve_page(
    target: WikiPage,
    pages: list[WikiPage],
    model: TinyPedantixModel,
    *,
    max_steps: int = 80,
) -> list[SolveStep]:
    game = PedantixGame(target, similarity_model=model)
    solver = TinySolver(pages=pages, model=model)
    steps: list[SolveStep] = []

    for _ in range(max_steps):
        guess = solver.next_guess()
        if not guess:
            break
        result = game.guess(guess)
        solver.observe(guess, result)
        steps.append(
            SolveStep(
                guess=guess,
                exact_hits=len(result.exact),
                semantic_hits=len(result.semantic),
                candidates=len(solver.candidates),
                solved=result.solved,
            )
        )
        if result.solved:
            break

    return steps
