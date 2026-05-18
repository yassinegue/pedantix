from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .dataset import WikiPage
from .text import canonical_word


class SimilarityModel(Protocol):
    def similarity(self, left: str, right: str) -> float:
        ...


@dataclass(frozen=True)
class GuessResult:
    guess: str
    exact: dict[int, str]
    semantic: dict[int, int]
    solved: bool

    @property
    def hit_count(self) -> int:
        return len(self.exact)


@dataclass
class PedantixGame:
    page: WikiPage
    similarity_model: SimilarityModel | None = None
    semantic_threshold: float = 0.34
    revealed: set[int] = field(default_factory=set)
    guessed: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.tokens = self.page.tokens()
        self.title_word_indices = {
            idx for idx, tok in enumerate(self.tokens) if tok.is_word and tok.in_title
        }

    def guess(self, word: str) -> GuessResult:
        canon = canonical_word(word)
        if not canon:
            return GuessResult(word, {}, {}, self.solved)
        self.guessed.add(canon)

        exact: dict[int, str] = {}
        semantic: dict[int, int] = {}

        for idx, tok in enumerate(self.tokens):
            if not tok.is_word:
                continue
            if tok.canon == canon or tok.norm == canon:
                self.revealed.add(idx)
                exact[idx] = tok.text

        if self.similarity_model is not None:
            for idx, tok in enumerate(self.tokens):
                if not tok.is_word or idx in self.revealed or idx in self.title_word_indices:
                    continue
                sim = self.similarity_model.similarity(canon, tok.canon)
                if sim >= self.semantic_threshold:
                    semantic[idx] = round(sim * 100)

        return GuessResult(word, exact, semantic, self.solved)

    @property
    def solved(self) -> bool:
        return bool(self.title_word_indices) and self.title_word_indices <= self.revealed

    def masked_text(self, *, reveal_semantic: dict[int, int] | None = None) -> str:
        reveal_semantic = reveal_semantic or {}
        parts: list[str] = []
        for idx, tok in enumerate(self.tokens):
            if not tok.is_word:
                parts.append(tok.text)
            elif idx in self.revealed:
                parts.append(tok.text)
            elif idx in reveal_semantic:
                parts.append(f"[{reveal_semantic[idx]}]")
            else:
                parts.append("█" * max(1, len(tok.text)))
        return "".join(parts)
