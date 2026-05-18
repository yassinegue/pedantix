from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from .dataset import WikiPage
from .text import content_words


@dataclass
class TinyPedantixModel:
    """Small JSON-serializable model.

    It learns document frequency weights and a compact word-neighbor table from
    local Wikipedia introductions. This is not Fauconnier's full Word2Vec model;
    it is the tiny local fallback that can run comfortably on a MacBook Air.
    """

    idf: dict[str, float]
    neighbors: dict[str, dict[str, float]]
    vocabulary: list[str]
    starter_words: list[str] = field(default_factory=list)

    def similarity(self, left: str, right: str) -> float:
        if left == right:
            return 1.0
        return self.neighbors.get(left, {}).get(right, self.neighbors.get(right, {}).get(left, 0.0))

    def save(self, path: str | Path) -> None:
        payload = {
            "idf": self.idf,
            "neighbors": self.neighbors,
            "vocabulary": self.vocabulary,
            "starter_words": self.starter_words,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyPedantixModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            idf={str(k): float(v) for k, v in payload["idf"].items()},
            neighbors={
                str(k): {str(n): float(s) for n, s in v.items()} for k, v in payload["neighbors"].items()
            },
            vocabulary=list(payload["vocabulary"]),
            starter_words=list(payload.get("starter_words", [])),
        )


def train_tiny_model(
    pages: list[WikiPage],
    *,
    max_vocab: int = 5000,
    max_neighbors: int = 25,
    starter_words: list[str] | None = None,
) -> TinyPedantixModel:
    doc_freq: Counter[str] = Counter()
    term_freq: Counter[str] = Counter()
    page_words: list[list[str]] = []

    for page in pages:
        words = content_words(page.full_text)
        page_words.append(words)
        unique = set(words)
        doc_freq.update(unique)
        term_freq.update(words)

    vocab = [
        word
        for word, _ in term_freq.most_common()
        if len(word) >= 3 and not word.isdigit()
    ][:max_vocab]
    vocab_set = set(vocab)
    n_docs = max(1, len(pages))
    idf = {word: math.log((1 + n_docs) / (1 + doc_freq[word])) + 1.0 for word in vocab}

    cooc: dict[str, Counter[str]] = defaultdict(Counter)
    for words in page_words:
        filtered = sorted(set(word for word in words if word in vocab_set))
        for left, right in combinations(filtered[:250], 2):
            weight = idf[left] * idf[right]
            cooc[left][right] += weight
            cooc[right][left] += weight

    neighbors: dict[str, dict[str, float]] = {}
    for word, counts in cooc.items():
        if not counts:
            continue
        max_score = max(counts.values())
        top = counts.most_common(max_neighbors)
        neighbors[word] = {other: round(score / max_score, 4) for other, score in top}

    starters = starter_words or [
        "france",
        "monde",
        "siecle",
        "ville",
        "pays",
        "guerre",
        "homme",
        "femme",
        "art",
        "science",
        "politique",
        "histoire",
    ]
    starters = [word for word in starters if word in vocab_set or not vocab_set]

    return TinyPedantixModel(idf=idf, neighbors=neighbors, vocabulary=vocab, starter_words=starters)
