from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

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


class FastTextEmbedder:
    """Thin wrapper around the pre-built cc.fr.300 FastText NPZ.
    Implements the SimilarityModel protocol so it can replace TinyPedantixModel
    as game.similarity_model while keeping IDF/vocab from the TinyModel."""

    def __init__(
        self,
        matrix: "np.ndarray",
        vocab_index: dict[str, int],
        idf: dict[str, float],
        vocabulary: list[str],
        starter_words: list[str],
    ) -> None:
        self._matrix = matrix  # (N, 300) float32, L2-normalised
        self._vocab_index = vocab_index
        self.idf = idf
        self.vocabulary = vocabulary
        self.starter_words = starter_words

    def similarity(self, left: str, right: str) -> float:
        if left == right:
            return 1.0
        i = self._vocab_index.get(left, -1)
        j = self._vocab_index.get(right, -1)
        if i < 0 or j < 0:
            return 0.0
        return float(np.dot(self._matrix[i], self._matrix[j]))

    @classmethod
    def load(cls, npz_path: "str | Path", tiny_model: TinyPedantixModel) -> "FastTextEmbedder":
        data = np.load(str(npz_path), allow_pickle=True)
        matrix = data["matrix"].astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        vocab = data["vocab"].tolist()
        vocab_index = {str(w): i for i, w in enumerate(vocab)}
        return cls(
            matrix=matrix,
            vocab_index=vocab_index,
            idf=tiny_model.idf,
            vocabulary=tiny_model.vocabulary,
            starter_words=tiny_model.starter_words,
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


# ---------------------------------------------------------------------------
# FastTextWikiModel — richer signal model for LLM trajectory generation
# ---------------------------------------------------------------------------

class FastTextWikiModel:
    """FastText cc.fr.300 vectors filtered to Wikipedia vocabulary.

    Backed by a float32 numpy matrix (N×300) for fast vectorised similarity.
    Stores the full intersection of FastText and Wikipedia vocabs (~400-600K words)
    in ~700MB of float32 binary, vs ~3-4GB as JSON.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        word2idx: dict[str, int],
        centroid: "np.ndarray | None" = None,
        idf: "dict[str, float] | None" = None,
        starter_words: "list[str] | None" = None,
    ) -> None:
        self.matrix = matrix      # shape (N, 300), float32, unit-normalised rows
        self.word2idx = word2idx  # word -> row index
        # IDF and starter_words — copied from TinyPedantixModel when loaded via
        # _load_similarity_model. Used by _token_information and _french_dictionary.
        self.idf: dict[str, float] = idf or {}
        self.starter_words: list[str] = starter_words or []
        # Corpus centroid: mean of all word vectors, unit-normalised.
        # background_similarity(w) = dot(w_vec, centroid) gives the expected
        # similarity of w to a random Wikipedia article — the "average signal".
        # Contrastive reward = sim(w, page) - background_similarity(w) strips
        # out the part of the signal that generic words get on every page.
        if centroid is not None:
            norm = np.linalg.norm(centroid)
            self.centroid: np.ndarray = centroid / norm if norm > 1e-9 else centroid
        else:
            c = matrix.mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(c)
            self.centroid = c / norm if norm > 1e-9 else c

    def similarity(self, left: str, right: str) -> float:
        if left == right:
            return 1.0
        il = self.word2idx.get(left)
        ir = self.word2idx.get(right)
        if il is None or ir is None:
            return 0.0
        return float(np.clip(self.matrix[il] @ self.matrix[ir], 0.0, 1.0))

    def background_similarity(self, word: str) -> float:
        """Expected similarity of `word` to a random Wikipedia article.

        Computed as dot(word_vec, corpus_centroid). Generic words (france, partie,
        monde) have high background similarity (~0.3–0.5). Domain-specific words
        (chiropratique, kaltchyk) have low background similarity (~0.0–0.1).

        Use contrastive_sim = max(0, sim_to_page - background_similarity(word))
        as the reward signal to strip out uninformative generic guesses.
        """
        i = self.word2idx.get(word)
        if i is None:
            return 0.0
        return float(np.clip(self.matrix[i] @ self.centroid, 0.0, 1.0))

    def max_sim_to_words(self, guess: str, page_canons: Sequence[str]) -> float:
        """Return max cosine similarity between guess and any word in page_canons.

        Vectorised: one matmul instead of len(page_canons) individual lookups.
        """
        if guess in page_canons:
            return 1.0
        ig = self.word2idx.get(guess)
        if ig is None:
            return 0.0
        valid = [self.word2idx[w] for w in page_canons if w in self.word2idx]
        if not valid:
            return 0.0
        sims = self.matrix[valid] @ self.matrix[ig]
        return float(np.max(sims))

    def contrastive_sim(self, guess: str, page_canons: Sequence[str]) -> float:
        """max_sim_to_words(guess, page) minus background_similarity(guess).

        Positive only when the guess is more similar to THIS page than to the
        average Wikipedia article. This is the information-theoretic reward:
        generic words get ~0, domain-specific words get a large positive value.
        """
        return max(0.0, self.max_sim_to_words(guess, page_canons) - self.background_similarity(guess))

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def save(self, path: str | Path) -> None:
        path = Path(path).with_suffix(".npz")
        vocab = [""] * len(self.word2idx)
        for w, i in self.word2idx.items():
            vocab[i] = w
        np.savez_compressed(
            path,
            matrix=self.matrix,
            vocab=np.array(vocab, dtype=object),
            centroid=self.centroid,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FastTextWikiModel":
        path = Path(path)
        # Support old JSON format for backward compatibility
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            vocab_list = list(payload["vectors"].keys())
            matrix = np.array(list(payload["vectors"].values()), dtype=np.float32)
            word2idx = {w: i for i, w in enumerate(vocab_list)}
            return cls(matrix=matrix, word2idx=word2idx)
        # Native numpy format
        load_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        data = np.load(load_path, allow_pickle=True)
        matrix = data["matrix"].astype(np.float32)
        vocab = data["vocab"].tolist()
        word2idx = {w: i for i, w in enumerate(vocab)}
        centroid = data["centroid"].astype(np.float32) if "centroid" in data else None
        return cls(matrix=matrix, word2idx=word2idx, centroid=centroid)

    @property
    def vocabulary(self) -> list[str]:
        result = [""] * len(self.word2idx)
        for w, i in self.word2idx.items():
            result[i] = w
        return result


def build_fasttext_wiki_model(
    pages_path: Path,
    fasttext_vec_gz: Path,
    output_path: Path,
    *,
    max_vocab: int | None = None,
    verbose: bool = True,
) -> FastTextWikiModel:
    """Filter FastText cc.fr.300.vec.gz to Wikipedia vocabulary and save as .npz.

    Steps:
    1. Collect all canonical words from the Wikipedia corpus.
    2. Stream through cc.fr.300.vec.gz and keep vectors for matching words.
    3. Unit-normalise each vector (cosine sim = dot product of unit vecs).
    4. Save as compressed numpy binary (output_path with .npz suffix).

    Args:
        pages_path: path to clean_pages.jsonl
        fasttext_vec_gz: path to cc.fr.300.vec.gz
        output_path: where to save the model (suffix forced to .npz)
        max_vocab: cap on number of words to keep (None = no cap, exhaustive)
    """
    import gzip

    from .dataset import WikiPage
    from .text import canonical_word, content_words

    # Collect Wikipedia vocabulary
    if verbose:
        print("Collecting Wikipedia vocabulary...")
    wiki_words: set[str] = set()
    with open(pages_path, encoding="utf-8") as f:
        for line in f:
            page_data = json.loads(line)
            page = WikiPage(title=page_data["title"], intro=page_data["intro"])
            for w in content_words(page.full_text):
                wiki_words.add(w)
            wiki_words.add(canonical_word(page_data["title"]))
    if verbose:
        print(f"  Wikipedia vocabulary: {len(wiki_words):,} unique canonical words")

    # Stream FastText vectors, keep matching words
    if verbose:
        print(f"Streaming {fasttext_vec_gz} ...")
    vocab_list: list[str] = []
    matrix_rows: list[np.ndarray] = []
    seen: set[str] = set()

    with gzip.open(fasttext_vec_gz, "rt", encoding="utf-8") as f:
        f.readline()  # skip "vocab_size dim" header
        for line_no, line in enumerate(f, 1):
            if max_vocab is not None and len(vocab_list) >= max_vocab:
                break
            parts = line.rstrip().split(" ")
            word = parts[0]
            canon = canonical_word(word)
            if canon not in wiki_words and word not in wiki_words:
                continue
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if len(vec) != 300:
                continue
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                continue
            key = canon if canon in wiki_words else word
            if key in seen:
                continue
            seen.add(key)
            vocab_list.append(key)
            matrix_rows.append(vec / norm)
            if verbose and line_no % 200_000 == 0:
                print(f"  scanned {line_no:,} lines, kept {len(vocab_list):,} ...")

    if verbose:
        print(f"Kept {len(vocab_list):,} word vectors. Building matrix ...")
    matrix = np.stack(matrix_rows, axis=0).astype(np.float32)
    word2idx = {w: i for i, w in enumerate(vocab_list)}
    model = FastTextWikiModel(matrix=matrix, word2idx=word2idx)
    if verbose:
        print(f"Saving to {output_path} ...")
    model.save(output_path)
    if verbose:
        print("Done.")
    return model


_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
_FAUCONNIER_URL = "https://embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"


def download_fasttext_french(dest_dir: Path, *, verbose: bool = True) -> Path:
    """Download cc.fr.300.vec.gz if not already present. Returns the local path."""
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "cc.fr.300.vec.gz"
    if dest.exists():
        if verbose:
            print(f"Already downloaded: {dest}")
        return dest
    if verbose:
        print(f"Downloading FastText French vectors (~1.6 GB) from {_FASTTEXT_URL} ...")
        print("This takes ~10-30 minutes depending on connection speed.")

    def _progress(block, block_size, total):
        if total > 0 and block % 500 == 0:
            pct = block * block_size / total * 100
            print(f"  {pct:.1f}%", end="\r", flush=True)

    urllib.request.urlretrieve(_FASTTEXT_URL, dest, reporthook=_progress)
    if verbose:
        print(f"\nSaved to {dest}")
    return dest
