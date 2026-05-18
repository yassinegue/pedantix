from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


WORD_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    is_word: bool
    norm: str
    canon: str
    in_title: bool = False


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_word(value: str) -> str:
    value = strip_accents(value.lower().strip())
    value = re.sub(r"^[^\w]+|[^\w]+$", "", value, flags=re.UNICODE)
    return value


def canonical_word(value: str) -> str:
    """Cheap French-ish canonicalization.

    This is deliberately small and dependency-free. It is not a real lemmatizer,
    but it captures enough plural/feminine variants to make local play usable.
    """
    word = normalize_word(value)
    if len(word) <= 3:
        return word
     if word == "pays":
        return word
    if word.endswith(("ais", "ois", "is")):
        return word

    replacements = (
        ("aux", "al"),
        ("eaux", "eau"),
        ("euses", "eur"),
        ("euse", "eur"),
        ("trices", "teur"),
        ("trice", "teur"),
        ("ives", "if"),
        ("ive", "if"),
        ("aises", "ais"),
        ("oises", "ois"),
        ("ees", "e"),
        ("ee", "e"),
        ("es", "e"),
        ("s", ""),
        ("x", ""),
    )
    for suffix, replacement in replacements:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def tokenize(text: str, *, in_title: bool = False) -> list[Token]:
    tokens: list[Token] = []
    for part in WORD_RE.findall(text):
        norm = normalize_word(part)
        is_word = bool(norm and any(ch.isalnum() for ch in norm))
        tokens.append(
            Token(
                text=part,
                is_word=is_word,
                norm=norm if is_word else "",
                canon=canonical_word(norm) if is_word else "",
                in_title=in_title,
            )
        )
    return tokens


def word_set(text: str) -> set[str]:
    return {tok.canon for tok in tokenize(text) if tok.is_word and tok.canon}


def content_words(text: str, min_len: int = 3) -> list[str]:
    words = []
    seen = set()
    for tok in tokenize(text):
        if tok.is_word and len(tok.canon) >= min_len and tok.canon not in seen:
            words.append(tok.canon)
            seen.add(tok.canon)
    return words
