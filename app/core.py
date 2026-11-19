"""Local educational corpus and n-gram language model."""

from __future__ import annotations

import re
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

CORPUS_PATH = Path(__file__).with_name("data") / "educational_corpus.txt"
UNKNOWN_TOKEN = "<unk>"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def load_corpus(path: str | Path | None = None) -> list[str]:
    """Load educational text documents from a UTF-8 file, one document per line."""
    corpus_path = Path(path) if path is not None else CORPUS_PATH
    try:
        contents = corpus_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Could not read corpus file: {corpus_path}") from exc
    return [line.strip() for line in contents.splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    """Normalize text and return word tokens, retaining contractions as one token."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def preprocess(text: str) -> list[str]:
    """Tokenize a document and replace an empty document with no tokens."""
    return tokenize(text)


class NGramModel:
    """Bigram and trigram counts with add-k conditional probabilities."""

    def __init__(self, k: float = 1.0):
        if not math.isfinite(k) or k < 0:
            raise ValueError("smoothing k must be a finite non-negative number")
        self.k = float(k)
        self.vocabulary: set[str] = {UNKNOWN_TOKEN}
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.trigram_counts: Counter[tuple[str, str, str]] = Counter()
        self.bigram_context_counts: Counter[str] = Counter()
        self.trigram_context_counts: Counter[tuple[str, str]] = Counter()

    def fit(self, documents: Iterable[str | Sequence[str]]) -> NGramModel:
        """Count n-grams in documents without joining across document boundaries."""
        self.vocabulary = {UNKNOWN_TOKEN}
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.bigram_context_counts.clear()
        self.trigram_context_counts.clear()
        for document in documents:
            tokens = tokenize(document) if isinstance(document, str) else [str(t).casefold() for t in document]
            if not tokens:
                continue
            self.vocabulary.update(tokens)
            self.unigram_counts.update(tokens)
            for index in range(1, len(tokens)):
                self.bigram_counts[(tokens[index - 1], tokens[index])] += 1
                self.bigram_context_counts[tokens[index - 1]] += 1
            for index in range(2, len(tokens)):
                context = (tokens[index - 2], tokens[index - 1])
                self.trigram_counts[(*context, tokens[index])] += 1
                self.trigram_context_counts[context] += 1
        return self

    @property
    def vocabulary_size(self) -> int:
        return max(1, len(self.vocabulary))

    def probability(self, word: str, context: Sequence[str] = ()) -> float:
        """Return P(word | context) with add-k smoothing (context length 0–2)."""
        if len(context) > 2:
            raise ValueError("context may contain at most two tokens")
        word = word.casefold()
        if word not in self.vocabulary:
            word = UNKNOWN_TOKEN
        normalized_context = tuple(token.casefold() for token in context)
        if not normalized_context:
            count = self.unigram_counts[word]
            total = sum(self.unigram_counts.values())
        elif len(normalized_context) == 1:
            count = self.bigram_counts[(normalized_context[0], word)]
            total = self.bigram_context_counts[normalized_context[0]]
        else:
            count = self.trigram_counts[(*normalized_context, word)]
            total = self.trigram_context_counts[normalized_context]
        return (count + self.k) / (total + self.k * self.vocabulary_size) if total or self.k else 0.0

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation of the trained model."""
        return {
            "format_version": 1,
            "k": self.k,
            "vocabulary": sorted(self.vocabulary),
            "unigram_counts": dict(self.unigram_counts),
            "bigram_counts": [[*ngram, count] for ngram, count in self.bigram_counts.items()],
            "trigram_counts": [[*ngram, count] for ngram, count in self.trigram_counts.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> NGramModel:
        """Restore a model saved by :meth:`to_dict`."""
        if not isinstance(data, dict):
            raise ValueError("model data must be a JSON object")
        if data.get("format_version") != 1:
            raise ValueError("unsupported model file format")
        model = cls(k=float(data["k"]))
        model.vocabulary = set(data["vocabulary"])
        model.unigram_counts.update({str(word): int(count) for word, count in data["unigram_counts"].items()})
        for first, second, count in data["bigram_counts"]:
            model.bigram_counts[(first, second)] = int(count)
            model.bigram_context_counts[first] += int(count)
        for first, second, third, count in data["trigram_counts"]:
            model.trigram_counts[(first, second, third)] = int(count)
            model.trigram_context_counts[(first, second)] += int(count)
        if not model.unigram_counts:
            raise ValueError("model file contains no training data")
        return model


def save_model(model: NGramModel, path: str | Path) -> None:
    """Save a trained model as readable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(path: str | Path) -> NGramModel:
    """Load a model JSON file and report file or format errors clearly."""
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return NGramModel.from_dict(data)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model file not found: {target}") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model file '{target}': {exc}") from exc


def train_model(corpus: Iterable[str] | None = None, k: float = 1.0) -> NGramModel:
    """Build a model from supplied documents or the bundled educational corpus."""
    return NGramModel(k=k).fit(load_corpus() if corpus is None else corpus)


def generate_text(prompt: str, model: NGramModel | None = None, max_words: int = 40) -> str:
    """Append up to ``max_words`` locally predicted words while preserving prompt text."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must contain at least one non-space character")
    if max_words < 0:
        raise ValueError("max_words must be non-negative")
    active_model = model or train_model()
    if not isinstance(active_model, NGramModel) or not active_model.unigram_counts:
        raise ValueError("a trained n-gram model is required")
    generated = tokenize(prompt)
    if not generated or max_words == 0:
        return prompt
    continuation: list[str] = []
    for _ in range(max_words):
        context = generated[-2:]
        # Select only observed words so the model never emits the unknown sentinel.
        candidates = sorted(active_model.unigram_counts)
        next_word = max(
            candidates,
            key=lambda word: (
                active_model.probability(word, context),
                active_model.probability(word, context[-1:]) if context else 0.0,
                active_model.unigram_counts[word],
                word,
            ),
        )
        continuation.append(next_word)
        generated.append(next_word)
    return prompt + (" " if not prompt[-1].isspace() else "") + " ".join(continuation)
