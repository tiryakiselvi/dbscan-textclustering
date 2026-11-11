"""Loading, validation and sampling of the three course corpora.

All files are JSONL (one JSON object per line), including the letter files
that carry a .json extension.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    path: str  # relative to the data directory
    text_fields: tuple[str, ...]
    label_fields: tuple[str, ...]
    expected_size: int
    expected_classes: dict[str, int] = field(default_factory=dict)


CORPORA: dict[str, CorpusSpec] = {
    "letters_small": CorpusSpec(
        name="letters_small",
        path="letters/clustering/clustering_data_small.json",
        text_fields=("text",),
        label_fields=("author", "lang"),
        expected_size=1_000,
        expected_classes={"author": 2},
    ),
    "letters": CorpusSpec(
        name="letters",
        path="letters/clustering/clustering_data_full.json",
        text_fields=("text",),
        label_fields=("author", "lang"),
        expected_size=43_958,
        expected_classes={"author": 7, "lang": 8},
    ),
    "news": CorpusSpec(
        name="news",
        path="news/full_data/classification_news.jsonl",
        text_fields=("headline", "short_description"),
        label_fields=("category",),
        expected_size=199_724,
        expected_classes={"category": 34},
    ),
    "imdb": CorpusSpec(
        name="imdb",
        path="sentiment/full_data/classification_sentiment.jsonl",
        text_fields=("text",),
        label_fields=("sentiment",),
        expected_size=50_000,
        expected_classes={"sentiment": 2},
    ),
}

_HTML_BREAK = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({err})") from err
    return records


def clean_text(text: str) -> str:
    text = _HTML_BREAK.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def build_text(record: dict, fields: tuple[str, ...]) -> str:
    parts = [str(record.get(f) or "").strip() for f in fields]
    return clean_text(". ".join(p for p in parts if p))


def validate(records: list[dict], spec: CorpusSpec) -> dict:
    """Check schema and label distribution against the data manifest.

    Returns a small report; raises if something is clearly wrong.
    """
    if len(records) != spec.expected_size:
        raise ValueError(f"{spec.name}: expected {spec.expected_size} records, got {len(records)}")

    required = set(spec.text_fields) | set(spec.label_fields)
    missing = Counter(f for r in records for f in required if f not in r)
    if missing:
        raise ValueError(f"{spec.name}: missing fields {dict(missing)}")

    report = {"corpus": spec.name, "n_records": len(records), "labels": {}}
    for label in spec.label_fields:
        counts = Counter(r[label] for r in records)
        expected = spec.expected_classes.get(label)
        if expected is not None and len(counts) != expected:
            raise ValueError(f"{spec.name}: expected {expected} values for '{label}', got {len(counts)}")
        report["labels"][label] = dict(counts.most_common())

    texts = [build_text(r, spec.text_fields) for r in records]
    report["empty_texts"] = sum(1 for t in texts if not t)
    lengths = np.array([len(t.split()) for t in texts])
    report["words_median"] = float(np.median(lengths))
    report["words_max"] = int(lengths.max())
    return report


def sample_indices(n_total: int, sample_size: int | None, seed: int) -> np.ndarray:
    """Uniform random sample without replacement (no stratification: that would use labels)."""
    if sample_size is None or sample_size >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=sample_size, replace=False))


@dataclass
class Corpus:
    spec: CorpusSpec
    texts: list[str]
    labels: dict[str, np.ndarray]
    indices: np.ndarray  # positions in the original file


def load_corpus(data_dir: str | Path, name: str, sample_size: int | None, seed: int) -> tuple[Corpus, dict]:
    spec = CORPORA[name]
    records = read_jsonl(Path(data_dir) / spec.path)
    report = validate(records, spec)

    # empty texts are dropped before sampling so that every method sees the same input
    keep = [i for i, r in enumerate(records) if build_text(r, spec.text_fields)]
    chosen = np.asarray(keep)[sample_indices(len(keep), sample_size, seed)]

    texts = [build_text(records[i], spec.text_fields) for i in chosen]
    labels = {lab: np.array([str(records[i][lab]) for i in chosen]) for lab in spec.label_fields}
    report["sample_size"] = len(chosen)
    report["seed"] = seed
    return Corpus(spec=spec, texts=texts, labels=labels, indices=chosen), report
