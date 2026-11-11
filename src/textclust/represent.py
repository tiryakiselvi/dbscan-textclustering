"""Text representations: TF-IDF (sparse), TF-IDF + SVD (LSA) and sentence embeddings."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)


def tfidf(texts: list[str], min_df: int = 5, max_df: float = 0.5, ngram_max: int = 2,
          max_features: int | None = 50_000) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        # keep tokens of all scripts; the default pattern drops one-letter words only
        token_pattern=r"(?u)\b\w\w+\b",
    )
    X = vectorizer.fit_transform(texts)
    # rows without any surviving term get a zero vector; cosine distance is undefined there
    n_empty = int((X.getnnz(axis=1) == 0).sum())
    if n_empty:
        log.warning("%d documents have no TF-IDF term after filtering", n_empty)
    return X.tocsr(), vectorizer


def reduce(X, n_components: int, seed: int) -> np.ndarray:
    """SVD for sparse input (LSA), PCA for dense input; output is L2-normalised."""
    if sparse.issparse(X):
        model = TruncatedSVD(n_components=n_components, random_state=seed)
    else:
        model = PCA(n_components=n_components, random_state=seed)
    Z = model.fit_transform(X)
    explained = float(np.sum(model.explained_variance_ratio_))
    log.info("reduced to %d dims, explained variance %.3f", n_components, explained)
    return normalize(Z)


def cache_key(texts: list[str], model_name: str) -> str:
    h = hashlib.sha256(model_name.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def embed(texts: list[str], model_name: str, cache_dir: str | Path, prefix: str = "",
          batch_size: int = 64, max_seq_length: int = 512, device: str | None = None) -> np.ndarray:
    """Sentence embeddings with an on-disk cache keyed by model and exact input texts."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{model_name.replace('/', '__')}_{cache_key(texts, model_name)}.npy"
    if path.exists():
        log.info("loading cached embeddings %s", path.name)
        return np.load(path)

    from sentence_transformers import SentenceTransformer  # optional dependency

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    E = model.encode([prefix + t for t in texts], batch_size=batch_size, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
    np.save(path, E.astype(np.float32))
    return E


def load_cached_embeddings(texts: list[str], model_name: str, cache_dir: str | Path) -> np.ndarray | None:
    path = Path(cache_dir) / f"{model_name.replace('/', '__')}_{cache_key(texts, model_name)}.npy"
    return np.load(path) if path.exists() else None
