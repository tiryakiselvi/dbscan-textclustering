"""Internal and external evaluation, with explicit handling of DBSCAN noise (label -1).

Two views are reported for every result:

* ``all``:       every document, noise counted as one extra group "-1"
* ``clustered``: only documents assigned to a cluster, plus the coverage

The silhouette score is not defined for noise. scikit-learn does not know
about it, so passing -1 labels unchanged silently treats noise as a cluster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics

NOISE = -1


def _silhouette(D: np.ndarray, labels: np.ndarray, sample_size: int | None, seed: int) -> float:
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > len(labels) - 1:
        return float("nan")
    return float(metrics.silhouette_score(D, labels, metric="precomputed",
                                          sample_size=sample_size, random_state=seed))


def internal(D: np.ndarray, labels: np.ndarray, sample_size: int | None = None, seed: int = 0) -> dict:
    """Internal measures on the precomputed cosine distance matrix D."""
    mask = labels != NOISE
    clusters = labels[mask]
    sizes = np.bincount(clusters) if mask.any() else np.array([], dtype=int)
    sizes = sizes[sizes > 0]
    return {
        "n_clusters": int(len(sizes)),
        "noise_ratio": float(1.0 - mask.mean()),
        "largest_cluster_share": float(sizes.max() / len(labels)) if len(sizes) else 0.0,
        "silhouette_clustered": _silhouette(D[np.ix_(mask, mask)], clusters, sample_size, seed),
        "silhouette_all": _silhouette(D, labels, sample_size, seed),
    }


def _external(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    h, c, v = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    return {
        "ari": float(metrics.adjusted_rand_score(y_true, y_pred)),
        "ami": float(metrics.adjusted_mutual_info_score(y_true, y_pred)),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
    }


def external(y_true: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != NOISE
    out = {f"{k}_all": val for k, val in _external(y_true, labels).items()}
    if mask.sum() >= 2:
        out.update({f"{k}_clustered": val for k, val in _external(y_true[mask], labels[mask]).items()})
    else:
        out.update({f"{k}_clustered": float("nan") for k in ("ari", "ami", "homogeneity", "completeness", "v_measure")})
    out["coverage"] = float(mask.mean())
    return out


def contingency(y_true: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Cluster x class table (rows: clusters incl. noise, columns: gold classes)."""
    table = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y_true, name="class"))
    return table.rename(index={NOISE: "noise"})


def majority_mapping(y_true: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map every cluster to its most frequent gold class; noise stays 'NOISE'.

    Only an illustrative add-on so the result can be read in the format of a
    classification report; several clusters may map to the same class.
    """
    mapping = {}
    for c in np.unique(labels):
        if c == NOISE:
            continue
        values, counts = np.unique(y_true[labels == c], return_counts=True)
        mapping[int(c)] = str(values[np.argmax(counts)])
    y_pred = np.array([mapping.get(int(c), "NOISE") for c in labels])
    return y_pred, mapping


def classification_view(y_true: np.ndarray, labels: np.ndarray) -> dict:
    y_pred, mapping = majority_mapping(y_true, labels)
    classes = sorted(set(y_true))
    report = metrics.classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    confusion = pd.crosstab(pd.Series(y_true, name="gold"), pd.Series(y_pred, name="predicted"))
    return {"mapping": mapping, "report": report, "confusion": confusion}
