"""Clustering methods and the parameter heuristics for DBSCAN.

All density-based methods work on a precomputed cosine distance matrix D.
At n = 10,000 the matrix needs about 0.8 GB (float64), which is cheaper than
recomputing neighbourhoods for every point of the eps sweep, and it lets
DBSCAN, OPTICS and the silhouette score use exactly the same distances.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, KMeans, cluster_optics_xi
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class ClusterResult:
    method: str
    labels: np.ndarray
    params: dict
    runtime_s: float
    extra: dict | None = None


def distance_matrix(X) -> np.ndarray:
    D = cosine_distances(X)
    np.clip(D, 0.0, 2.0, out=D)  # rounding can produce tiny negatives
    np.fill_diagonal(D, 0.0)
    return D


def k_distances(D: np.ndarray, k: int) -> np.ndarray:
    """Sorted distance of every point to its k-th neighbour, the point itself counted.

    This matches scikit-learn's DBSCAN, where min_samples includes the point,
    so with k = min_samples a point is a core point exactly if its k-distance <= eps.
    """
    kth = np.partition(D, k - 1, axis=1)[:, k - 1]
    return np.sort(kth)


def knee(values: np.ndarray) -> float:
    """Knee of a sorted, increasing curve: the point farthest below the chord
    between the first and the last point (the geometric idea behind Kneedle)."""
    y = np.asarray(values, dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    span = y[-1] - y[0]
    if span <= 0:
        return float(y[-1])
    gap = x - (y - y[0]) / span
    return float(y[int(np.argmax(gap))])


def eps_grid(kdist: np.ndarray, n: int = 30, low: float = 5, high: float = 95) -> np.ndarray:
    lo, hi = np.percentile(kdist, [low, high])
    return np.unique(np.linspace(lo, hi, n))


def run_dbscan(D: np.ndarray, eps: float, min_samples: int) -> ClusterResult:
    start = time.perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = model.fit_predict(D)
    return ClusterResult("dbscan", labels, {"eps": float(eps), "min_samples": min_samples},
                         time.perf_counter() - start, {"n_core": int(len(model.core_sample_indices_))})


def run_optics(X, min_samples: int, xis: list[float],
               min_cluster_size: float | int | None = None) -> list[ClusterResult]:
    """Fit OPTICS once, then extract clusters for several xi values from the same ordering.

    OPTICS gets the feature matrix, not D: with metric="precomputed" scikit-learn
    re-validates the whole n x n matrix for every single neighbourhood query,
    which makes the fit cubic in practice. Cosine distances on X are identical.
    """
    start = time.perf_counter()
    model = OPTICS(min_samples=min_samples, metric="cosine", algorithm="brute", cluster_method="xi",
                   xi=xis[0], min_cluster_size=min_cluster_size)
    model.fit(X)
    fit_time = time.perf_counter() - start

    results = []
    for xi in xis:
        t0 = time.perf_counter()
        labels, _ = cluster_optics_xi(
            reachability=model.reachability_, predecessor=model.predecessor_,
            ordering=model.ordering_, min_samples=min_samples, xi=xi,
            min_cluster_size=min_cluster_size,
        )
        results.append(ClusterResult(
            "optics", labels,
            {"min_samples": min_samples, "xi": xi, "min_cluster_size": min_cluster_size},
            fit_time + (time.perf_counter() - t0),
            {"reachability_ordered": model.reachability_[model.ordering_], "ordering": model.ordering_},
        ))
    return results


def run_kmeans(X, k: int, seed: int, n_init: int = 10) -> ClusterResult:
    """Reference method. k = number of gold classes is an 'oracle k' and is reported as such."""
    start = time.perf_counter()
    model = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed)
    labels = model.fit_predict(X)
    return ClusterResult("kmeans", labels, {"k": k, "seed": seed, "n_init": n_init},
                         time.perf_counter() - start, {"inertia": float(model.inertia_)})
