import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from textclust import cluster, evaluate
from textclust.data import CorpusSpec, build_text, clean_text, sample_indices, validate


@pytest.fixture
def blobs():
    # three blobs at 120 degree angles: well separated under cosine distance
    centers = [[5.0, 0.0], [-2.5, 4.33], [-2.5, -4.33]]
    return make_blobs(n_samples=300, centers=centers, cluster_std=0.3, random_state=0)


def test_clean_text_removes_html_breaks():
    assert clean_text("good<br /><br />film  ") == "good film"


def test_build_text_joins_fields_and_skips_empty():
    rec = {"headline": "Title", "short_description": ""}
    assert build_text(rec, ("headline", "short_description")) == "Title"


def test_sample_is_reproducible_and_sorted():
    a = sample_indices(1000, 100, seed=42)
    b = sample_indices(1000, 100, seed=42)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)
    assert len(np.unique(a)) == 100


def test_validate_rejects_wrong_size():
    spec = CorpusSpec("x", "x.jsonl", ("text",), ("label",), expected_size=3)
    with pytest.raises(ValueError):
        validate([{"text": "a", "label": "p"}], spec)


def test_k_distance_matches_dbscan_core_definition(blobs):
    X, _ = blobs
    D = cluster.distance_matrix(X)
    k = 10
    kdist_unsorted = np.partition(D, k - 1, axis=1)[:, k - 1]
    eps = float(np.median(kdist_unsorted))
    res = cluster.run_dbscan(D, eps, k)
    # a point is core exactly if its k-distance (self included) is <= eps
    assert res.extra["n_core"] == int((kdist_unsorted <= eps).sum())


def test_dbscan_recovers_separated_blobs(blobs):
    X, y = blobs
    D = cluster.distance_matrix(X)
    res = cluster.run_dbscan(D, eps=0.01, min_samples=5)
    ext = evaluate.external(y, res.labels)
    assert ext["ari_clustered"] > 0.95


def test_noise_handling_in_metrics():
    y = np.array(["a"] * 5 + ["b"] * 5)
    labels = np.array([0, 0, 0, 0, -1, 1, 1, 1, 1, -1])
    ext = evaluate.external(y, labels)
    assert ext["coverage"] == pytest.approx(0.8)
    assert ext["ari_clustered"] == pytest.approx(1.0)
    assert ext["ari_all"] < 1.0  # noise counted as its own group lowers agreement


def test_silhouette_excludes_noise():
    X = np.array([[1, 0], [1, 0.01], [0, 1], [0.01, 1], [1, 1]], dtype=float)
    labels = np.array([0, 0, 1, 1, -1])
    D = cluster.distance_matrix(X)
    res = evaluate.internal(D, labels)
    assert res["noise_ratio"] == pytest.approx(0.2)
    assert res["silhouette_clustered"] > res["silhouette_all"]


def test_single_cluster_gives_nan_silhouette():
    D = cluster.distance_matrix(np.random.default_rng(0).random((20, 3)))
    res = evaluate.internal(D, np.zeros(20, dtype=int))
    assert np.isnan(res["silhouette_clustered"])


def test_knee_of_convex_curve():
    y = np.concatenate([np.linspace(0, 1, 90), np.linspace(1.1, 10, 10)])
    assert 0.9 <= cluster.knee(y) <= 1.2


def test_majority_mapping_marks_noise():
    y = np.array(["a", "a", "b", "b"])
    labels = np.array([0, 0, 1, -1])
    pred, mapping = evaluate.majority_mapping(y, labels)
    assert mapping == {0: "a", 1: "b"}
    assert pred[-1] == "NOISE"


def test_optics_extracts_several_xi(blobs):
    X, _ = blobs
    results = cluster.run_optics(X, min_samples=10, xis=[0.05, 0.1])
    assert [r.params["xi"] for r in results] == [0.05, 0.1]
    assert all(len(r.labels) == len(X) for r in results)
    json.dumps(results[0].params)  # params must stay serialisable
