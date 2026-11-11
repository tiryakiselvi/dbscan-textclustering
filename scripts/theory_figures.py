"""Illustrations for the theory part of the talk (synthetic 2D data, fixed seeds).

    python scripts/theory_figures.py  ->  figures/theory/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

OUT = Path("figures/theory")
GREEN, LIGHT, GREY, RED, INK = "#2F854C", "#9CC9A9", "#B0B0B0", "#C0392B", "#2B2B2B"
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def moons_with_noise(seed=0):
    X, _ = make_moons(n_samples=400, noise=0.06, random_state=seed)
    rng = np.random.default_rng(seed)
    noise = rng.uniform([-1.3, -0.8], [2.3, 1.3], size=(25, 2))
    return np.vstack([X, noise])


def kmeans_vs_dbscan():
    X = moons_with_noise()
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    db = DBSCAN(eps=0.15, min_samples=5).fit_predict(X)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, labels, title in [(axes[0], km, "K-Means, k = 2"), (axes[1], db, "DBSCAN, ε = 0,15, MinPts = 5 (grau = Rauschen)")]:
        colors = np.where(labels == -1, GREY, np.where(labels == 0, GREEN, RED))
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=12, linewidths=0)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis("off")
    fig.savefig(OUT / "kmeans_vs_dbscan.png")
    plt.close(fig)


def point_types():
    """Core, border and noise points for eps = 1, MinPts = 4 (point itself counted)."""
    P = np.array([[0, 0], [0.7, 0.3], [0.4, -0.6], [-0.5, 0.5], [-0.6, -0.3],
                  [1.5, 0.4], [3.3, -0.4], [-1.2, 1.2]])
    eps, min_pts = 1.0, 4
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    core = (d <= eps).sum(1) >= min_pts
    border = ~core & (d[:, core] <= eps).any(1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, p in enumerate(P):
        if core[i]:
            ax.add_patch(plt.Circle(p, eps, color=GREEN, alpha=0.08))
            ax.add_patch(plt.Circle(p, eps, fill=False, color=GREEN, lw=0.6, alpha=0.6))
    ax.scatter(*P[core].T, s=90, c=GREEN, zorder=3, label="Kernpunkt")
    ax.scatter(*P[border].T, s=90, c="white", edgecolors=GREEN, linewidths=2, zorder=3, label="Randpunkt")
    noise = ~core & ~border
    ax.scatter(*P[noise].T, s=90, c=GREY, zorder=3, label="Rauschen")
    ax.annotate("", xy=(1.0, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    ax.text(0.5, -0.12, "ε", ha="center", va="top", fontsize=13, color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 4.0), ax.set_ylim(-1.8, 2.0)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_title("MinPts = 4 (Punkt selbst mitgezählt)", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "point_types.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    kmeans_vs_dbscan()
    point_types()
    print("written to", OUT)
