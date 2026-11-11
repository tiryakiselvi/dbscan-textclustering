"""Figures for the talk. Every function writes one file and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import silhouette_samples  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

INK = "#2b2b2b"
ACCENT = "#1f6feb"
SECOND = "#d1495b"
GREY = "#9a9a9a"


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def k_distance_plot(kdist: np.ndarray, eps: float, k: int, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(np.arange(len(kdist)), kdist, color=INK, lw=1.2)
    ax.axhline(eps, color=ACCENT, ls="--", lw=1)
    ax.text(len(kdist) * 0.02, eps, f"  Knick: ε = {eps:.3f}", color=ACCENT, va="bottom")
    ax.set_xlabel("Punkte, sortiert nach k-Distanz")
    ax.set_ylabel(f"Kosinusdistanz zum {k}. Nachbarn")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def reachability_plot(reach: np.ndarray, labels_ordered: np.ndarray, title: str, path: Path) -> Path:
    reach = np.where(np.isinf(reach), np.nan, reach)
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = np.where(labels_ordered == -1, GREY, ACCENT)
    ax.bar(np.arange(len(reach)), reach, width=1.0, color=colors, linewidth=0)
    ax.set_xlabel("Punkte in OPTICS-Reihenfolge")
    ax.set_ylabel("Erreichbarkeitsdistanz")
    ax.set_title(title + "  (grau = Rauschen)", loc="left")
    return _save(fig, path)


def eps_sweep_plot(sweep: pd.DataFrame, title: str, path: Path, label_key: str) -> Path:
    """Noise ratio, silhouette (clustered points) and AMI (all points) over eps."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(sweep["eps"], sweep["noise_ratio"], color=GREY, lw=1.5, label="Noise-Anteil")
    ax.plot(sweep["eps"], sweep["silhouette_clustered"], color=SECOND, lw=1.5,
            label="Silhouette (nur Clusterpunkte)")
    ax.plot(sweep["eps"], sweep[f"ami_all__{label_key}"], color=ACCENT, lw=1.5,
            label=f"AMI gegen {label_key} (alle Texte)")
    ax.set_xlabel("ε (Kosinusdistanz)")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    return _save(fig, path)


def contingency_heatmap(table: pd.DataFrame, title: str, path: Path, max_rows: int = 25) -> Path:
    """Row-normalised cluster x class table; rows sorted by size."""
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index].head(max_rows)
    share = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(0.45 * table.shape[1] + 3, 0.32 * table.shape[0] + 1.5))
    im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(table.shape[0]), [f"{i} (n={n})" for i, n in zip(table.index, table.sum(axis=1))],
                  fontsize=8)
    ax.set_xlabel("Goldklasse")
    ax.set_ylabel("Cluster")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Anteil im Cluster")
    return _save(fig, path)


def silhouette_plot(D: np.ndarray, labels: np.ndarray, title: str, path: Path, max_clusters: int = 15) -> Path:
    """Silhouette values of clustered points (noise excluded), largest clusters first."""
    mask = labels != -1
    lc = labels[mask]
    values = silhouette_samples(D[np.ix_(mask, mask)], lc, metric="precomputed")
    order = pd.Series(lc).value_counts().index[:max_clusters]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    y = 0
    for c in order:
        v = np.sort(values[lc == c])
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, color=ACCENT, alpha=0.7, linewidth=0)
        ax.text(-0.05, y + len(v) / 2, str(c), ha="right", va="center", fontsize=7)
        y += len(v) + 10
    ax.axvline(values.mean(), color=SECOND, ls="--", lw=1)
    ax.set_xlabel("Silhouette-Wert s(i)")
    ax.set_yticks([])
    ax.set_title(title, loc="left")
    return _save(fig, path)
