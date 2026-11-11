"""Run the full experiment grid and write results, tables and figures.

    python run_experiments.py                         # all corpora from config.yaml
    python run_experiments.py --corpora letters_small --sample-size 0   # smoke test

Gold labels are used only after clustering, for evaluation. Eps is chosen
from the k-distance graph; the eps sweep with ARI/AMI is a diagnostic analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from textclust import cluster, evaluate, plots
from textclust.data import load_corpus
from textclust.represent import load_cached_embeddings, reduce, tfidf

log = logging.getLogger("experiments")


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def representations(corpus, cfg: dict) -> tuple[dict, object]:
    """Yield the feature spaces that are available for this corpus."""
    seed, dims = cfg["seed"], cfg["reduction_dims"]
    X_tfidf, vectorizer = tfidf(corpus.texts, **cfg["tfidf"])
    reps = {
        "tfidf": {"X": X_tfidf, "info": {"n_features": X_tfidf.shape[1]}},
        f"tfidf_svd{dims}": {"X": reduce(X_tfidf, dims, seed), "info": {"from": "tfidf"}},
    }
    emb_cfg = cfg["embedding"]
    E = load_cached_embeddings(corpus.texts, emb_cfg["model"], cfg["cache_dir"])
    if E is None:
        log.warning("%s: no cached embeddings for %s, run notebooks/embeddings_colab.ipynb first",
                    corpus.spec.name, emb_cfg["model"])
    else:
        reps["emb"] = {"X": E, "info": {"model": emb_cfg["model"]}}
        reps[f"emb_pca{dims}"] = {"X": reduce(E, dims, seed), "info": {"from": "emb"}}
    return reps, vectorizer


def evaluate_result(res, D, corpus, cfg) -> list[dict]:
    rows = []
    intern = evaluate.internal(D, res.labels, cfg["silhouette_sample_size"], cfg["seed"])
    for label_key, y in corpus.labels.items():
        row = {"method": res.method, "label": label_key, "runtime_s": round(res.runtime_s, 3),
               **{f"param_{k}": v for k, v in res.params.items()}, **intern,
               **evaluate.external(y, res.labels)}
        rows.append(row)
    return rows


def top_terms(X_tfidf, vectorizer, labels: np.ndarray, n: int = 8) -> dict:
    terms = np.asarray(vectorizer.get_feature_names_out())
    out = {}
    for c in np.unique(labels):
        if c == -1:
            continue
        centroid = np.asarray(X_tfidf[labels == c].mean(axis=0)).ravel()
        out[int(c)] = terms[np.argsort(centroid)[::-1][:n]].tolist()
    return out


def run_corpus(name: str, cfg: dict, out: Path, fig_dir: Path) -> list[dict]:
    corpus, report = load_corpus(cfg["data_dir"], name, cfg["sample_size"], cfg["seed"])
    log.info("%s: %d texts (seed %d)", name, len(corpus.texts), cfg["seed"])
    (out / "samples").mkdir(parents=True, exist_ok=True)
    np.save(out / "samples" / f"{name}_seed{cfg['seed']}_n{len(corpus.indices)}.npy", corpus.indices)
    (out / "validation").mkdir(exist_ok=True)
    (out / "validation" / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    reps, vectorizer = representations(corpus, cfg)
    rows = []
    for rep_name, rep in reps.items():
        tag = f"{name}/{rep_name}"
        t0 = time.perf_counter()
        D = cluster.distance_matrix(rep["X"])
        log.info("%s: distance matrix %.1fs", tag, time.perf_counter() - t0)

        def step(what):
            log.info("%s: %s done after %.0fs", tag, what, time.perf_counter() - t0)

        def add(results):
            for r in results:
                for row in evaluate_result(r, D, corpus, cfg):
                    rows.append({"corpus": name, "representation": rep_name, **row})

        # DBSCAN with eps from the k-distance knee (no labels involved)
        chosen = {}
        for ms in cfg["dbscan"]["min_samples"]:
            kdist = cluster.k_distances(D, ms)
            eps = cluster.knee(kdist)
            res = cluster.run_dbscan(D, eps, ms)
            res.params["eps_rule"] = "knee"
            add([res])
            chosen[ms] = (res, kdist)

        step("DBSCAN (knee)")
        ms = cfg["dbscan"]["sweep_min_samples"]
        res, kdist = chosen[ms]
        plots.k_distance_plot(kdist, res.params["eps"], ms, f"k-Distanz-Graph · {tag}",
                              fig_dir / name / f"{rep_name}_kdistance.png")

        # eps sweep (diagnostic): H1 noise/eps behaviour, H2 silhouette vs. AMI
        sweep_rows = []
        for eps in cluster.eps_grid(kdist, cfg["dbscan"]["sweep_points"]):
            r = cluster.run_dbscan(D, eps, ms)
            row = {"eps": eps, **evaluate.internal(D, r.labels, cfg["silhouette_sample_size"], cfg["seed"])}
            for label_key, y in corpus.labels.items():
                ext = evaluate.external(y, r.labels)
                row[f"ami_all__{label_key}"] = ext["ami_all"]
                row[f"ari_all__{label_key}"] = ext["ari_all"]
                row[f"ami_clustered__{label_key}"] = ext["ami_clustered"]
            sweep_rows.append(row)
        sweep = pd.DataFrame(sweep_rows)
        (out / "sweeps").mkdir(exist_ok=True)
        sweep.to_csv(out / "sweeps" / f"{name}_{rep_name}.csv", index=False)
        for label_key in corpus.labels:
            plots.eps_sweep_plot(sweep, f"ε-Sweep · {tag} · MinPts = {ms}",
                                 fig_dir / name / f"{rep_name}_sweep_{label_key}.png", label_key)

        step("eps sweep")

        # Second label-free eps rule: the largest eps at which no cluster holds more
        # than half of the texts. The knee often sits right at the point where
        # everything merges into one component, so we report both rules.
        ok = sweep[(sweep["n_clusters"] >= 2) & (sweep["largest_cluster_share"] <= 0.5)]
        candidates = {"knee": res}
        if len(ok):
            alt = cluster.run_dbscan(D, float(ok["eps"].max()), ms)
            alt.params["eps_rule"] = "largest_cluster<=50%"
            add([alt])
            candidates["nogiant"] = alt

        # tables and figures for both DBSCAN results
        (out / "contingency").mkdir(exist_ok=True)
        (out / "terms").mkdir(exist_ok=True)
        for rule, r in candidates.items():
            for label_key, y in corpus.labels.items():
                table = evaluate.contingency(y, r.labels)
                table.to_csv(out / "contingency" / f"{name}_{rep_name}_dbscan-{rule}_{label_key}.csv")
                plots.contingency_heatmap(table, f"DBSCAN ({rule}, ε = {r.params['eps']:.3f}) · {tag} · {label_key}",
                                          fig_dir / name / f"{rep_name}_contingency_{rule}_{label_key}.png")
            if evaluate.internal(D, r.labels)["n_clusters"] >= 2:
                plots.silhouette_plot(D, r.labels, f"Silhouette · DBSCAN ({rule}) · {tag}",
                                      fig_dir / name / f"{rep_name}_silhouette_{rule}.png")
            (out / "terms" / f"{name}_{rep_name}_dbscan-{rule}.json").write_text(
                json.dumps(top_terms(reps["tfidf"]["X"], vectorizer, r.labels), ensure_ascii=False, indent=1))

        step("tables and figures")

        # OPTICS, one fit, several xi
        oc = cfg["optics"]
        optics_results = cluster.run_optics(rep["X"], oc["min_samples"], oc["xi"], oc["min_cluster_size"])
        add(optics_results)
        ref = next(r for r in optics_results if r.params["xi"] == 0.05) if 0.05 in oc["xi"] else optics_results[0]
        plots.reachability_plot(ref.extra["reachability_ordered"], ref.labels[ref.extra["ordering"]],
                                f"OPTICS · {tag} · ξ = {ref.params['xi']}",
                                fig_dir / name / f"{rep_name}_reachability.png")

        step("OPTICS")

        # K-Means reference with oracle k (dense spaces only; full TF-IDF is skipped)
        if not hasattr(rep["X"], "tocsr"):
            for label_key, y in corpus.labels.items():
                km = cluster.run_kmeans(rep["X"], len(np.unique(y)), cfg["seed"])
                km.params["k_rule"] = f"oracle:{label_key}"
                add([km])

        del D
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--corpora", nargs="+")
    parser.add_argument("--sample-size", type=int, help="0 = use all texts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.corpora:
        cfg["corpora"] = args.corpora
    if args.sample_size is not None:
        cfg["sample_size"] = args.sample_size or None

    out, fig_dir = Path(cfg["results_dir"]), Path(cfg["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    run_info = {"config": cfg, "environment": environment()}

    all_rows = []
    for name in cfg["corpora"]:
        t0 = time.perf_counter()
        all_rows += run_corpus(name, cfg, out, fig_dir)
        log.info("%s finished in %.0fs", name, time.perf_counter() - t0)

    tag = "_".join(cfg["corpora"])
    summary = pd.DataFrame(all_rows)
    summary.to_csv(out / f"summary_{tag}.csv", index=False)
    run_info["environment"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / f"run_{tag}.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False))
    log.info("wrote %d result rows to %s", len(summary), out / f"summary_{tag}.csv")


if __name__ == "__main__":
    main()
