# DBSCAN auf Texten

Referatsprojekt im Hauptseminar *Klassifikation und Clustering* (LMU München, CIS, WS 2026/27, Dr. Stefan Langer).
Suheda und Selvinaz Tiryaki · Vortrag am 21.12.2026

## Fragestellung

Unter welchen Bedingungen (Textrepräsentation, Dimensionsreduktion, ε und MinPts) findet DBSCAN in den drei Kurskorpora Cluster, die mit den vorhandenen Klassen übereinstimmen? Und wie verlässlich ist der Silhouette Score, wenn ein großer Teil der Texte als Rauschen aussortiert wird?

| Korpus | Datei | Klassen |
| --- | --- | --- |
| Briefe (Segmente) | `letters/clustering/clustering_data_full.json` | Autor (7), Sprache (8) |
| Huffington Post | `news/full_data/classification_news.jsonl` | Kategorie (34) |
| IMDB | `sentiment/full_data/classification_sentiment.jsonl` | Sentiment (2) |

Verglichen werden DBSCAN und OPTICS, jeweils auf TF-IDF und auf mehrsprachigen Sentence-Embeddings (`intfloat/multilingual-e5-base`), im vollen Raum und nach Reduktion auf 50 Dimensionen. K-Means dient als Vergleichsbasis.

## Aufbau

```
config.yaml               alle Parameter (Seed, Stichprobe, TF-IDF, ε-Regeln, OPTICS)
run_experiments.py        komplette Versuchsreihe → results/, figures/
src/textclust/
    data.py               Laden (JSONL), Prüfung gegen das Datenmanifest, Stichprobe
    represent.py          TF-IDF, SVD/PCA, Embeddings mit Cache
    cluster.py            Distanzmatrix, k-Distanz, Knick, DBSCAN, OPTICS, K-Means
    evaluate.py           interne und externe Maße mit Noise-Behandlung
    plots.py              Abbildungen für die Folien
notebooks/
    embeddings_colab.ipynb  Embeddings auf der Colab-GPU berechnen
scripts/
    theory_figures.py     Illustrationen für den Theorieteil (synthetische 2D-Daten)
    export_for_colab.py   Stichproben als Eingabe für das Colab-Notebook
tests/                    Tests mit synthetischen Daten (pytest)
```

## Installation

Python 3.10 oder neuer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; unter Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Daten

Die Kursdaten sind nicht im Repository. Das Archiv `classification_data.zip` aus dem Kursmaterial in den Ordner `data/` entpacken (Passwort siehe Kursmaterial), sodass z. B. `data/letters/clustering/clustering_data_full.json` existiert. Der Ordner ist in `.gitignore` eingetragen.

Beim Laden werden Größe, Pflichtfelder und Klassenzahl gegen das Datenmanifest geprüft; das Ergebnis steht in `results/validation/`.

## Ablauf

1. Smoke-Test (1.000 Briefe, ca. 10 Sekunden):
   `python run_experiments.py --corpora letters_small --sample-size 0`
2. Embeddings in Colab berechnen: `python scripts/export_for_colab.py` schreibt die Stichproben nach `cache/colab_input.jsonl`; dann `notebooks/embeddings_colab.ipynb` in Colab öffnen, die Datei hochladen und `embeddings.zip` nach `cache/` entpacken
3. Vollständige Versuchsreihe: `python run_experiments.py`

Ohne Embeddings im Cache läuft die Versuchsreihe nur mit TF-IDF und gibt eine Warnung aus.

## Ausgaben

| Pfad | Inhalt |
| --- | --- |
| `results/summary_<korpora>.csv` | eine Zeile pro Lauf und Goldlabel: Parameter, Clusterzahl, Noise-Anteil, Silhouette, ARI, AMI, Homogeneity, Completeness, V-Measure, Laufzeit |
| `results/sweeps/` | DBSCAN über 30 ε-Werte (Noise, Silhouette, ARI/AMI) |
| `results/contingency/` | Kontingenztabellen Cluster × Klasse |
| `results/terms/` | häufigste TF-IDF-Terme pro Cluster |
| `results/samples/` | Indizes der Stichproben |
| `results/run_<korpora>.json` | Konfiguration, Bibliotheksversionen, Hardware, Zeitstempel |
| `figures/` | k-Distanz-Graph, Reachability-Plot, ε-Sweep, Kontingenz, Silhouette-Plot |

## Methodische Entscheidungen

- **Goldlabels nur zur Evaluation.** ε wird ohne Labels gewählt. Der ε-Sweep mit ARI/AMI ist eine diagnostische Analyse und wird so bezeichnet.
- **Zwei labelfreie ε-Regeln:** (1) Knick im k-Distanz-Graphen (Ester et al. 1996), (2) größtes ε, bei dem kein Cluster mehr als 50 % der Texte enthält. Regel 2 ist unsere eigene Setzung; sie fängt den Fall ab, dass der Knick genau dort liegt, wo alles zu einem Cluster verschmilzt.
- **Kosinusdistanz, vorab berechnet.** Bei 10.000 Texten ist die Distanzmatrix etwa 0,8 GB groß. DBSCAN, OPTICS und Silhouette nutzen so exakt dieselben Distanzen.
- **Rauschen (Label −1):** Jedes Maß wird zweimal berichtet: über alle Texte (Rauschen als eigene Gruppe) und nur über geclusterte Texte zusammen mit der Abdeckung. Der Silhouette Score ist für Rauschpunkte nicht definiert; `silhouette_clustered` lässt sie weg, `silhouette_all` zählt sie als Gruppe.
- **MinPts:** In scikit-learn zählt `min_samples` den Punkt selbst mit. Die k-Distanz wird deshalb mit k = `min_samples` inklusive des Punktes berechnet (siehe Test `test_k_distance_matches_dbscan_core_definition`).
- **K-Means** bekommt k = Zahl der Goldklassen. Das ist ein „Oracle-k“ und nur als Vergleichsbasis gedacht.
- **Stichproben** sind zufällig und nicht stratifiziert, weil Stratifizieren die Labels benutzen würde.

## Literatur

- Ester, Kriegel, Sander, Xu (1996): A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. KDD-96.
- Ankerst, Breunig, Kriegel, Sander (1999): OPTICS: Ordering Points To Identify the Clustering Structure. SIGMOD.
- Schubert, Sander, Ester, Kriegel, Xu (2017): DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN. ACM TODS 42(3).
- Rousseeuw (1987): Silhouettes. Journal of Computational and Applied Mathematics 20.
- Vinh, Epps, Bailey (2010): Information Theoretic Measures for Clusterings Comparison. JMLR 11.
- Reimers, Gurevych (2019): Sentence-BERT. EMNLP-IJCNLP.
- Enevoldsen et al. (2025): MMTEB: Massive Multilingual Text Embedding Benchmark. ICLR.
