"""Write the sampled texts to one JSONL file for the Colab notebook.

Each line holds one corpus: its texts in sample order and the cache file name
that run_experiments.py will look for. Colab therefore needs neither the
repository nor the full course data, only this file.

    python scripts/export_for_colab.py  ->  cache/colab_input.jsonl
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from textclust.data import load_corpus  # noqa: E402
from textclust.represent import cache_key  # noqa: E402

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
model = cfg["embedding"]["model"]
out = Path(cfg["cache_dir"]) / "colab_input.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)

with open(out, "w", encoding="utf-8") as f:
    for name in cfg["corpora"]:
        corpus, _ = load_corpus(cfg["data_dir"], name, cfg["sample_size"], cfg["seed"])
        cache_file = f"{model.replace('/', '__')}_{cache_key(corpus.texts, model)}.npy"
        f.write(json.dumps({"corpus": name, "cache_file": cache_file, "model": model,
                            "prefix": cfg["embedding"]["prefix"],
                            "max_seq_length": cfg["embedding"]["max_seq_length"],
                            "texts": corpus.texts}, ensure_ascii=False) + "\n")
        print(name, len(corpus.texts), cache_file)
print("written", out, f"{out.stat().st_size / 1e6:.1f} MB")
