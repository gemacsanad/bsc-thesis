"""Export the W&B training logs that the thesis figures are drawn from.

Only the run-11 runs (the final model and its ablations) are exported: the reader, the joint
fine-tune and the eight GNN-grid variants.

Each run goes to diagrams/data/<run name>/:

    history.csv      every logged scalar at every step (scan_history, not the downsampled
                     history())
    histograms.json  histogram metrics (gate/per_layer) per step, as W&B's bins and counts;
                     only written if the run logged any
    config.json      the run's hyperparameters
    summary.json     the run's final values
    meta.json        run id, URL, state, timestamps

Run from the repository root, with WANDB_KEY in .env_local:
    python diagrams/export_wandb.py
"""
import json
import os

import pandas as pd
import wandb
from dotenv import dotenv_values

PROJECT = "dytag-adapter"
JOB = "14541449"   # the SLURM job of run11_all.sh
RUNS = [f"run11-reader-{JOB}", f"run11-joint-{JOB}"] + [
    f"run11-gnn-{conv}-l{layers}-{loss}-{JOB}"
    for conv in ("gatv2", "sage") for layers in (1, 2) for loss in ("cos", "cosnce")
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "diagrams", "data")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    os.environ["WANDB_API_KEY"] = dotenv_values(os.path.join(ROOT, ".env_local"))["WANDB_KEY"]
    api = wandb.Api()
    path = f"{api.default_entity}/{PROJECT}"
    by_name = {r.name: r for r in api.runs(path)}
    missing = [n for n in RUNS if n not in by_name]
    if missing:
        raise SystemExit(f"not found in {path}: {missing}")

    for name in RUNS:
        run = by_name[name]
        out = os.path.join(OUT, name)
        os.makedirs(out, exist_ok=True)

        history = pd.DataFrame(list(run.scan_history()))
        scalar = [c for c in history.columns
                  if not history[c].map(lambda v: isinstance(v, (dict, list))).any()]
        nonscalar = sorted(set(history.columns) - set(scalar))
        if nonscalar:
            hist = history[["_step"] + nonscalar].dropna(how="all", subset=nonscalar)
            write_json(os.path.join(out, "histograms.json"),
                       [{k: v for k, v in row.items() if not (isinstance(v, float) and v != v)}
                        for row in hist.to_dict("records")])
        history = history[scalar].sort_values("_step")
        front = [c for c in ("_step", "_runtime", "_timestamp") if c in history.columns]
        history = history[front + sorted(c for c in history.columns if c not in front)]
        history.to_csv(os.path.join(out, "history.csv"), index=False)

        write_json(os.path.join(out, "config.json"), dict(run.config))
        write_json(os.path.join(out, "summary.json"),
                   {k: v for k, v in run.summary._json_dict.items()
                    if not isinstance(v, dict)})
        write_json(os.path.join(out, "meta.json"),
                   {"id": run.id, "name": run.name, "url": run.url, "state": run.state,
                    "created_at": run.created_at, "project": path})
        print(f"{name}: {len(history)} rows, {len(history.columns)} columns"
              + (f", histograms {nonscalar}" if nonscalar else ""))


if __name__ == "__main__":
    main()
