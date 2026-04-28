
"""Plot simulated-drift metrics for a single model.

This utility collects `metrics.json` found under
`<exp_root>/<model>/simulated_drift_*` directories and creates:

- a line plot of each metric versus drift level
- an average metrics bar plot across the simulated drift datasets

Outputs are saved under `experiments/<model>/plots/` by default.

Example:
    python -m src.utils.plot_single_model --model gmm

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


# Slide-inspired color palette
PALETTE = {
    "navy": "#0A3B5C",
    "teal": "#13D19A",
    "cream": "#F4EFD9",
    "light_blue": "#D6E9EE",
    "accent": "#2AB3C8",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_drift_level(dataset_name: str) -> float:
    return float(dataset_name.split("simulated_drift_")[-1])


def _collect_simulated_summaries(exp_root: Path, model: str) -> Dict[str, Dict[str, float]]:
    model_dir = exp_root / model
    summaries: Dict[str, Dict[str, float]] = {}
    if not model_dir.exists():
        return summaries

    for child in sorted(model_dir.iterdir()):
        if child.is_dir() and child.name.startswith("simulated_drift_"):
            mpath = child / "metrics.json"
            if mpath.exists():
                try:
                    summaries[child.name] = _load_json(mpath)
                except Exception:
                    continue
    return summaries


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_drift(simulated: Dict[str, Dict[str, float]], metric_key: str, out_path: Path) -> None:
    points = []
    for ds_name, metrics in simulated.items():
        if metric_key in metrics:
            points.append((_parse_drift_level(ds_name), float(metrics[metric_key])))

    if not points:
        return

    points.sort(key=lambda x: x[0])
    drifts = [p[0] for p in points]
    values = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(drifts, values, marker="o", linewidth=2, color=PALETTE["teal"])
    ax.set_xlabel("Drift Level")
    ax.set_ylabel(metric_key.upper().replace("_", "-"))
    ax.set_title(f"{metric_key.upper()} vs Drift")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, out_path)


def plot_average_metrics_bar(simulated: Dict[str, Dict[str, float]], metric_keys: list[str], out_path: Path) -> None:
    means = []
    for k in metric_keys:
        vals = [float(m[k]) for m in simulated.values() if k in m]
        means.append(float(np.mean(vals)) if vals else np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [PALETTE["teal"], PALETTE["accent"], PALETTE["navy"], PALETTE["cream"], PALETTE["light_blue"]]
    ax.bar(range(len(metric_keys)), means, color=colors, alpha=0.92)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([k.upper().replace("_", "-") for k in metric_keys])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Average Metrics Across Simulated Drift Datasets")
    ax.grid(True, axis="y", alpha=0.25)

    _save_fig(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model name folder under experiments (e.g. gmm, dpmm, dpmm_da)")
    p.add_argument("--exp-root", default=Path("experiments"), type=Path, help="Root experiments folder")
    p.add_argument("--out-dir", default=None, type=Path, help="Optional output dir override")
    args = p.parse_args()

    simulated = _collect_simulated_summaries(args.exp_root, args.model)
    if not simulated:
        raise SystemExit(f"No simulated_drift_* datasets found for model {args.model} under {args.exp_root}")

    out_root = args.out_dir if args.out_dir else args.exp_root / args.model / "plots"
    out_root.mkdir(parents=True, exist_ok=True)

    metric_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

    # Per-metric vs drift plots
    for metric in metric_keys:
        plot_metric_vs_drift(simulated, metric, out_root / f"{metric}_vs_drift.png")

    # Average bar
    plot_average_metrics_bar(simulated, metric_keys, out_root / "average_metrics.png")

    # Save summary for convenience
    summary_path = args.exp_root / args.model / "simulated_summary.json"
    summary_path.write_text(json.dumps(simulated, indent=2), encoding="utf-8")
    print(f"Saved simulated plots to {out_root} and summary to {summary_path}")


if __name__ == "__main__":
    main()
