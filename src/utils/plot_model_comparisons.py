from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_simulated_summaries(exp_root: Path, models: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        summary_path = exp_root / model / "summary.json"
        if summary_path.exists():
            data[model] = _load_json(summary_path)
    return data


def _parse_drift_level(dataset_name: str) -> float:
    # Expected format: simulated_drift_<float>
    return float(dataset_name.split("simulated_drift_")[-1])


def _plot_simulated_metric_vs_drift(
    simulated: dict[str, dict[str, dict[str, float]]],
    metric_key: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, summary in simulated.items():
        points: list[tuple[float, float]] = []
        for dataset_name, metrics in summary.items():
            if dataset_name.startswith("simulated_drift_") and metric_key in metrics:
                points.append((_parse_drift_level(dataset_name), float(metrics[metric_key])))

        if not points:
            continue

        points.sort(key=lambda x: x[0])
        drifts = [p[0] for p in points]
        values = [p[1] for p in points]
        ax.plot(drifts, values, marker="o", linewidth=2, label=model_name)

    ax.set_xlabel("Drift Level")
    ax.set_ylabel(metric_key.upper().replace("_", "-"))
    ax.set_title(f"Simulated Drift Comparison ({metric_key})")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_simulated_average_bars(
    simulated: dict[str, dict[str, dict[str, float]]],
    metric_keys: list[str],
    out_path: Path,
) -> None:
    model_names = list(simulated.keys())
    if not model_names:
        return

    values_by_model: dict[str, list[float]] = {}
    for model_name, summary in simulated.items():
        per_metric_means: list[float] = []
        for metric_key in metric_keys:
            values = [
                float(m[metric_key])
                for ds_name, m in summary.items()
                if ds_name.startswith("simulated_drift_") and metric_key in m
            ]
            per_metric_means.append(float(np.mean(values)) if values else np.nan)
        values_by_model[model_name] = per_metric_means

    x = np.arange(len(metric_keys))
    width = 0.8 / max(1, len(model_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        ax.bar(x + i * width, values_by_model[model_name], width=width, label=model_name, alpha=0.9)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in metric_keys])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Average Metrics Across Simulated Drift Datasets")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pick_nsl_config_for_model(
    nsl_summary: dict[str, dict[str, float]],
    model_name: str,
    prefer_online: bool,
) -> tuple[str, dict[str, float]] | None:
    candidates = [(name, metrics) for name, metrics in nsl_summary.items() if name.startswith(f"{model_name}_")]
    if not candidates:
        return None

    if prefer_online:
        online_candidates = [c for c in candidates if "_online_" in c[0]]
        if online_candidates:
            candidates = online_candidates

    # Deterministic choice: pick lexicographically last config.
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def _plot_nsl_metric_bars(
    nsl_summary: dict[str, dict[str, float]],
    models: list[str],
    metric_keys: list[str],
    out_path: Path,
    prefer_online: bool,
) -> None:
    chosen: list[tuple[str, str, dict[str, float]]] = []
    for model in models:
        picked = _pick_nsl_config_for_model(nsl_summary, model, prefer_online=prefer_online)
        if picked is not None:
            config_name, metrics = picked
            chosen.append((model, config_name, metrics))

    if not chosen:
        return

    x = np.arange(len(metric_keys))
    width = 0.8 / len(chosen)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model_name, _, metrics) in enumerate(chosen):
        vals = [float(metrics[k]) if k in metrics else np.nan for k in metric_keys]
        ax.bar(x + i * width, vals, width=width, label=model_name, alpha=0.9)

    ax.set_xticks(x + width * (len(chosen) - 1) / 2)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in metric_keys])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    title_suffix = "online-preferred" if prefer_online else "latest-per-model"
    ax.set_title(f"NSL-KDD Metrics Comparison ({title_suffix})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save the exact configs used for reproducibility.
    meta = {
        model_name: {"config": config_name, "metrics": metrics}
        for model_name, config_name, metrics in chosen
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def generate_model_comparison_plots(
    exp_root: Path = Path("experiments"),
    out_dir: Path = Path("experiments/comparisons"),
    models: list[str] | None = None,
    prefer_online_nsl: bool = True,
) -> None:
    models = models or ["gmm", "dpmm", "dpmm_da"]
    metric_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

    simulated = _load_simulated_summaries(exp_root, models)
    if simulated:
        _plot_simulated_average_bars(
            simulated=simulated,
            metric_keys=metric_keys,
            out_path=out_dir / "simulated_metrics_average.png",
        )
        for metric_key in metric_keys:
            _plot_simulated_metric_vs_drift(
                simulated=simulated,
                metric_key=metric_key,
                out_path=out_dir / f"simulated_{metric_key}_vs_drift.png",
            )

    nsl_summary_path = exp_root / "nsl_kdd_dpmm" / "summary.json"
    if nsl_summary_path.exists():
        nsl_summary = _load_json(nsl_summary_path)
        _plot_nsl_metric_bars(
            nsl_summary=nsl_summary,
            models=models,
            metric_keys=metric_keys,
            out_path=out_dir / "nsl_metrics_comparison.png",
            prefer_online=prefer_online_nsl,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison plots for gmm/dpmm/dpmm_da")
    parser.add_argument("--exp-root", type=str, default="experiments", help="Experiments root directory")
    parser.add_argument("--out-dir", type=str, default="experiments/comparisons", help="Output directory")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gmm", "dpmm", "dpmm_da"],
        help="Model names to include in comparisons",
    )
    parser.add_argument(
        "--nsl-latest",
        action="store_true",
        help="For NSL-KDD comparison, pick latest config per model instead of preferring online configs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_model_comparison_plots(
        exp_root=Path(args.exp_root),
        out_dir=Path(args.out_dir),
        models=args.models,
        prefer_online_nsl=not args.nsl_latest,
    )


if __name__ == "__main__":
    main()
