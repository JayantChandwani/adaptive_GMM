from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
)


def plot_bic(
    bic_scores: dict[int, float],
    plot_path: str | Path = "experiments/nsl_kdd/bic_sweep.png",
) -> None:
    k_values = list(bic_scores.keys())
    scores = list(bic_scores.values())
    best_k = min(bic_scores, key=bic_scores.__getitem__)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, scores, marker="o", linewidth=2, color="steelblue", label="BIC")
    ax.axvline(best_k, color="tomato", linestyle="--", label=f"Best K={best_k}")
    ax.scatter([best_k], [bic_scores[best_k]], color="tomato", zorder=5)

    ax.set_xlabel("Number of Components (K)", fontsize=12)
    ax.set_ylabel("BIC Score (lower is better)", fontsize=12)
    ax.set_title("BIC Sweep — NSL-KDD (Normal Traffic Only)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"BIC plot saved to {plot_path}")
    print(f"Lowest BIC at K={best_k} (BIC={bic_scores[best_k]:.1f})")


def plot_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str = "GMM",
    plot_path: str | Path = "experiments/nsl_kdd/roc_curve.png",
) -> None:
    # scores are log-likelihoods: lower = more anomalous, so negate for ROC
    fpr, tpr, _ = roc_curve(y_true, -scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, color="steelblue", label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — NSL-KDD", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved to {plot_path}")


def plot_pr(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str = "GMM",
    plot_path: str | Path = "experiments/nsl_kdd/pr_curve.png",
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, -scores)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, linewidth=2, color="darkorange", label=f"{label} (AUC={pr_auc:.3f})")
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1, label=f"Random (={baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — NSL-KDD", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PR curve saved to {plot_path}")


def plot_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str = "GMM",
    plot_path: str | Path = "experiments/nsl_kdd/score_distribution.png",
) -> None:
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    # clip to 1st–99th percentile of all scores to avoid extreme outliers dominating
    lo = float(np.percentile(scores, 1))
    hi = float(np.percentile(scores, 99))
    normal_scores = np.clip(normal_scores, lo, hi)
    anomaly_scores = np.clip(anomaly_scores, lo, hi)
    bins = np.linspace(lo, hi, 80)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(normal_scores, bins=bins, alpha=0.6, color="steelblue", label="Normal", density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, color="tomato", label="Anomaly", density=True)

    ax.set_xlabel("Log-Likelihood Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution — {label} on NSL-KDD", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Score distribution saved to {plot_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "GMM",
    plot_path: str | Path = "experiments/nsl_kdd/confusion_matrix.png",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {label}", fontsize=13)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

    fig.tight_layout()
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {plot_path}")


def plot_metrics_bar(
    all_metrics: dict[str, dict[str, float]],
    plot_path: str | Path = "experiments/nsl_kdd/metrics_comparison.png",
) -> None:
    metric_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = list(all_metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(labels)
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, metrics) in enumerate(all_metrics.items()):
        values = [metrics[m] for m in metric_keys]
        ax.bar(x + i * width, values, width, label=label, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Metrics Comparison — NSL-KDD Baseline GMM", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics comparison saved to {plot_path}")


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str = "GMM",
    plot_path: str | Path = "experiments/nsl_kdd/threshold_sensitivity.png",
) -> None:
    from sklearn.metrics import f1_score as f1

    quantiles = np.linspace(0.001, 0.15, 60)
    f1_scores = []
    for q in quantiles:
        thresh = float(np.quantile(scores, q))
        y_pred = (scores < thresh).astype(int)
        f1_scores.append(f1(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(quantiles, f1_scores, linewidth=2, color="steelblue")
    ax.axvline(0.01, color="tomato", linestyle="--", label="Default threshold (q=0.01)")
    ax.set_xlabel("Threshold Quantile", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"Threshold Sensitivity — {label} on NSL-KDD", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Threshold sensitivity saved to {plot_path}")


def generate_all_plots(
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    all_metrics: dict[str, dict[str, float]],
    label: str = "GMM",
    out_dir: str | Path = "experiments/nsl_kdd",
) -> None:
    out_dir = Path(out_dir)
    plot_roc(y_true, scores, label=label, plot_path=out_dir / "roc_curve.png")
    plot_pr(y_true, scores, label=label, plot_path=out_dir / "pr_curve.png")
    plot_score_distribution(y_true, scores, label=label, plot_path=out_dir / "score_distribution.png")
    plot_confusion_matrix(y_true, y_pred, label=label, plot_path=out_dir / "confusion_matrix.png")
    plot_threshold_sensitivity(y_true, scores, label=label, plot_path=out_dir / "threshold_sensitivity.png")
    plot_metrics_bar(all_metrics, plot_path=out_dir / "metrics_comparison.png")
