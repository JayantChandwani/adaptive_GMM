"""NSL-KDD **baseline** experiments: BIC sweep and per-K sklearn GMM.

Data protocol (see ``run_nsl_kdd`` and ``bic_sweep``):

- Load ``data/nsl_kdd/nsl_kdd.npz`` with ``X_train``, ``y_train``, ``X_test``, ``y_test``.
- **Scaling:** ``StandardScaler`` fit only on **normal** training rows
  (``X_train[y_train == 0]``).
- **Training:** A ``GMM`` is fit on that scaled normal data only (unsupervised density
  of normal traffic).
- **Testing:** ``X_test`` is transformed with the same scaler; anomalies are low-density
  vs a threshold from training (no ``partial_fit`` on test for this baseline).

For **DPGMM** (or GMM with the online simulated pipeline), use
``src.eval.train_and_evaluate`` instead, e.g. ``--nsl --model dpmm`` — see that module’s
``run_nsl_kdd_dpmm``, which mirrors the same NSL preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.models.basemodel import GMMBaseModel
from src.models.gmm import GMM
from src.utils.generate_simulated_ds import save_dataset_suite, save_simulated_dataset


def bic_sweep(
    dataset_path: str | Path = "data/nsl_kdd/nsl_kdd.npz",
    k_values: list[int] | None = None,
    plot_path: str | Path = "experiments/nsl_kdd/bic_sweep.png",
) -> dict[int, float]:
    """Fit GMMs at several K values on normal-only training data and plot BIC scores."""
    from src.eval.plots import plot_bic

    k_values = k_values or [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
    data = _load_npz(Path(dataset_path))

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_normal = X_train[y_train == 0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    bic_scores: dict[int, float] = {}
    for k in k_values:
        print(f"  Fitting K={k}...", flush=True)
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, max_iter=200)
        gm.fit(X_scaled)
        bic_scores[k] = gm.bic(X_scaled)
        print(f"    BIC={bic_scores[k]:.1f}")

    plot_bic(bic_scores, plot_path=plot_path)
    return bic_scores


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def train(
    model: GMMBaseModel | None = None,
    dataset_path: str | Path = "data/simulated/simulated_gmm_drift.npz",
    save_dir: str | Path = "experiments/default",
) -> tuple[GMMBaseModel, StandardScaler, dict[str, np.ndarray]]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        save_simulated_dataset(output_path=dataset_path)

    data = _load_npz(dataset_path)
    X_train = data["X_train"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = model or GMM(n_components=2, random_state=42, threshold_quantile=0.01)
    model.fit(X_train_scaled)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_state_path = save_dir / "gmm_state.npz"
    state = model.get_state()
    np.savez(model_state_path, **state)

    return model, scaler, data

#this fucntion batches and parital fits the model on the test data. not used. 
def test(
    model: GMMBaseModel,
    scaler: StandardScaler,
    data: dict[str, np.ndarray],
    batch_size: int = 128,
    metrics_path: str | Path = "experiments/default/metrics.json",
) -> dict[str, float]:
    X_test = data["X_test"]
    y_test = data["y_test"]

    all_scores: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for i in range(0, len(X_test), batch_size):
        batch = X_test[i : i + batch_size]
        batch_scaled = scaler.transform(batch)
        batch_scores = model.score_samples(batch_scaled)
        batch_preds = model.predict_anomaly(batch_scaled).astype(int)
        all_scores.append(batch_scores)
        all_preds.append(batch_preds)
        model.partial_fit(batch_scaled)

    scores = np.concatenate(all_scores)
    y_pred = np.concatenate(all_preds)

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, -scores)),
        "pr_auc": float(average_precision_score(y_test, -scores)),
        "n_test": int(len(y_test)),
        "n_anomaly": int(y_test.sum()),
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def run_nsl_kdd(
    k_values: list[int] | None = None,
    dataset_path: str | Path = "data/nsl_kdd/nsl_kdd.npz",
    exp_root: str | Path = "experiments/nsl_kdd",
) -> dict[str, dict[str, float]]:
    """Train and test baseline GMM on NSL-KDD for each K in k_values."""
    k_values = k_values or [8, 12]
    data = _load_npz(Path(dataset_path))

    X_train, y_train = data["X_train"], data["y_train"]

    # train on normal traffic only
    X_normal = X_train[y_train == 0]
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    exp_root = Path(exp_root)
    all_metrics: dict[str, dict[str, float]] = {}

    for k in k_values:
        print(f"\nTraining GMM with K={k}...")
        model = GMM(n_components=k, random_state=42, threshold_quantile=0.01)
        model.fit(X_normal_scaled)

        exp_dir = exp_root / f"k{k}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        np.savez(exp_dir / "gmm_state.npz", **model.get_state())

        # baseline: score all test samples with frozen model, no partial_fit
        X_test = data["X_test"]
        y_test = data["y_test"]
        X_test_scaled = scaler.transform(X_test)
        scores = model.score_samples(X_test_scaled)
        y_pred = model.predict_anomaly(X_test_scaled).astype(int)

        metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, -scores)),
            "pr_auc": float(average_precision_score(y_test, -scores)),
            "n_test": int(len(y_test)),
            "n_anomaly": int(y_test.sum()),
        }

        metrics_path = exp_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez(exp_dir / "scores.npz", scores=scores, y_true=y_test, y_pred=y_pred)

        all_metrics[f"k{k}"] = metrics
        print(f"  K={k}: f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}")

    summary_path = exp_root / "summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")

    # generate per-K plots and a combined comparison
    from src.eval.plots import generate_all_plots
    for k in k_values:
        exp_dir = exp_root / f"k{k}"
        npz = np.load(exp_dir / "scores.npz")
        generate_all_plots(
            y_true=npz["y_true"],
            scores=npz["scores"],
            y_pred=npz["y_pred"],
            all_metrics={f"K={k}": all_metrics[f"k{k}"]},
            label=f"GMM K={k}",
            out_dir=exp_dir,
        )

    from src.eval.plots import plot_metrics_bar
    plot_metrics_bar(
        {k: v for k, v in all_metrics.items()},
        plot_path=exp_root / "metrics_comparison.png",
    )

    return all_metrics


def run_pipeline() -> dict[str, Any]:
    save_dataset_suite()
    dataset_dir = Path("data/simulated")
    exp_root = Path("experiments")
    exp_root.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, float]] = {}
    for dataset_path in sorted(dataset_dir.glob("simulated_drift_*.npz")):
        exp_dir = exp_root / dataset_path.stem
        model, scaler, data = train(dataset_path=dataset_path, save_dir=exp_dir)
        metrics = test(
            model=model,
            scaler=scaler,
            data=data,
            batch_size=128,
            metrics_path=exp_dir / "metrics.json",
        )
        all_metrics[dataset_path.stem] = metrics
        print(f"{dataset_path.stem}: f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}")

    summary_path = exp_root / "summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return {"metrics": all_metrics}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bic":
        bic_sweep()
    elif len(sys.argv) > 1 and sys.argv[1] == "nsl":
        run_nsl_kdd()
    else:
        run_pipeline()