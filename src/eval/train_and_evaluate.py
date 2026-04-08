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
from sklearn.preprocessing import StandardScaler

from src.models.basemodel import GMMBaseModel
from src.models.gmm import GMM
from src.utils.generate_simulated_ds import save_dataset_suite, save_simulated_dataset


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
    run_pipeline()