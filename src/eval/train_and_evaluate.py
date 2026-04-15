from __future__ import annotations

import argparse
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
from src.models.dpmm import DPGMM
# from src.models.dpmm_da import DPGMMDriftAware
from src.models.gmm import GMM
from src.utils.generate_simulated_ds import save_dataset_suite, save_simulated_dataset


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _build_model(model_name: str) -> GMMBaseModel:
    if model_name == "gmm":
        return GMM(n_components=2, random_state=42, threshold_quantile=0.01)
    if model_name == "dpmm":
        return DPGMM(
            n_components=8,
            covariance_type="diag",
            random_state=42,
            max_iter=600,
            tol=1e-2,
            n_init=2,
            reg_covar=5e-3,
            alpha=0.3,
            threshold_quantile=0.01,
        )
    # if model_name == "dpmm_da":
    #     return DPGMMDriftAware(
    #         n_components=8,
    #         covariance_type="diag",
    #         random_state=42,
    #         max_iter=600,
    #         tol=1e-2,
    #         n_init=2,
    #         reg_covar=5e-3,
    #         alpha0=0.3,
    #         gamma=0.05,
    #         beta=0.3,
    #         threshold_quantile=0.01,
    #         forgetting_factor=0.1,
    #     )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _model_tag(model: GMMBaseModel) -> str:
    if isinstance(model, GMM):
        return "gmm"
    if isinstance(model, DPGMMDriftAware):
        return "dpmm_da"
    if isinstance(model, DPGMM):
        return "dpmm"
    return model.__class__.__name__.lower()


def train(
    model: GMMBaseModel | None = None,
    model_name: str = "gmm",
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

    model = model or _build_model(model_name)
    model.fit(X_train_scaled)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return model, scaler, data


def test(
    model: GMMBaseModel,
    scaler: StandardScaler,
    data: dict[str, np.ndarray],
    batch_size: int = 1,
    update_normal_only: bool = True,
    freeze_threshold: bool = True,
    metrics_path: str | Path = "experiments/default/metrics.json",
) -> dict[str, float]:
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_train = data["X_train"]

    train_scores = model.score_samples(scaler.transform(X_train))
    threshold_quantile = float(getattr(model, "threshold_quantile", 0.01))
    fixed_threshold = float(np.quantile(train_scores, threshold_quantile))

    all_scores: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for i in range(0, len(X_test), batch_size):
        batch = X_test[i : i + batch_size]
        batch_scaled = scaler.transform(batch)
        batch_scores = model.score_samples(batch_scaled)
        threshold = fixed_threshold if freeze_threshold else None
        batch_preds = model.predict_anomaly(batch_scaled, threshold=threshold).astype(int)
        all_scores.append(batch_scores)
        all_preds.append(batch_preds)

        update_batch = batch_scaled
        if update_normal_only:
            update_batch = batch_scaled[batch_preds == 0]

        min_update_samples = 2
        if isinstance(model, DPGMM):
            min_update_samples = max(2, int(model.n_components))

        if len(update_batch) >= min_update_samples:
            model.partial_fit(update_batch)

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
        "update_normal_only": bool(update_normal_only),
        "freeze_threshold": bool(freeze_threshold),
        "threshold_quantile": threshold_quantile,
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def run_pipeline() -> dict[str, Any]:
    return run_model_pipeline(model_name="gmm", batch_size=1)


def run_model_pipeline(model_name: str = "gmm", batch_size: int = 1) -> dict[str, Any]:
    save_dataset_suite()
    dataset_dir = Path("data/simulated")
    exp_root = Path("experiments") / model_name
    exp_root.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, float]] = {}
    for dataset_path in sorted(dataset_dir.glob("simulated_drift_*.npz")):
        exp_dir = exp_root / dataset_path.stem
        model, scaler, data = train(
            dataset_path=dataset_path,
            save_dir=exp_dir,
            model_name=model_name,
        )
        metrics = test(
            model=model,
            scaler=scaler,
            data=data,
            batch_size=batch_size,
            metrics_path=exp_dir / "metrics.json",
        )
        all_metrics[dataset_path.stem] = metrics
        print(
            f"{model_name} | {dataset_path.stem}: "
            f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )

    summary_path = exp_root / "summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return {"metrics": all_metrics}


def run_all_models(batch_size: int = 1) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for model_name in ("gmm", "dpmm", "dpmm_da"):
        combined[model_name] = run_model_pipeline(model_name=model_name, batch_size=batch_size)

    summary_path = Path("experiments") / "summary_all_models.json"
    summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate GMM family models")
    parser.add_argument(
        "--model",
        default="gmm",
        choices=["gmm", "dpmm", "dpmm_da", "all"],
        help="Model to run",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Online evaluation batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.model == "all":
        run_all_models(batch_size=args.batch_size)
    else:
        run_model_pipeline(model_name=args.model, batch_size=args.batch_size)
