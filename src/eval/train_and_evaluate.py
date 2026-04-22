"""Simulated-drift and NSL-KDD evaluation for GMM / DPGMM.

NSL-KDD baseline (BIC sweep, per-K GMM) lives in ``train_and_evaluate_baselineGMM.py``.
For DPGMM (or GMM) on NSL-KDD with the **same data protocol** as that baseline, use:

    python -m src.eval.train_and_evaluate --nsl --model dpmm

See ``run_nsl_kdd_dpmm`` docstring for details.
"""
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
from src.models.gmm import GMM
from src.utils.generate_simulated_ds import save_dataset_suite, save_simulated_dataset

# DPGMMDriftAware is a planned model. Import is guarded so the rest of the
# pipeline works while src/models/dpmm_da.py is still a placeholder.
try:
    from src.models.dpmm_da import DPGMMDriftAware  # type: ignore[import-not-found]
except ImportError:
    DPGMMDriftAware = None  # type: ignore[assignment]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _arrival_order(n: int, t: np.ndarray | None) -> np.ndarray:
    """Return indices that sort the stream by time of arrival.

    If ``t`` is provided, sort by it (stable). Otherwise fall back to the
    natural row order, which IS the arrival order for both NSL-KDD (ARFF row
    order) and the simulated stream (index-by-index generation).
    """
    if t is None:
        return np.arange(n)
    t = np.asarray(t)
    if t.shape[0] != n:
        raise ValueError(
            f"timestamp vector length {t.shape[0]} does not match data length {n}"
        )
    return np.argsort(t, kind="stable")


def _sort_by_arrival(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a copy of ``data`` with X_test/y_test sorted by t_test.

    Train data is left untouched (the model is fit in one shot, order does not
    matter), but t_train is preserved if present.
    """
    out = dict(data)
    X_test = out["X_test"]
    y_test = out["y_test"]
    t_test = out.get("t_test")
    order = _arrival_order(len(X_test), t_test)
    out["X_test"] = X_test[order]
    out["y_test"] = y_test[order]
    if t_test is not None:
        out["t_test"] = np.asarray(t_test)[order]
    return out


def _build_model(
    model_name: str,
    n_components: int | None = None,
) -> GMMBaseModel:
    if model_name == "gmm":
        k = n_components if n_components is not None else 2
        return GMM(n_components=k, random_state=42, threshold_quantile=0.01)
    if model_name == "dpmm":
        k = n_components if n_components is not None else 12
        return DPGMM(
            n_components=k,
            covariance_type="diag",
            random_state=42,
            max_iter=600,
            tol=1e-2,
            # n_init=1 is correct with warm_start=True; n_init>1 was both
            # ignored after the first fit and a source of convergence-warning
            # noise (each init runs independently up to max_iter).
            n_init=1,
            reg_covar=5e-3,
            alpha=0.3,
            threshold_quantile=0.01,
        )
    if model_name == "dpmm_da":
        if DPGMMDriftAware is None:
            raise NotImplementedError(
                "DPGMMDriftAware is not implemented yet. "
                "Add the class to src/models/dpmm_da.py to enable --model dpmm_da."
            )
        k = n_components if n_components is not None else 8
        return DPGMMDriftAware(
            n_components=k,
            covariance_type="diag",
            random_state=42,
            max_iter=600,
            tol=1e-2,
            n_init=1,
            reg_covar=5e-3,
            alpha0=0.3,
            gamma=0.05,
            beta=0.3,
            threshold_quantile=0.01,
            forgetting_factor=0.1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _model_tag(model: GMMBaseModel) -> str:
    if isinstance(model, GMM):
        return "gmm"
    if DPGMMDriftAware is not None and isinstance(model, DPGMMDriftAware):
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
    # Sort the test stream by arrival time once; train rows are independent
    # i.i.d. samples, so order does not matter for the one-shot fit.
    data = _sort_by_arrival(data)
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
    return_arrays: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray]:
    # Caller is expected to pass arrival-ordered data (use _sort_by_arrival).
    # We do NOT re-sort here so that returned (scores, y_pred) line up 1:1 with
    # the X_test / y_test the caller already holds.
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
        if hasattr(model, "n_components"):
            min_update_samples = max(2, int(getattr(model, "n_components", 2)))

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

    if return_arrays:
        return metrics, scores, y_pred
    return metrics


def run_nsl_kdd_dpmm(
    dataset_path: str | Path = "data/nsl_kdd/nsl_kdd.npz",
    exp_root: str | Path = "experiments/nsl_kdd_dpmm",
    model_name: str = "dpmm",
    batch_size: int = 32,
    online: bool = False,
    freeze_threshold: bool = True,
    update_normal_only: bool = True,
    n_components: int | None = None,
) -> dict[str, Any]:
    """Evaluate GMM or DPGMM on NSL-KDD using the same protocol as ``run_nsl_kdd`` in
    ``train_and_evaluate_baselineGMM.py``:

    1. Load ``X_train``, ``y_train``, ``X_test``, ``y_test`` from the processed ``.npz``.
    2. **Scaler:** ``StandardScaler`` is fit on **normal training rows only**
       (``X_train[y_train == 0]``), matching the baseline (no test leakage).
    3. **Fit:** The model is fit only on scaled normal training data (density model of
       "normal" traffic).
    4. **Frozen eval (default):** Score all of ``X_test`` in one pass; no ``partial_fit``.
       Anomaly = low log-density vs threshold learned on training normals (inside
       ``model.fit`` / ``_update_threshold``), same idea as the GMM baseline.
    5. **Online eval (``online=True``):** Stream test in batches; optional
       ``partial_fit`` on predicted-normal points; threshold can stay fixed from step 3
       (``freeze_threshold=True``) via ``test()`` — pass only normal training rows in
       ``data["X_train"]`` so the quantile threshold is not polluted by attack rows.

    Outputs go under ``exp_root / model_name /`` so baseline GMM runs under
    ``experiments/nsl_kdd/`` stay untouched.
    """
    data = _load_npz(Path(dataset_path))
    # Streaming evaluation must process records in time-of-arrival order. We
    # sort once here using t_test (if present) and use the sorted arrays for
    # both online and frozen branches to keep scores/labels/plots aligned.
    data = _sort_by_arrival(data)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    X_normal = X_train[y_train == 0]
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    model = _build_model(model_name, n_components=n_components)
    model.fit(X_normal_scaled)

    exp_root = Path(exp_root)
    effective_k = int(getattr(model, "n_components", 0))
    mode = "online" if online else "frozen"
    update_tag = "normalonly" if update_normal_only else "all"
    threshold_tag = "fixthr" if freeze_threshold else "movingthr"
    config_name = f"{model_name}_{mode}_bs{batch_size}_k{effective_k}_{update_tag}_{threshold_tag}"
    exp_dir = exp_root / config_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {exp_dir}")

    if online:
        eval_data = {
            "X_train": X_normal,
            "X_test": X_test,
            "y_test": y_test,
        }
        metrics, scores, y_pred = test(
            model=model,
            scaler=scaler,
            data=eval_data,
            batch_size=batch_size,
            update_normal_only=update_normal_only,
            freeze_threshold=freeze_threshold,
            metrics_path=exp_dir / "metrics.json",
            return_arrays=True,
        )
        metrics = {**metrics, "online": True, "model_name": model_name}
    else:
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
            "online": False,
            "model_name": model_name,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    np.savez(exp_dir / "scores.npz", scores=scores, y_true=y_test, y_pred=y_pred)
    exp_root.mkdir(parents=True, exist_ok=True)

    # merge into a cumulative summary across configs (keyed by config_name)
    summary_path = exp_root / "summary.json"
    summary_data: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text())
        except Exception:
            summary_data = {}
    summary_data[config_name] = metrics
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    print(
        f"NSL-KDD | {model_name}: f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, pr_auc={metrics['pr_auc']:.4f}"
    )

    from src.eval.plots import generate_all_plots

    generate_all_plots(
        y_true=y_test,
        scores=scores,
        y_pred=y_pred,
        all_metrics={config_name: metrics},
        label=f"{model_name.upper()} NSL-KDD ({mode}, bs={batch_size})",
        out_dir=exp_dir,
    )

    return {"metrics": metrics, "exp_dir": str(exp_dir)}


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
    model_names = ["gmm", "dpmm"]
    if DPGMMDriftAware is not None:
        model_names.append("dpmm_da")
    for model_name in model_names:
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
        help="Model for simulated drift suite or for --nsl (dpmm_da requires src/models/dpmm_da.py to be implemented)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (simulated online test or --nsl --online)")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of components / DPGMM truncation level (overrides the per-model default in _build_model)",
    )
    parser.add_argument(
        "--nsl",
        action="store_true",
        help="NSL-KDD: normal-only train, same protocol as train_and_evaluate_baselineGMM.run_nsl_kdd",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="With --nsl: batched evaluation with partial_fit (else one-shot frozen test)",
    )
    parser.add_argument("--dataset", type=str, default="data/nsl_kdd/nsl_kdd.npz", help="NSL-KDD .npz path")
    parser.add_argument(
        "--exp-root",
        type=str,
        default="experiments/nsl_kdd_dpmm",
        help="Output directory root for --nsl runs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.nsl:
        if args.model == "all":
            raise SystemExit("--model all is for the simulated suite only; use --model dpmm or --model gmm with --nsl")
        run_nsl_kdd_dpmm(
            dataset_path=args.dataset,
            exp_root=args.exp_root,
            model_name=args.model,
            batch_size=args.batch_size,
            online=args.online,
            n_components=args.k,
        )
    elif args.model == "all":
        run_all_models(batch_size=args.batch_size)
    else:
        run_model_pipeline(model_name=args.model, batch_size=args.batch_size)
