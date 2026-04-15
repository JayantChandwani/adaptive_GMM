from __future__ import annotations

from pathlib import Path

import numpy as np


def _sample_mixture(
    n_samples: int,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    component_ids = rng.choice(len(weights), size=n_samples, p=weights)
    X = np.zeros((n_samples, means.shape[1]), dtype=float)
    for k in range(len(weights)):
        idx = component_ids == k
        count = int(idx.sum())
        if count:
            X[idx] = rng.multivariate_normal(mean=means[k], cov=covs[k], size=count)
    return X


def generate_simulated_dataset(
    n_train: int = 5000,
    n_test_normal: int = 3000,
    n_test_anomaly: int = 400,
    drift_level: float = 0.5,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a 2D stream with gradual drift and injected anomalies.

    Returns dictionary with:
        X_train, y_train, X_test, y_test
    where y=1 marks anomalies.
    """

    rng = np.random.default_rng(random_state)

    # Training distribution (normal data)
    train_means = np.array([[-2.0, -2.0], [2.2, 2.0]], dtype=float)
    train_covs = np.array(
        [
            [[0.8, 0.15], [0.15, 0.9]],
            [[0.9, -0.1], [-0.1, 0.8]],
        ],
        dtype=float,
    )
    train_weights = np.array([0.5, 0.5], dtype=float)

    # Drift end-point relative to the training distribution.
    mean_shift = np.array([[1.8, -0.8], [1.8, -0.8]], dtype=float) * drift_level
    end_means = train_means + mean_shift
    end_covs = np.array(
        [
            [[0.8 + 0.3 * drift_level, 0.1], [0.1, 0.9 + 0.3 * drift_level]],
            [[0.9 + 0.3 * drift_level, -0.05], [-0.05, 0.8 + 0.3 * drift_level]],
        ],
        dtype=float,
    )
    end_weights = np.array([0.5 - 0.1 * drift_level, 0.5 + 0.1 * drift_level], dtype=float)

    X_train = _sample_mixture(n_train, train_means, train_covs, train_weights, rng)
    y_train = np.zeros(n_train, dtype=int)

    n_test = n_test_normal + n_test_anomaly
    anomaly_rate = n_test_anomaly / max(1, n_test)
    X_test = np.zeros((n_test, train_means.shape[1]), dtype=float)
    y_test = np.zeros(n_test, dtype=int)

    # Online stream: distribution drifts smoothly over time; no shuffling.
    for t in range(n_test):
        progress = t / max(1, n_test - 1)

        if rng.random() < anomaly_rate:
            X_test[t] = rng.uniform(low=-8.5, high=8.5, size=train_means.shape[1])
            y_test[t] = 1
            continue

        means_t = train_means + progress * (end_means - train_means)
        covs_t = train_covs + progress * (end_covs - train_covs)
        weights_t = train_weights + progress * (end_weights - train_weights)
        weights_t = np.clip(weights_t, 1e-6, None)
        weights_t = weights_t / weights_t.sum()
        X_test[t] = _sample_mixture(1, means_t, covs_t, weights_t, rng)[0]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "drift_level": np.array([drift_level], dtype=float),
    }


def save_simulated_dataset(
    output_path: str | Path = "data/simulated/simulated_gmm_drift.npz",
    n_train: int = 5000,
    n_test_normal: int = 3000,
    n_test_anomaly: int = 400,
    drift_level: float = 0.5,
    random_state: int = 42,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_simulated_dataset(
        n_train=n_train,
        n_test_normal=n_test_normal,
        n_test_anomaly=n_test_anomaly,
        drift_level=drift_level,
        random_state=random_state,
    )

    np.savez(output_path, **data)
    return output_path


def save_dataset_suite(
    output_dir: str | Path = "data/simulated",
    drift_levels: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0),
    random_state: int = 42,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for i, drift_level in enumerate(drift_levels):
        file_path = output_dir / f"simulated_drift_{drift_level:.1f}.npz"
        save_simulated_dataset(
            output_path=file_path,
            drift_level=drift_level,
            random_state=random_state + i,
        )
        saved_paths.append(file_path)

    return saved_paths


if __name__ == "__main__":
    dataset_paths = save_dataset_suite()
    for dataset_path in dataset_paths:
        print(f"Saved simulated dataset to: {dataset_path}")
