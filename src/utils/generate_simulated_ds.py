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
    """Generate a 2D dataset with controllable drift + anomalies.

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

    # Drift is applied as shift from training distribution to test distribution.
    mean_shift = np.array([[1.8, -0.8], [1.8, -0.8]], dtype=float) * drift_level
    test_means = train_means + mean_shift
    test_covs = np.array(
        [
            [[0.8 + 0.3 * drift_level, 0.1], [0.1, 0.9 + 0.3 * drift_level]],
            [[0.9 + 0.3 * drift_level, -0.05], [-0.05, 0.8 + 0.3 * drift_level]],
        ],
        dtype=float,
    )
    test_weights = np.array([0.5 - 0.1 * drift_level, 0.5 + 0.1 * drift_level], dtype=float)

    X_train = _sample_mixture(n_train, train_means, train_covs, train_weights, rng)
    y_train = np.zeros(n_train, dtype=int)

    X_test_normal = _sample_mixture(
        n_test_normal,
        test_means,
        test_covs,
        test_weights,
        rng,
    )

    # Anomalies sampled from a wide range away from the core modes
    X_test_anomaly = rng.uniform(low=-8.5, high=8.5, size=(n_test_anomaly, 2))

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.hstack(
        [
            np.zeros(n_test_normal, dtype=int),
            np.ones(n_test_anomaly, dtype=int),
        ]
    )

    # Shuffle test set
    perm = rng.permutation(len(X_test))
    X_test = X_test[perm]
    y_test = y_test[perm]

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
