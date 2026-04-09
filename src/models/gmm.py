from typing import Any
import numpy as np
from sklearn.mixture import GaussianMixture

try:
    from .basemodel import GMMBaseModel
except ImportError:
    from basemodel import GMMBaseModel


class GMM(GMMBaseModel):
    """Sklearn-backed Gaussian Mixture Model wrapper.

    Notes:
        - `GaussianMixture` does not support true online `partial_fit`.
        - Here, `partial_fit` is implemented by keeping a rolling buffer and
          re-fitting on the buffered data.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        random_state: int | None = None,
        max_iter: int = 200,
        reg_covar: float = 1e-6,
        threshold_quantile: float = 0.01,
        max_buffer_size: int = 50_000,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.threshold_quantile = threshold_quantile
        self.max_buffer_size = max_buffer_size

        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
            reg_covar=self.reg_covar,
        )

        self._threshold: float = 0.0
        self._buffer: np.ndarray | None = None

    def _update_threshold(self, X: np.ndarray) -> None:
        scores = self.model.score_samples(X)
        self._threshold = float(np.quantile(scores, self.threshold_quantile))

    def fit(self, X: np.ndarray | list[list[float]]) -> "GMM":
        self.model.fit(X)
        self._buffer = X[-self.max_buffer_size :].copy()
        self._update_threshold(X)
        return self

    def partial_fit(self, X: np.ndarray | list[list[float]]) -> "GMM":
        X_new = X.copy()

        if self._buffer is None:
            return self.fit(X_new)

        self._buffer = np.vstack([self._buffer, X_new])

        if len(self._buffer) > self.max_buffer_size:
            self._buffer = self._buffer[-self.max_buffer_size :]

        self.model.fit(self._buffer)
        self._update_threshold(self._buffer)
        return self

    def score_samples(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        return self.model.score_samples(X)

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        return self.model.predict(X)

    def predict_anomaly(
        self,
        X: np.ndarray | list[list[float]],
        threshold: float | None = None,
    ) -> np.ndarray:
        scores = self.model.score_samples(X)
        final_threshold = self._threshold if threshold is None else threshold
        return scores < final_threshold

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "params": {
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "reg_covar": self.reg_covar,
                "threshold_quantile": self.threshold_quantile,
                "max_buffer_size": self.max_buffer_size,
            },
            "threshold": self._threshold,
            "buffer": self._buffer,
            "model": {
                "weights_": self.model.weights_,
                "means_": self.model.means_,
                "covariances_": self.model.covariances_,
                "precisions_cholesky_": self.model.precisions_cholesky_,
                "converged_": self.model.converged_,
                "n_iter_": self.model.n_iter_,
                "lower_bound_": self.model.lower_bound_,
            },
        }

        return state
