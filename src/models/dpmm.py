import numpy as np
from sklearn.mixture import BayesianGaussianMixture

from .basemodel import GMMBaseModel


class DPGMM(GMMBaseModel):
    """Dirichlet-Process Gaussian Mixture via truncated variational inference."""

    def __init__(
        self,
        n_components: int = 8,
        covariance_type: str = "diag",
        random_state: int | None = None,
        max_iter: int = 600,
        tol: float = 1e-2,
        n_init: int = 1,
        reg_covar: float = 5e-3,
        alpha: float = 0.3,
        threshold_quantile: float = 0.01,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.alpha = alpha
        self.threshold_quantile = threshold_quantile

        self.model = self._build_model(self.alpha)
        self._threshold: float = 0.0

    def _build_model(self, alpha: float) -> BayesianGaussianMixture:
        return BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            reg_covar=self.reg_covar,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=alpha,
            init_params="kmeans",
            warm_start=True,
        )

    def _update_threshold(self, X: np.ndarray) -> None:
        scores = self.model.score_samples(X)
        self._threshold = float(np.quantile(scores, self.threshold_quantile))

    def fit(self, X: np.ndarray | list[list[float]]) -> "DPGMM":
        X_arr = np.asarray(X)
        self.model.fit(X_arr)
        self._update_threshold(X_arr)
        return self

    def partial_fit(self, X: np.ndarray | list[list[float]]) -> "DPGMM":
        X_arr = np.asarray(X)
        self.model.fit(X_arr)
        self._update_threshold(X_arr)
        return self

    def score_samples(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        return self.model.score_samples(np.asarray(X))

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        return self.model.predict(np.asarray(X))

    def predict_anomaly(
        self,
        X: np.ndarray | list[list[float]],
        threshold: float | None = None,
    ) -> np.ndarray:
        scores = self.score_samples(X)
        final_threshold = self._threshold if threshold is None else threshold
        return scores < final_threshold