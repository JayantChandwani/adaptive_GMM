import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
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
        partial_fit_max_iter: int = 100,
        partial_fit_tol: float = 1e-3,
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
        # partial_fit warm-starts from the converged solution and only needs a
        # few VI iterations to absorb each new batch. Running max_iter=600 per
        # batch is wasteful and floods stderr with ConvergenceWarning noise
        # (the ELBO change between iterations on small batches is smaller than
        # the strict full-fit tol but the model is already near optimum).
        self.partial_fit_max_iter = partial_fit_max_iter
        self.partial_fit_tol = partial_fit_tol

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
        # Temporarily relax tolerance + cap iterations for the warm-started
        # update; restore originals afterwards so a follow-up full fit() still
        # uses the strict settings.
        prev_max_iter = self.model.max_iter
        prev_tol = self.model.tol
        self.model.max_iter = self.partial_fit_max_iter
        self.model.tol = self.partial_fit_tol
        try:
            with warnings.catch_warnings():
                # Warm-started VI on tiny batches frequently brushes max_iter
                # without crossing tol; this is expected and not a bug.
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.model.fit(X_arr)
        finally:
            self.model.max_iter = prev_max_iter
            self.model.tol = prev_tol
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