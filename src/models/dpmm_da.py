import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

from .basemodel import GMMBaseModel


class DPGMMDriftAware(GMMBaseModel):
    """Drift-Aware Dirichlet-Process Gaussian Mixture Model.

    Dynamically updates the concentration parameter (alpha) based on
    data likelihood to adapt to concept drift.
    """

    def __init__(
        self,
        n_components: int = 8,
        covariance_type: str = "diag",
        random_state: int | None = None,
        max_iter: int = 600,
        tol: float = 1e-2,
        n_init: int = 1,
        reg_covar: float = 5e-3,
        alpha0: float = 0.3,
        gamma: float = 0.05,
        beta: float = 0.3,
        threshold_quantile: float = 0.01,
        forgetting_factor: float = 0.1,
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
        
        # Drift-aware parameters
        self.alpha0 = alpha0
        self.gamma = gamma
        self.beta = beta
        self.alpha_t = alpha0
        
        self.threshold_quantile = threshold_quantile
        self.forgetting_factor = forgetting_factor
        
        self.partial_fit_max_iter = partial_fit_max_iter
        self.partial_fit_tol = partial_fit_tol

        self.model = self._build_model(self.alpha_t)
        self._threshold: float = 0.0
        
        # Track history of D_t and alpha_t
        self.history = {"D_t": [], "alpha_t": []}

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

    def fit(self, X: np.ndarray | list[list[float]]) -> "DPGMMDriftAware":
        X_arr = np.asarray(X)
        self.model.fit(X_arr)
        self._update_threshold(X_arr)
        return self

    def partial_fit(self, X: np.ndarray | list[list[float]]) -> "DPGMMDriftAware":
        X_arr = np.asarray(X)
        
        # Compute drift signal D_t (average negative log-likelihood)
        if hasattr(self.model, "weights_"):
            scores = self.model.score_samples(X_arr)
            D_t = -np.mean(scores)
            
            # Update alpha_t according to the drift proposal equation:
            # alpha_t = (1 - gamma) * alpha_{t-1} + gamma * alpha_0 * (1 + beta * D_t)
            self.alpha_t = (1 - self.gamma) * self.alpha_t + self.gamma * self.alpha0 * (1 + self.beta * D_t)
            # Ensure alpha_t remains strictly positive for the Dirichlet prior
            self.alpha_t = max(1e-5, self.alpha_t)
            
            # Update the concentration parameter within the scikit-learn model
            self.model.weight_concentration_prior = self.alpha_t
            
            self.history["D_t"].append(float(D_t))
            self.history["alpha_t"].append(float(self.alpha_t))

        # Warm-started VI update with relaxed tol and max_iter
        prev_max_iter = self.model.max_iter
        prev_tol = self.model.tol
        self.model.max_iter = self.partial_fit_max_iter
        self.model.tol = self.partial_fit_tol
        try:
            with warnings.catch_warnings():
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
