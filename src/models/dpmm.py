from typing import Any

import numpy as np
from sklearn.mixture import BayesianGaussianMixture

try:
	from .basemodel import GMMBaseModel
except ImportError:
	from basemodel import GMMBaseModel


class DPGMM(GMMBaseModel):
	"""Dirichlet-Process Gaussian Mixture via truncated variational inference.

	Uses sklearn's `BayesianGaussianMixture` with
	`weight_concentration_prior_type='dirichlet_process'`.
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
		alpha: float = 0.3, ##########################
		threshold_quantile: float = 0.01,
		max_buffer_size: int = 50_000,
		forgetting_factor: float = 0.1,
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
		self.max_buffer_size = max_buffer_size
		self.forgetting_factor = forgetting_factor

		self.model = self._build_model(self.alpha)
		self._threshold: float = 0.0
		self._buffer: np.ndarray | None = None

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

	def _recency_weights(self, n: int) -> np.ndarray:
		decay = 1.0 - self.forgetting_factor
		if decay <= 0.0:
			return np.ones(n, dtype=float)
		return decay ** np.arange(n - 1, -1, -1, dtype=float)

	def fit(self, X: np.ndarray | list[list[float]]) -> "DPGMM":
		X_arr = np.asarray(X)
		self.model.fit(X_arr)
		self._buffer = X_arr[-self.max_buffer_size :].copy()
		self._update_threshold(X_arr)
		return self

	def partial_fit(self, X: np.ndarray | list[list[float]]) -> "DPGMM":
		X_new = np.asarray(X).copy()

		if self._buffer is None:
			return self.fit(X_new)

		self._buffer = np.vstack([self._buffer, X_new])
		if len(self._buffer) > self.max_buffer_size:
			self._buffer = self._buffer[-self.max_buffer_size :]

		self.model.fit(self._buffer)
		self._update_threshold(self._buffer)
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

	def get_state(self) -> dict[str, Any]:
		active_components = int(np.sum(self.model.weights_ > 1e-3))
		return {
			"params": {
				"n_components": self.n_components,
				"covariance_type": self.covariance_type,
				"random_state": self.random_state,
				"max_iter": self.max_iter,
				"tol": self.tol,
				"n_init": self.n_init,
				"reg_covar": self.reg_covar,
				"alpha": self.alpha,
				"threshold_quantile": self.threshold_quantile,
				"max_buffer_size": self.max_buffer_size,
				"forgetting_factor": self.forgetting_factor,
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
				"active_components": active_components,
			},
		}
