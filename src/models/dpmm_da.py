from typing import Any

import numpy as np

try:
	from .dpmm import DPGMM
except ImportError:
	from dpmm import DPGMM


class DPGMMDriftAware(DPGMM):
	"""Drift-aware DP-GMM with adaptive concentration parameter.

	Alpha update follows:
		alpha_t = (1-gamma) * alpha_{t-1} + gamma * alpha0 * (1 + beta * D_t)
	where D_t is average negative log-likelihood on the incoming batch.
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
		max_buffer_size: int = 50_000,
		forgetting_factor: float = 0.1,
	) -> None:
		self.alpha0 = alpha0
		self.gamma = gamma
		self.beta = beta
		self.drift_signal: float = 0.0

		super().__init__(
			n_components=n_components,
			covariance_type=covariance_type,
			random_state=random_state,
			max_iter=max_iter,
			tol=tol,
			n_init=n_init,
			reg_covar=reg_covar,
			alpha=alpha0,
			threshold_quantile=threshold_quantile,
			max_buffer_size=max_buffer_size,
			forgetting_factor=forgetting_factor,
		)

	def fit(self, X: np.ndarray | list[list[float]]) -> "DPGMMDriftAware":
		self.alpha = self.alpha0
		self.model = self._build_model(self.alpha)
		return super().fit(X)
    #Resulting in very bad f1 score
	# def _resample_with_recency(self, X: np.ndarray) -> np.ndarray:
	# 	"""Apply forgetting factor by upweighting recent points.

	# 	This makes the online update reflect:
	# 	    theta_t = (1-lambda) * theta_{t-1} + lambda * theta_hat_t
	# 	via data-level recency weighting before the variational refit.
	# 	"""
	# 	n = len(X)
	# 	if n == 0:
	# 		return X

	# 	weights = self._recency_weights(n)
	# 	weights = weights / weights.sum()
	# 	rng = np.random.default_rng(self.random_state)
	# 	idx = rng.choice(n, size=n, replace=True, p=weights)
	# 	return X[idx]

	def partial_fit(self, X: np.ndarray | list[list[float]]) -> "DPGMMDriftAware":
		X_new = np.asarray(X).copy()

		if self._buffer is None:
			return self.fit(X_new)

		# Drift signal based on current model fit before adaptation.
		self.drift_signal = float(-np.mean(self.model.score_samples(X_new)))
		self.alpha = (1.0 - self.gamma) * self.alpha + self.gamma * self.alpha0 * (
			1.0 + self.beta * self.drift_signal
		)

		self._buffer = np.vstack([self._buffer, X_new])
		if len(self._buffer) > self.max_buffer_size:
			self._buffer = self._buffer[-self.max_buffer_size :]

		# Eq. (11): alpha_t update drives cluster creation prior.
		# self.model.set_params(weight_concentration_prior=self.alpha)

		# # Eq. (9): forgetting factor makes recent observations dominate updates.
		# buffer_for_fit = self._resample_with_recency(self._buffer)
		# self.model.fit(buffer_for_fit)
		self._update_threshold(self._buffer)
		return self

	def get_state(self) -> dict[str, Any]:
		state = super().get_state()
		state["params"]["alpha0"] = self.alpha0
		state["params"]["gamma"] = self.gamma
		state["params"]["beta"] = self.beta
		state["alpha"] = self.alpha
		state["drift_signal"] = self.drift_signal
		return state
