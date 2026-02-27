from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


@dataclass
class GPModel:
    random_state: int = 0

    def __post_init__(self) -> None:
        kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPModel":
        self.gp.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, std = self.gp.predict(X, return_std=True)
        return mu, np.maximum(std, 1e-9)
