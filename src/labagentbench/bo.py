from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement_min(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    """EI for minimisation (lower is better)."""
    imp = best_y - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def sample_uniform(bounds: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random((n, bounds.shape[0]))
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def sample_gaussian_around(center: np.ndarray, bounds: np.ndarray, n: int, seed: int = 0, rel_std: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    span = bounds[:, 1] - bounds[:, 0]
    std = rel_std * span
    X = rng.normal(loc=center, scale=std, size=(n, len(center)))
    # clip to bounds
    return np.clip(X, bounds[:, 0], bounds[:, 1])
