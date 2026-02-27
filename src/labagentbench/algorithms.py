from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tasks import Task
from .models import GPModel
from .bo import expected_improvement_min, sample_uniform, sample_gaussian_around
from .hints import parse_hint


@dataclass
class RunResult:
    X: np.ndarray
    y: np.ndarray
    best_so_far: np.ndarray


def random_search(task: Task, iters: int, seed: int = 0, bounds: Optional[np.ndarray] = None) -> RunResult:
    b = bounds if bounds is not None else task.bounds
    X = sample_uniform(b, iters, seed=seed)
    y = np.array([task.objective(x) for x in X], dtype=float)
    best = np.minimum.accumulate(y)
    return RunResult(X=X, y=y, best_so_far=best)


def gp_ei_bo(
    task: Task,
    iters: int,
    seed: int = 0,
    bounds: Optional[np.ndarray] = None,
    n_init: int = 6,
    n_cand: int = 4096,
    sample_center: Optional[np.ndarray] = None,
) -> RunResult:
    b = bounds if bounds is not None else task.bounds
    rng = np.random.default_rng(seed)

    # init
    if sample_center is None:
        X = sample_uniform(b, n_init, seed=seed)
    else:
        X = sample_gaussian_around(sample_center, b, n_init, seed=seed, rel_std=0.22)

    y = np.array([task.objective(x) for x in X], dtype=float)

    while X.shape[0] < iters:
        model = GPModel(random_state=int(rng.integers(0, 10_000))).fit(X, y)
        # candidate sampling
        if sample_center is None:
            Xcand = sample_uniform(b, n_cand, seed=int(rng.integers(0, 10_000)))
        else:
            Xcand = sample_gaussian_around(sample_center, b, n_cand, seed=int(rng.integers(0, 10_000)), rel_std=0.18)

        mu, sigma = model.predict(Xcand)
        best_y = float(np.min(y))
        ei = expected_improvement_min(mu, sigma, best_y=best_y)
        x_next = Xcand[int(np.argmax(ei))]
        y_next = float(task.objective(x_next))

        X = np.vstack([X, x_next[None, :]])
        y = np.append(y, y_next)

    best = np.minimum.accumulate(y)
    return RunResult(X=X, y=y, best_so_far=best)


def run(task: Task, iters: int, seed: int, mode: str) -> RunResult:
    if mode == "random":
        return random_search(task, iters=iters, seed=seed)

    if mode == "bo":
        return gp_ei_bo(task, iters=iters, seed=seed)

    if mode == "text_prior_bo":
        if not task.hint_text:
            raise ValueError("Task has no hint_text.")
        parsed = parse_hint(task.hint_text, task.var_names, task.bounds)
        b = parsed.bounds if parsed.bounds is not None else task.bounds
        return gp_ei_bo(task, iters=iters, seed=seed, bounds=b, sample_center=parsed.center)

    raise ValueError(f"Unknown mode: {mode}")
