from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Task:
    name: str
    dim: int
    bounds: np.ndarray  # shape (dim, 2)
    objective: Callable[[np.ndarray], float]
    var_names: Tuple[str, ...]
    description: str
    hint_text: str | None = None


def branin(x: np.ndarray) -> float:
    """Branin-Hoo function (minimisation). x in R^2."""
    x1, x2 = x[0], x[1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def hartmann3(x: np.ndarray) -> float:
    """Hartmann 3D function (minimisation), x in [0,1]^3."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35],
        ]
    )
    P = 1e-4 * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    )
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i]) ** 2)
        outer += alpha[i] * np.exp(-inner)
    return -outer


def get_tasks() -> Dict[str, Task]:
    tasks: Dict[str, Task] = {}

    tasks["branin"] = Task(
        name="branin",
        dim=2,
        bounds=np.array([[-5.0, 10.0], [0.0, 15.0]]),
        objective=branin,
        var_names=("x1", "x2"),
        description="Classic 2D Branin-Hoo function (minimisation).",
    )

    tasks["branin_text"] = Task(
        name="branin_text",
        dim=2,
        bounds=np.array([[-5.0, 10.0], [0.0, 15.0]]),
        objective=branin,
        var_names=("x1", "x2"),
        description="Branin with a synthetic text hint that can be parsed into a prior.",
        hint_text=(
            "Prior note: the optimum often lies around x1 ~ -pi to -2 and x2 ~ 10 to 13. "
            "If exploring efficiently, focus x1 between -4 and 0 and x2 between 8 and 14."
        ),
    )

    tasks["hartmann3"] = Task(
        name="hartmann3",
        dim=3,
        bounds=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        objective=hartmann3,
        var_names=("x1", "x2", "x3"),
        description="Hartmann 3D function (minimisation) on [0,1]^3.",
    )

    tasks["hartmann3_text"] = Task(
        name="hartmann3_text",
        dim=3,
        bounds=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        objective=hartmann3,
        var_names=("x1", "x2", "x3"),
        description="Hartmann3 with a synthetic text hint to narrow bounds.",
        hint_text="Hint: promising region is x1 between 0.25 and 0.55, x2 between 0.35 and 0.80, x3 between 0.55 and 0.95.",
    )

    return tasks
