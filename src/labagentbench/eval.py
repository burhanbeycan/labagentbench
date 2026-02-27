from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .algorithms import run
from .tasks import get_tasks


def run_and_save(task_name: str, iters: int, seed: int, mode: str, outdir: Path) -> Path:
    tasks = get_tasks()
    if task_name not in tasks:
        raise KeyError(f"Unknown task: {task_name}. Available: {list(tasks)}")
    task = tasks[task_name]

    res = run(task, iters=iters, seed=seed, mode=mode)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(res.X, columns=list(task.var_names))
    df["y"] = res.y
    df["best_so_far"] = res.best_so_far
    csv_path = outdir / f"{task_name}__{mode}__seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    # plot
    plt.figure()
    plt.plot(df["best_so_far"].to_numpy())
    plt.xlabel("iteration")
    plt.ylabel("best objective (min)")
    plt.title(f"{task_name} — {mode}")
    fig_path = outdir / f"{task_name}__{mode}__seed{seed}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    return csv_path
