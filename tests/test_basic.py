import numpy as np

from labagentbench.tasks import get_tasks
from labagentbench.hints import parse_hint
from labagentbench.algorithms import run


def test_tasks_exist():
    tasks = get_tasks()
    assert "branin" in tasks
    assert tasks["branin"].dim == 2


def test_hint_parser():
    tasks = get_tasks()
    t = tasks["branin_text"]
    parsed = parse_hint(t.hint_text, t.var_names, t.bounds)
    assert parsed.bounds is not None
    assert parsed.bounds.shape == t.bounds.shape


def test_run_text_prior_bo_small():
    tasks = get_tasks()
    t = tasks["branin_text"]
    res = run(t, iters=12, seed=0, mode="text_prior_bo")
    assert res.X.shape == (12, t.dim)
    assert res.best_so_far.shape == (12,)
