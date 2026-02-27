# LabAgentBench — Benchmarks for literature- and human-informed BO

This mini-project provides a small benchmark suite to compare:

- **vanilla Bayesian optimisation**
- **text-prior BO** (hints → bounds/prior shaping)
- **preference-based optimisation** (simulated human comparisons)

The goal is a repo that can underpin a methods paper and show clean engineering practices.

---

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# run a text-hint task for 25 iterations
labagentbench run --task branin_text --iters 25 --seed 0
```

This writes results to `outputs/`.

---

## Extending

- Add your own tasks in `src/labagentbench/tasks.py`
- Add new “hint parsers” in `src/labagentbench/hints.py`
- Swap scikit-learn GP for BoTorch/qNEI when targeting publications
