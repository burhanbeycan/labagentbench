from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .tasks import get_tasks
from .eval import run_and_save

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def list_tasks() -> None:
    tasks = get_tasks()
    table = Table(title="Available tasks")
    table.add_column("task")
    table.add_column("dim")
    table.add_column("hint?")
    table.add_column("description")
    for t in tasks.values():
        table.add_row(t.name, str(t.dim), "yes" if t.hint_text else "no", t.description)
    console.print(table)


@app.command()
def run(
    task: str = "branin_text",
    iters: int = 25,
    seed: int = 0,
    mode: str = "text_prior_bo",
    outdir: Path = Path("outputs"),
) -> None:
    """Run a benchmark task.

    mode:
      - random
      - bo
      - text_prior_bo
    """
    csv_path = run_and_save(task, iters=iters, seed=seed, mode=mode, outdir=outdir)
    console.print(f"[green]Saved run to {csv_path}[/green]")


if __name__ == "__main__":
    app()
