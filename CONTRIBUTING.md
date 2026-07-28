# Contributing

Thank you for considering a contribution.

## Development setup

```bash
git clone https://github.com/elbakari01/shrinkage_estimator.git
cd shrinkage_estimator
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```bash
pytest
ruff check .
```

## Pull requests

1. Create a focused branch.
2. Keep scientific defaults and reproducibility guarantees unchanged unless the
   change is explicitly documented.
3. Add or update tests for behavioral changes.
4. Describe numerical, statistical, and runtime implications.
5. Do not commit generated simulation results unless they are small, intentional,
   and needed to reproduce a documented result.

## Reporting problems

When reporting a numerical or reproducibility issue, include the Python version,
operating system, command used, configuration, random seed, and relevant traceback.
