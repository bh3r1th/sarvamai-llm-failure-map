# Code-Switching Failure Map

A production-oriented experiment scaffold for mapping failures in code-switched language understanding.
The repository separates data curation, prompting, model execution, evaluation, and reporting boundaries.
This phase provides structure only, with implementation intentionally deferred.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Current Repo Layout

- `src/code_switch_failure_map/`: Core package organized by functional boundaries.
- `scripts/`: Reproducible CLI entrypoints for experiment stages.
- `data/`, `outputs/`, `docs/`: Inputs, generated artifacts, and experiment documentation.
- `tests/`: Placeholder test suite aligned to package modules.

## Placeholder CLI Workflow

```bash
csfm make-dataset
csfm run-models
csfm evaluate
csfm build-report
csfm mine-failures
```

Each command is scaffolded and currently a placeholder.
