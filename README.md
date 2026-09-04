# MLOps Python Package

[![Check](https://github.com/DiogoRibeiro7/mlops-python-package/actions/workflows/check.yml/badge.svg)](https://github.com/DiogoRibeiro7/mlops-python-package/actions/workflows/check.yml)

An independently maintained Python project for classical machine-learning workflows, maintained by Diogo Ribeiro. The current implementation uses bike rental demand as its worked example.

The development goal is reproducible training and auditable model promotion. The inherited implementation is a starting point and is not yet a validated production system.

## Current capabilities

- Typed configuration with Pydantic and dataframe validation with Pandera.
- Separate modules for models, data access, registries and execution jobs.
- MLflow experiment tracking, dataset logging and model registration.
- Training, time-series cross-validation, batch inference, evaluation and SHAP explanations.
- Python 3.13, uv, pytest, Ruff, mypy and Bandit.

The package and command remain named `bikes` for compatibility. Version `4.1.0` is inherited and does not represent a new independent release.

## Development setup

Install Python 3.13 and uv, then run:

```bash
git clone https://github.com/DiogoRibeiro7/mlops-python-package.git
cd mlops-python-package
uv sync --locked --all-groups
uv run bikes --help
uv run bikes --schema
```

Run the same core checks used by CI:

```bash
uv run ruff check --no-fix src tests
uv run ruff format --check src tests
uv run mypy src tests
uv run bandit --recursive --configfile=pyproject.toml src
uv run pytest -n auto --cov=src --cov-fail-under=80 tests
```

Use `uv.lock` for development. The inherited `requirements.txt`, `python_env.yaml` and container environment are not yet reconciled. Do not treat them as equivalent installation paths.

## Structure

| Path | Responsibility |
| --- | --- |
| `src/bikes/core/` | Models, metrics and dataframe schemas |
| `src/bikes/io/` | Configuration, datasets, MLflow services and registry adapters |
| `src/bikes/jobs/` | Training, tuning, promotion, inference, evaluation and explanations |
| `src/bikes/utils/` | Splitters, hyperparameter search and model signatures |
| `confs/` | Example job configurations |
| `tests/` | Unit and integration tests with sample data |
| `documentation/` | Maintained project documentation and roadmap |

## Known limitations

The current bike model includes `casual` and `registered`, the components of the target `cnt`. Those features must be removed before reporting forecasting performance. Prediction time and feature availability also need an explicit definition.

The default evaluation configuration reads training data. Promotion selects the latest version when none is supplied and changes the Champion alias without requiring passing evaluation evidence. The inherited `just project` sequence promotes before evaluating. Do not use that sequence to approve a deployment.

Dependency exports disagree with the uv lockfile. Release publishing is therefore manual while the independent baseline is repaired. Dispatching Publish writes documentation to `gh-pages` and publishes a container under `ghcr.io/diogoribeiro7/mlops-python-package`; it should only be run after the release gates in the roadmap pass. Hosted documentation is not assumed to be configured.

See the [roadmap](documentation/ROADMAP.md) for acceptance criteria and implementation order.

## Development and provenance

Changes are proposed through pull requests. CI runs on pull requests and pushes to `main`. Upstream synchronization and template update metadata have been removed so upstream updates require deliberate review.

This project derives from [fmind/mlops-python-package](https://github.com/fmind/mlops-python-package), originally authored by Médéric HURIER. The MIT licence is preserved. See [attribution](ATTRIBUTION.md), the [licence](LICENSE.txt) and the [dataset citation](data/Readme.txt).
