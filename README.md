# MLOps Python Package

[![Check](https://github.com/DiogoRibeiro7/mlops-python-package/actions/workflows/check.yml/badge.svg)](https://github.com/DiogoRibeiro7/mlops-python-package/actions/workflows/check.yml)

**This is a derivative project, originally forked from [fmind/mlops-python-package](https://github.com/fmind/mlops-python-package), authored by Médéric HURIER and upstream contributors.** Its original architecture, bike example and most of its current implementation were inherited. Detaching the GitHub fork does not change that provenance.

Diogo Ribeiro maintains this independent continuation. Our aim is to improve reproducibility, scientific validation and model-promotion controls while giving the original work clear credit.

The development goal is reproducible training and auditable model promotion. The inherited implementation is a starting point and is not yet a validated production system.

## What has changed here

The independent work so far establishes maintenance ownership, removes automatic upstream synchronization, corrects repository and publishing links, and documents the inherited limitations. Runtime environment exports now come from the uv lockfile with a CI freshness check and preserved platform markers. A separate Linux/Python 3.13 job builds the wheel, installs it in a clean runtime environment, checks dependency consistency, and verifies package origin plus CLI help and JSON schema output outside the checkout.

The bike model now excludes the target components `casual` and `registered`. Input validation removes those legacy columns before training, inference, explanations and signature generation; they are no longer required. Regression tests check that changing them cannot change a newly trained model’s predictions. Independent evaluation and evidence-based promotion remain open work. The [roadmap](documentation/ROADMAP.md) defines their acceptance criteria, and [ATTRIBUTION.md](ATTRIBUTION.md) records the starting point.

## Inherited capabilities

- Typed configuration with Pydantic and dataframe validation with Pandera.
- Separate modules for models, data access, registries and execution jobs.
- MLflow experiment tracking, dataset logging and model registration.
- Training, time-series cross-validation, batch inference, evaluation and SHAP explanations.
- Python 3.13, uv, pytest, Ruff, mypy and Bandit.

The package and command remain named `bikes` for compatibility. Version `4.1.0` is inherited and does not represent a new independent release.

## Development setup

Install Python 3.13 and uv 0.11.33 (the version pinned in CI), then run:

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

Use `uv.lock` as the dependency source of truth. Regenerate and verify runtime exports with:

```bash
python scripts/export_environment.py
python scripts/export_environment.py --check
```

Both exports exclude default development groups and retain platform conditions. This aligns them with the existing MLflow 2.20.3 baseline; it is not a dependency security upgrade. The pending dependency update requires a separate compatibility review. The wheel smoke check covers imports and CLI startup on Linux/Python 3.13. It does not establish model quality, full job execution, other operating systems or container correctness. Container validation remains open.

## Structure

| Path | Responsibility |
| --- | --- |
| `src/bikes/core/` | Models, metrics and dataframe schemas |
| `src/bikes/io/` | Configuration, datasets, MLflow services and registry adapters |
| `src/bikes/jobs/` | Training, tuning, promotion, inference, evaluation and explanations |
| `src/bikes/utils/` | Splitters, hyperparameter search and model signatures |
| `confs/` | Example job configurations |
| `tests/` | Unit and integration tests with sample data |
| `scripts/` | Runtime environment export and verification |
| `documentation/` | Maintained project documentation and roadmap |

## Known limitations

The model estimates hourly rental count (`cnt`) using calendar fields and observed weather for that hour. This is a retrospective estimation example, not a validated advance forecast: a prediction horizon and weather availability at prediction time still need to be defined. Removing `casual` and `registered` fixes direct target-component leakage only.

Existing registered models retain their old features and signatures: retrain and register a new version to use this change. The inherited notebooks and their saved outputs are historical exploratory work and may still use target components; they are not evidence for the corrected package’s performance.

The default evaluation configuration reads training data. Promotion selects the latest version when none is supplied and changes the Champion alias without requiring passing evaluation evidence. The inherited `just project` sequence promotes before evaluating. Do not use that sequence to approve a deployment.

Release publishing remains manual while scientific validation, dependency security review and container validation are incomplete. Dispatching Publish writes documentation to `gh-pages` and publishes a container under `ghcr.io/diogoribeiro7/mlops-python-package`; it should only be run after the release gates in the roadmap pass. Hosted documentation is not assumed to be configured.

See the [roadmap](documentation/ROADMAP.md) for acceptance criteria and implementation order.

## Development and provenance

Changes are proposed through pull requests. CI runs on pull requests and pushes to `main`. Upstream synchronization and template update metadata have been removed so upstream updates require deliberate review.

This project derives from [fmind/mlops-python-package](https://github.com/fmind/mlops-python-package), originally authored by Médéric HURIER. The MIT licence is preserved. See [attribution](ATTRIBUTION.md), the [licence](LICENSE.txt) and the [dataset citation](data/Readme.txt).
