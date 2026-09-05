# Independent maintenance: unreleased

- Require explicit promotion candidates and matching finished evaluation evidence, dataset digests and independently passing metrics. Record the previous alias and policy; stop automatic project execution after training.

- Narrow the evaluation acceptance regression’s exception scope to the validation call.

- Persist evaluation threshold policies and acceptance status. Reject non-finite threshold values and evaluation metrics; distinguish diagnostic runs with empty policies from passed evaluations.

- Resolve evaluation model aliases once, load the version URI and record model identity/source run tags, with an alias-movement integration regression.

- Enforce evaluation thresholds through MLflow’s explicit validation API and make the inherited rejection regression strict.

- Evaluate the example on separate test files and validate an optional development reference for disjoint IDs and later calendar hours, recording boundary status and reference lineage in MLflow.

- Reject unaligned or duplicate row IDs before fitting/scoring and unordered or duplicate calendar hours before splitting. Disallow shuffling and invalid time-series split configuration; document observation-based gaps.

- Verify registered custom-model inference with positional indices and the training job’s saved JSON serving example in split and records formats.

- Remove target components from package model features and required inputs; filter legacy count columns and test prediction invariance. Existing registered models require retraining.

- Add a Linux/Python 3.13 installed-wheel CI gate with runtime dependency checks, source isolation, and both CLI entry points.

- Generate consistent runtime environment exports from uv.lock, retaining platform markers and excluding default development groups.
- Check export freshness in CI and distinguish inherited capabilities from independent changes in the README.

- Establish independent maintenance by Diogo Ribeiro with preserved upstream attribution.
- Remove automatic upstream synchronization, active template linkage and inherited funding configuration.
- Replace upstream project links and container destination; make publication manual.
- Run locked CI installation on pull requests and main.
- Document known modelling and dependency limitations and their acceptance gates.

The entries below are retained historical release notes.

## v4.1.0 (2025-03-05)

### Feat

- **gemini**: add support for gemini code assist (#51)
- **dependabot**: add dependabot configuration file (#50)
- **github**: add default rulesets and installation (#47)

### Fix

- **workflows**: fix just in workflows

### Refactor

- **cruft**: update to new template version

## v4.0.0 (2025-03-04)

### Feat

- **tasks**: switch from pyinvoke to just (#42)
- **workflows**: bump GitHub action versions (#41)
- **versions**: bump python and package version (#40)
- **mindmap**: add mindmap of the package (#32)

### Fix

- **version**: ready to bump
- **datasets**: fix dtype backend (#44)

### Refactor

- **cruft**: update to new template version

## v2.0.0 (2024-07-28)

### Feat

- **cruft**: adopt cruft and link it to cookiecutter-mlops-package

## v1.1.3 (2024-07-28)

### Fix

- **mlproject**: fix calling mlflow run by adding project run in front

## v1.1.2 (2024-07-28)

### Fix

- **dependencies**: add setuptools to main dependency for mlflow

## v1.1.1 (2024-07-23)

### Fix

- **publish**: fix publication workflow by installing dev dependencies

## v1.0.1 (2024-06-28)

### Fix

- **version**: bump
