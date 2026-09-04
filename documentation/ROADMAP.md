# Independent development roadmap

Goal: a Python workflow for classical ML with reproducible training and auditable model promotion. Bike rental demand is the first reference example. Infrastructure generalization follows a validated example.

## 1. Independent maintenance

- Remove automatic upstream synchronization, inherited funding destination and active Cruft linkage.
- Retain the MIT licence, author attribution, Git history and dataset citation.
- Point metadata, status badges and publishing at this repository.
- Run checks on pull requests and main; require deliberate publication.

## 2. Reproducible installation

- Choose and validate a stable MLflow version with the existing APIs.
- Completed: generate runtime requirements and MLflow environment from one lockfile, preserving platform markers; verify freshness in CI.
- Reconcile the Compose server version with the client environment.
- Separate optional explanation, notebook and build dependencies from runtime needs.
- Completed: gate CI on Linux/Python 3.13 wheel installation, dependency consistency, installed package origin, and CLI help/schema execution outside the source tree.
- Validate installed end-to-end jobs and container execution separately.
- Review dependency security findings inherited from the fork.

Acceptance: development, installed wheel and documented job environment resolve compatible versions and pass the relevant smoke and integration checks.

## 3. Scientifically valid bike example

- Define the target as rental count, the prediction horizon and feature availability at prediction time.
- Remove casual and registered counts from predictive features and inference requirements.
- Define chronological training, validation and untouched test boundaries.
- Enforce index alignment, split disjointness and temporal ordering.
- Compare meaningful simple baselines with the random forest using the same evaluation contract.
- Keep hyperparameter selection inside the training/validation procedure.

Acceptance: regression tests reject target-derived features and overlapping splits; held-out results are generated reproducibly without test-set tuning.

## 4. Auditable promotion

- Bind dataset hashes, split membership, configuration, code revision and environment to each run.
- Evaluate an explicit candidate version before changing an alias.
- Require a passing evaluation record for that exact model and dataset version.
- Record the previous Champion and provide explicit rollback.
- Fail clearly for missing versions, non-finite metrics or absent evaluation evidence.

Acceptance: a local MLflow integration test demonstrates rejected promotion, accepted promotion and rollback with traceable evidence.

## 5. Reusable package and release

- Separate bike-specific schemas and features from reusable workflow interfaces.
- Decide a distinct distribution name and document migration from bikes.
- Add a second example to test whether the interfaces generalize.
- Publish API documentation, architecture and an end-to-end reproducibility guide.

Release gate: the installation, scientific validation and promotion criteria above pass before the first independently branded release. Do not equate test coverage alone with production readiness.
