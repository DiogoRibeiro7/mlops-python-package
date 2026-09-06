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
- Completed: the Compose image and locked client both use MLflow 2.20.3. Container execution remains unverified.
- Separate optional explanation, notebook and build dependencies from runtime needs.
- Completed: gate CI on Linux/Python 3.13 wheel installation, dependency consistency, installed package origin, and CLI help/schema execution outside the source tree.
- Completed: installed-wheel CI trains, registers and reloads a model outside the checkout, checking predictions and full-input fingerprints with runtime dependencies only.
- Completed: installed-wheel CI evaluates later development rows, rejects promotion with a wrong hash without changing aliases, then applies promotion with matching evidence and checks its audit. This uses a permissive smoke-only policy.
- Still open: installed rollback jobs and container execution.
- Review dependency security findings inherited from the fork.

Acceptance: development, installed wheel and documented job environment resolve compatible versions and pass the relevant smoke and integration checks.

## 3. Scientifically valid bike example

- Define the target as rental count, the prediction horizon and feature availability at prediction time.
- Completed: remove casual and registered counts from package predictive features and inference requirements, with legacy input filtering and regression coverage. Existing registered models require retraining; inherited notebooks remain historical.
- Define chronological training, validation and untouched test boundaries.
- Completed: enforce nonempty, unique, aligned row IDs before fitting/scoring and unique increasing calendar hours in package splitters; reject shuffling. Within each fold, training precedes test and row positions are disjoint.
- Completed: optional evaluation reference checks enforce disjoint IDs and later evaluation hours across supplied datasets; the example uses test files against the development reference. MLflow records reference lineage and explicit checked/unchecked status.
- Still open: bind the reference to exact model training/tuning history and enforce a frozen, untouched final test set.
- Compare meaningful simple baselines with the random forest using the same evaluation contract.
- Keep hyperparameter selection inside the training/validation procedure.

Acceptance: regression tests reject target-derived features and overlapping splits; held-out results are generated reproducibly without test-set tuning.

## 4. Auditable promotion

- Completed: record versioned full-table SHA-256 fingerprints for validated training/tuning/evaluation data, fitting/validation partitions and supplied evaluation references.
- Completed: evaluation resolves a selector once, loads the version URI and records the model identity and source run ID; an alias-movement regression checks version stability.
- Completed: persist the evaluation threshold policy and explicit acceptance status; reject non-finite policy bounds and reported metrics, and mark empty policies unchecked.
- Completed: require an explicit candidate and a finished matching evaluation with passed threshold/reference checks and operator-selected MLflow dataset digests before an alias write. Recheck a separate nonempty promotion policy.
- Completed: require operator-selected full-table hashes during promotion and match the evaluation reference to the source run’s recorded full development inputs.
- Still open: verify actual training history, tuning folds, configuration, code revision and environment, and enforce store access controls.
- Completed: record the previous alias version and promotion policy in a tracking run.
- Completed: explicit rollback validates a completed matching promotion, existing prior target and expected current alias; records the restoration and reason. Local tests cover promotion-to-rollback, CLI execution and stale-record rejection.
- Still open: recover atomically from registry/audit failures and serialize alias writes.
- Fail clearly for missing versions, non-finite metrics or absent evaluation evidence.

Acceptance: a local MLflow integration test demonstrates rejected promotion, accepted promotion and rollback with traceable evidence.

## 5. Reusable package and release

- Separate bike-specific schemas and features from reusable workflow interfaces.
- Decide a distinct distribution name and document migration from bikes.
- Add a second example to test whether the interfaces generalize.
- Publish API documentation, architecture and an end-to-end reproducibility guide.

Release gate: the installation, scientific validation and promotion criteria above pass before the first independently branded release. Do not equate test coverage alone with production readiness.
