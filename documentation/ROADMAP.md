# Independent development roadmap

Goal: a Python workflow for classical ML with reproducible training and auditable model promotion. Bike rental counts are the first reference example. This remains a derivative of [fmind/mlops-python-package](https://github.com/fmind/mlops-python-package), originally authored by Médéric HURIER and upstream contributors. Independent development does not erase that origin. See [ATTRIBUTION.md](../ATTRIBUTION.md).

This roadmap reflects the merged work through PR #23. Completed checks establish their stated engineering contracts, not production readiness or validated forecasting performance. The package name `bikes` and version `4.1.0` remain inherited; there has been no independently branded release.

## Current position

| Area | Current evidence | Remaining boundary |
| --- | --- | --- |
| Installation | Locked exports, clean wheel installation and package-container lifecycle CI | Separate Compose server, HTTP serving and deployment |
| Scientific example | Target-component leakage removed, chronological validation, safe metric arithmetic | Prediction-time contract, baseline comparison and prospective final evaluation |
| Provenance | Full-table hashes and evaluation reference matched to recorded source-run inputs | Actual execution history, tuning exposure and trusted storage |
| Promotion and rollback | Explicit evidence checks, metric policy, alias history and stale-record rejection | Concurrent writers and recovery after partial writes |
| Independent maintenance | Attribution preserved, upstream automation removed, repository links corrected | Distinct release identity and release qualification |

## Next priority: Compose tracking-server integration

The package container now passes its lifecycle smoke with networking disabled and temporary local MLflow storage. The separate service in `docker-compose.yml` has not been exercised by that check. Its image and the locked client both use MLflow 2.20.3, but matching versions alone does not establish interoperability.

Implement a bounded integration check that starts the actual Compose service, waits for readiness, connects with the locked client, and verifies tracking, artifact upload/download and model registration across the process boundary. Define persistent backend and artifact storage explicitly, and confirm records survive a service restart. Keep local ports bound to loopback and clean up the isolated CI resources.

Acceptance: CI runs against the repository's Compose configuration and demonstrates client/server operations and restart persistence. Record the tested scope without calling it a production deployment.

## Following engineering priorities

1. **HTTP model serving.** Serve a registered model and send real JSON scoring requests. Compare results with local predictions and verify malformed requests fail clearly. Existing tests exercise the scoring parser and package execution, not an HTTP server.
2. **Dependency maintenance.** Review inherited dependency security findings and the pending upgrade separately from feature work. Preserve the locked environment checks. Separate optional explanations, notebooks and build tooling from runtime dependencies where the interfaces permit it.
3. **Execution provenance.** Record code revision, configuration, environment identity and tuning-fold membership alongside dataset hashes. Define what can be verified from those records. Protect the tracking, registry and artifact stores; mutable tags alone are not proof of execution.
4. **Registry failure handling.** Define serialization for competing alias writers and recovery when an alias write succeeds but the audit update fails. Validate the design with concurrency and failure-injection tests before claiming atomic promotion or rollback.

## Scientific work and data requirements

The current example estimates hourly rental count using observed weather from the same hour. It is retrospective estimation, not a validated advance forecast.

- **Define the prediction contract before new performance claims.** Either retain retrospective estimation explicitly or specify a forecast horizon and demonstrate that every feature is available at prediction time.
- **Compare simple baselines and the random forest under one evaluation contract.** Keep model and hyperparameter selection inside the development procedure. Report the selection process and do not adjust acceptance thresholds after inspecting final outcomes.
- **Design a prospective final evaluation.** The existing test files have historical exposure and cannot be relabelled untouched. An untouched-test claim requires a defensible new data boundary and evidence that the data were not used for selection. This claim is blocked until that requirement is met.
- **Regenerate affected historical evaluations.** The unsigned metric-arithmetic fix applies to new runs; it does not repair old MLflow records. Historical notebooks may still use target components and are not evidence for the corrected package.

Acceptance: the target and feature-availability contract is explicit, baseline comparisons are reproducible, and any untouched-test claim is supported by actual data history. Passing an installation smoke is not a scientific result.

## Completed engineering work

### Maintenance and installation

- Removed automatic upstream synchronization, inherited funding destination and active template linkage; corrected repository and publishing links while retaining the MIT licence, attribution, Git history and dataset citation.
- Run checks on pull requests and main. Publication remains deliberate.
- Generate runtime requirements and the MLflow environment from `uv.lock`, preserving platform markers and checking export freshness in CI.
- Verify Linux/Python 3.13 wheel installation, dependency consistency, package origin and both CLI entry points outside the checkout.
- Exercise training, registration, reload, evaluation, rejected/accepted promotion and rollback from the installed wheel. Check predictions, fingerprints, alias preservation and audits.
- Build the package container with locked runtime dependencies and execute the same lifecycle with networking disabled and no mounted package sources. CI does not publish the image. The base-image tag remains mutable.
- Add README Mermaid diagrams for the implemented workflow and promotion/rollback decisions.

The lifecycle smoke uses development-data slices, a permissive test-only metric policy and temporary tracking storage. It does not open final-test files or change production thresholds.

### Validation and metrics

- Remove `casual` and `registered` from predictive features and inference requirements, with legacy-column filtering and regression coverage. Old model artifacts require retraining.
- Reject empty, duplicate and misaligned row IDs. Require unique increasing calendar hours in package splitters and reject shuffling. Split sizes and gaps count observations, not elapsed hours.
- Check disjoint IDs and strictly later hours against a supplied evaluation reference. Record explicit checked, rejected or unchecked boundary status.
- Convert target and prediction counts to float64 at scoring boundaries to prevent unsigned residual wraparound while retaining stored schema dtypes and fingerprints.
- Enforce configured evaluation thresholds, reject non-finite bounds and metrics, and distinguish diagnostic runs with empty policies from accepted evidence.

### Provenance and registry controls

- Record versioned full-table SHA-256 fingerprints for validated job data, training partitions and supplied evaluation references.
- Resolve an evaluation selector once to a version URI and record model identity and source run ID.
- Require an explicit promotion candidate, a finished matching evaluation, passed boundary/threshold checks, expected lineage digests and full-table hashes. Independently recheck the promotion metric policy.
- Match the evaluation reference hash to the candidate source run's recorded full development inputs. This establishes consistency of records, not verified training or tuning history.
- Record the previous alias target and promotion policy. Rollback validates matching promotion history, an existing prior target and the expected current alias, then records restoration and reason. Repeated rollback is rejected after restoration.

## Deferred generalization and release

After the reference workflow meets its installation, scientific and registry-control criteria:

- Separate bike-specific schemas and features from reusable workflow interfaces.
- Add a second example to test whether those interfaces generalize.
- Choose a distinct distribution name and document migration from `bikes`.
- Complete API documentation and an end-to-end reproducibility guide.

Before the first independently branded release, review the evidence for each remaining installation, scientific, provenance and registry-control boundary above. Resolve dependency security findings and document the supported deployment scope. Do not equate green CI, a container build or inherited version metadata with production readiness.
