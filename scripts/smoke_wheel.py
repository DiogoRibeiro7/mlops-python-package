"""Verify an installed wheel from outside the checkout using its locked runtime dependencies.

Invoke with the clean environment's Python and -I to disable source-path injection.
The checkout supplies expected metadata and sample data only; bikes must load from the environment.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bikes.io.services import MlflowService


def check_evaluation_and_promotion(root: Path, service: MlflowService, version: int) -> None:
    """Exercise acceptance and rejection with later development rows in a local store."""
    import pandas as pd

    from bikes.core import metrics, schemas
    from bikes.io import datasets, provenance, services
    from bikes.jobs import EvaluationsJob, PromotionJob, RollbackJob

    # These rows follow the training slice. No final-test files are opened.
    inputs = schemas.InputsSchema.check(
        pd.read_parquet(root / "data/inputs_train.parquet").iloc[1500:1668]
    )
    targets = schemas.TargetsSchema.check(
        pd.read_parquet(root / "data/targets_train.parquet").iloc[1500:1668]
    )
    reference = schemas.InputsSchema.check(pd.read_parquet("inputs.parquet"))
    inputs.to_parquet("evaluation_inputs.parquet")
    targets.to_parquet("evaluation_targets.parquet")
    # An explicit permissive smoke policy tests execution, not model quality.
    policy = {"mean_squared_error": metrics.Threshold(threshold=1e12, greater_is_better=False)}
    evaluation = EvaluationsJob(
        inputs=datasets.ParquetReader(path="evaluation_inputs.parquet"),
        targets=datasets.ParquetReader(path="evaluation_targets.parquet"),
        reference_inputs=datasets.ParquetReader(path="inputs.parquet"),
        alias_or_version=version,
        thresholds=policy,
        mlflow_service=service,
        logger_service=services.LoggerService(level="WARNING"),
        run_config=service.RunConfig(name="WheelEvaluation", log_system_metrics=False),
    )
    with evaluation:
        result = evaluation.run()
        client = service.client()
        evidence = client.get_run(result["run"].info.run_id)
        if (
            evidence.info.status != "FINISHED"
            or evidence.data.tags.get("evaluation.thresholds") != "passed"
            or evidence.data.tags.get("evaluation.boundary") != "passed_against_reference"
        ):
            raise RuntimeError("Installed evaluation did not record accepted evidence")
        expected_max_error = (
            (
                targets[schemas.TargetsSchema.cnt].astype("float64")
                - result["outputs"][schemas.OutputsSchema.prediction].astype("float64")
            )
            .abs()
            .max()
        )
        if evidence.data.metrics.get("max_error") != expected_max_error:
            raise RuntimeError("Installed evaluation maximum error disagrees with residuals")
        hashes = PromotionJob.DatasetHashes(
            inputs=provenance.fingerprint(inputs),
            targets=provenance.fingerprint(targets),
            reference_inputs=provenance.fingerprint(reference),
        )
        promotion = PromotionJob(
            version=version,
            evaluation_run_id=evidence.info.run_id,
            dataset_digests=PromotionJob.DatasetDigests.model_validate(
                {
                    item.dataset.name: item.dataset.digest
                    for item in evidence.inputs.dataset_inputs
                    if item.dataset.name in {"inputs", "targets", "reference_inputs"}
                }
            ),
            dataset_sha256=hashes,
            thresholds=policy,
            mlflow_service=service,
            run_config=service.RunConfig(name="WheelPromotion", log_system_metrics=False),
        )
        # A second registration of the real artifact supplies a prior alias target.
        # Version numbers encode registration order, not promotion order or quality.
        candidate = client.get_model_version(service.registry_name, str(version))
        previous = client.create_model_version(
            service.registry_name, source=candidate.source, run_id=candidate.run_id
        )
        client.set_registered_model_alias(
            service.registry_name, promotion.alias, str(previous.version)
        )
        before = dict(client.get_registered_model(service.registry_name).aliases)
        bad_hashes = hashes.model_dump() | {"inputs": "0" * 64}
        rejected = PromotionJob.model_validate(
            promotion.model_dump() | {"dataset_sha256": bad_hashes}
        )
        try:
            rejected.run()
        except ValueError as error:
            if "Evaluation full-table fingerprint mismatch: inputs" not in str(error):
                raise
        else:
            raise RuntimeError("Installed promotion accepted an incorrect input hash")
        if dict(client.get_registered_model(service.registry_name).aliases) != before:
            raise RuntimeError("Rejected promotion changed registry aliases")
        applied = promotion.run()
        audit = client.get_run(applied["run"].info.run_id)
        selected = client.get_model_version_by_alias(service.registry_name, promotion.alias)
        if (
            int(selected.version) != version
            or audit.info.status != "FINISHED"
            or audit.data.tags.get("promotion.status") != "applied"
            or audit.data.tags.get("promotion.evaluation_run_id") != evidence.info.run_id
        ):
            raise RuntimeError("Installed promotion did not apply and record the chosen version")
        if audit.data.tags.get("promotion.previous_version") != str(previous.version):
            raise RuntimeError("Installed promotion did not record the prior alias target")
        rollback = RollbackJob(
            promotion_run_id=audit.info.run_id,
            expected_version=version,
            target_version=int(previous.version),
            reason="Verify installed rollback routing and audit",
            alias=promotion.alias,
            mlflow_service=service,
            run_config=service.RunConfig(name="WheelRollback", log_system_metrics=False),
        )
        restored = rollback.run()
        rollback_audit = client.get_run(restored["run"].info.run_id)
        if (
            rollback_audit.info.status != "FINISHED"
            or rollback_audit.data.tags.get("rollback.status") != "applied"
            or rollback_audit.data.tags.get("rollback.promotion_run_id") != audit.info.run_id
            or rollback_audit.data.tags.get("rollback.to_version") != str(previous.version)
            or int(
                client.get_model_version_by_alias(service.registry_name, promotion.alias).version
            )
            != int(previous.version)
        ):
            raise RuntimeError("Installed rollback did not restore and audit the prior version")
        try:
            rollback.run()
        except ValueError as error:
            if "Alias no longer points to the expected promoted version" not in str(error):
                raise
        else:
            raise RuntimeError("Installed rollback accepted a stale promotion record")
        if dict(client.get_registered_model(service.registry_name).aliases) != before:
            raise RuntimeError("Repeated rollback changed registry aliases")
    print("Installed evaluation, promotion and rollback with stale-record rejection verified")


def check_training(root: Path) -> None:
    """Train, register and reload a small model using only the installed package.

    The caller has already checked package origin and changed to a temporary
    directory. Sample data is copied there before the job runs. This is an
    installation regression, not a model-quality or final-test evaluation.
    """
    import pandas as pd

    from bikes.core import schemas
    from bikes.io import datasets, provenance, registries, services
    from bikes.jobs import TrainingJob
    from bikes.utils import splitters

    inputs = schemas.InputsSchema.check(
        pd.read_parquet(root / "data/inputs_train.parquet").iloc[:1500]
    )
    targets = schemas.TargetsSchema.check(
        pd.read_parquet(root / "data/targets_train.parquet").iloc[:1500]
    )
    inputs.to_parquet("inputs.parquet")
    targets.to_parquet("targets.parquet")
    store_uri: str = (Path.cwd() / "mlruns").as_uri()
    service = services.MlflowService(
        tracking_uri=store_uri,
        registry_uri=store_uri,
        experiment_name="InstalledWheel",
        registry_name="InstalledWheel",
        autolog_disable=True,
    )
    job = TrainingJob(
        inputs=datasets.ParquetReader(path="inputs.parquet"),
        targets=datasets.ParquetReader(path="targets.parquet"),
        splitter=splitters.TrainTestSplitter(test_size=168),
        mlflow_service=service,
        logger_service=services.LoggerService(level="WARNING"),
        run_config=service.RunConfig(name="WheelTraining", log_system_metrics=False),
    )
    with job:
        result = job.run()
        version = result["model_version"]
        uri: str = registries.uri_for_model_version(service.registry_name, int(version.version))
        reloaded = registries.CustomLoader().load(uri)
        actual = reloaded.predict(result["inputs_test"])
        pd.testing.assert_frame_equal(actual, result["outputs_test"])
        run = service.client().get_run(result["run"].info.run_id)
        if run.info.status != "FINISHED" or version.run_id != run.info.run_id:
            raise RuntimeError("Installed training did not register its completed source run")
        tables: dict[str, pd.DataFrame] = {"inputs": inputs, "targets": targets}
        for role, table in tables.items():
            if (
                run.data.tags.get(f"data.{role}.sha256") != provenance.fingerprint(table)
                or run.data.tags.get(f"data.{role}.format") != provenance.FORMAT
            ):
                raise RuntimeError(f"Installed training fingerprint mismatch: {role}")
    print("Installed training, registration, reload and prediction roundtrip verified")
    check_evaluation_and_promotion(root, service, int(version.version))


def main() -> int:
    """Check package origin, installed version and both command-line entry points."""
    root: Path = Path(__file__).resolve().parents[1]
    metadata: dict[str, object] = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    project: object = metadata.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise RuntimeError("Project metadata must declare a string version")
    expected_version: str = project["version"]
    installed_version: str = importlib.metadata.version("bikes")
    if installed_version != expected_version:
        raise RuntimeError(f"Installed {installed_version}, expected {expected_version}")
    environment: dict[str, str] = dict(os.environ)
    # A subprocess must not inherit paths that could expose the source checkout.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    executable: Path = Path(sysconfig.get_path("scripts")) / (
        "bikes.exe" if os.name == "nt" else "bikes"
    )
    commands: list[list[str]] = [
        [str(executable)],
        [sys.executable, "-I", "-m", "bikes"],
    ]
    with (
        tempfile.TemporaryDirectory(prefix="bikes-wheel-smoke-") as directory,
        contextlib.chdir(directory),
    ):
        package: ModuleType = importlib.import_module("bikes")
        if package.__file__ is None:
            raise RuntimeError("Installed bikes package has no module file")
        origin: Path = Path(package.__file__).resolve()
        if not origin.is_relative_to(Path(sys.prefix).resolve()) or origin.is_relative_to(root):
            raise RuntimeError(f"Package loaded outside the clean environment: {origin}")
        for command in commands:
            help_result: subprocess.CompletedProcess[str] = subprocess.run(
                [*command, "--help"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                timeout=120,
            )
            if "--schema" not in help_result.stdout:
                raise RuntimeError("CLI help does not expose the schema option")
            schema_result: subprocess.CompletedProcess[str] = subprocess.run(
                [*command, "--schema"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                timeout=120,
            )
            schema: object = json.loads(schema_result.stdout)
            if not isinstance(schema, dict) or "job" not in schema.get("properties", {}):
                raise RuntimeError("CLI did not emit the expected settings JSON schema")
        check_training(root)
        print(f"Installed bikes {installed_version} verified at {origin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
