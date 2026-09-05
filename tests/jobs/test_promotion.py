"""Exercise promotion rejection and alias preservation against a local MLflow store."""

import json
from pathlib import Path

import mlflow
import pandas as pd
import pydantic as pdt
import pytest
from pytest_mock import MockerFixture

from bikes import jobs
from bikes.io import provenance, services


@pytest.fixture
def promotion(
    mlflow_service: services.MlflowService, alerts_service: services.AlertsService
) -> jobs.PromotionJob:
    """Create two registry versions and a controlled evaluation evidence record."""
    client = mlflow_service.client()
    name = mlflow_service.registry_name
    table = pd.DataFrame({"value": [1, 2]})
    with mlflow_service.run_context(mlflow_service.RunConfig(name="Source")) as source:
        provenance.log_frames({"inputs": table})
        client.create_registered_model(name)
        for _ in range(2):
            client.create_model_version(
                name, source=f"runs:/{source.info.run_id}/model", run_id=source.info.run_id
            )
    client.set_registered_model_alias(name, "Champion", "1")
    digests: dict[str, str] = {}
    with mlflow_service.run_context(mlflow_service.RunConfig(name="Evidence")) as run:
        mlflow.set_tags(
            {
                "evaluation.model_name": name,
                "evaluation.model_version": "2",
                "evaluation.model_uri": f"models:/{name}/2",
                "evaluation.model_source_run_id": source.info.run_id,
                "evaluation.thresholds": "passed",
                "evaluation.boundary": "passed_against_reference",
            }
        )
        provenance.log_frames(dict.fromkeys(("inputs", "targets", "reference_inputs"), table))
        for dataset_name in ("inputs", "targets", "reference_inputs"):
            dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
                pd.DataFrame({"value": [1, 2]}), name=dataset_name
            )
            mlflow.log_input(
                dataset,
                context="evaluation_reference"
                if dataset_name == "reference_inputs"
                else "Evaluations",
            )
            digests[dataset_name] = dataset.digest
        mlflow.log_metric("r2_score", 0.8)
    return jobs.PromotionJob(
        mlflow_service=mlflow_service,
        alerts_service=alerts_service,
        version=2,
        evaluation_run_id=run.info.run_id,
        dataset_digests=jobs.PromotionJob.DatasetDigests.model_validate(digests),
        dataset_sha256=jobs.PromotionJob.DatasetHashes.model_validate(
            dict.fromkeys(digests, provenance.fingerprint(table))
        ),
    )


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "failed",
        "running",
        "deleted",
        "experiment",
        "missing_run",
        "missing_version",
        "name",
        "version",
        "uri",
        "source",
        "unchecked",
        "rejected",
        "boundary",
        "missing_tag",
        "inputs_digest",
        "targets_digest",
        "reference_digest",
        "missing_metric",
        "low_metric",
        "nan",
        "infinity",
    ],
)
def test_promotion_evidence(promotion: jobs.PromotionJob, case: str, mocker: MockerFixture) -> None:
    """Every rejection must leave Champion unchanged and never call its setter."""
    client = promotion.mlflow_service.client()
    run_id = promotion.evaluation_run_id
    data = promotion.model_dump()
    tag_changes = {
        "name": ("evaluation.model_name", "Other"),
        "version": ("evaluation.model_version", "1"),
        "uri": ("evaluation.model_uri", "models:/Other/2"),
        "source": ("evaluation.model_source_run_id", "other"),
        "unchecked": ("evaluation.thresholds", "unchecked"),
        "rejected": ("evaluation.thresholds", "rejected"),
        "boundary": ("evaluation.boundary", "unchecked"),
    }
    if case in tag_changes:
        client.set_tag(run_id, *tag_changes[case])
    elif case in {"failed", "running"}:
        client.set_terminated(run_id, status=case.upper())
    elif case == "deleted":
        client.delete_run(run_id)
    elif case == "experiment":
        data["mlflow_service"]["experiment_name"] = "Different"
    elif case == "missing_run":
        data["evaluation_run_id"] = "0" * 32
    elif case == "missing_version":
        data["version"] = 999
    elif case == "missing_tag":
        client.delete_tag(run_id, "evaluation.thresholds")
    elif case.endswith("_digest"):
        key = {
            "inputs_digest": "inputs",
            "targets_digest": "targets",
            "reference_digest": "reference_inputs",
        }[case]
        data["dataset_digests"][key] = "wrong"
    elif case == "missing_metric":
        data["thresholds"] = {"absent": {"threshold": 0.5, "greater_is_better": True}}
    elif case in {"low_metric", "nan", "infinity"}:
        value = {"low_metric": 0.1, "nan": float("nan"), "infinity": float("inf")}[case]
        client.log_metric(run_id, "r2_score", value, step=1)
    job = jobs.PromotionJob.model_validate(data)
    setter = mocker.spy(mlflow.tracking.MlflowClient, "set_registered_model_alias")
    with job:
        if case == "valid":
            out = job.run()
            setter.assert_called_once()
            audit = client.get_run(out["run"].info.run_id)
            assert audit.info.status == "FINISHED"
            assert audit.data.tags["promotion.previous_version"] == "1"
            assert audit.data.tags["promotion.evaluation_run_id"] == run_id
            assert audit.data.tags["promotion.status"] == "applied"
            hashes_path = client.download_artifacts(
                audit.info.run_id, "promotion/dataset_sha256.json"
            )
            assert json.loads(Path(hashes_path).read_text()) == {
                "format": provenance.FORMAT,
                **job.dataset_sha256.model_dump(),
            }
            # Use the actual promotion audit to restore the previous alias.
            rollback = jobs.RollbackJob(
                mlflow_service=promotion.mlflow_service,
                alerts_service=promotion.alerts_service,
                promotion_run_id=audit.info.run_id,
                expected_version=2,
                target_version=1,
                reason="Exercise promotion and rollback history.",
            )
            rollback_out = rollback.run()
            restored = client.get_run(rollback_out["run"].info.run_id)
            assert restored.info.status == "FINISHED"
            assert restored.data.tags["rollback.promotion_run_id"] == audit.info.run_id
            assert restored.data.tags["rollback.status"] == "applied"

        else:
            with pytest.raises((ValueError, mlflow.exceptions.MlflowException)):
                job.run()
            setter.assert_not_called()
        assert (
            int(
                client.get_model_version_by_alias(
                    promotion.mlflow_service.registry_name, "Champion"
                ).version
            )
            == 1
        )


@pytest.mark.parametrize(
    "change", [{"version": None}, {"version": 0}, {"evaluation_run_id": ""}, {"thresholds": {}}]
)
def test_promotion_requires_explicit_evidence(
    promotion: jobs.PromotionJob, change: dict[str, object]
) -> None:
    """Configuration cannot opt back into latest-version or empty-policy promotion."""
    with pytest.raises(pdt.ValidationError):
        jobs.PromotionJob.model_validate(promotion.model_dump() | change)


@pytest.mark.parametrize("role", ["inputs", "targets", "reference_inputs", "source"])
@pytest.mark.parametrize("defect", ["missing", "changed", "format"])
def test_promotion_requires_fingerprints(
    promotion: jobs.PromotionJob, role: str, defect: str, mocker: MockerFixture
) -> None:
    """Reject inconsistent hashes even while all legacy lineage checks still pass."""
    client = promotion.mlflow_service.client()
    run_id = promotion.evaluation_run_id
    if role == "source":
        candidate = client.get_model_version(promotion.mlflow_service.registry_name, "2")
        assert candidate.run_id is not None
        run_id = candidate.run_id
        role = "inputs"
    key = f"data.{role}.sha256"
    if defect == "missing":
        client.delete_tag(run_id, key)
    elif defect == "changed":
        client.set_tag(run_id, key, "0" * 64)
    else:
        client.set_tag(run_id, f"data.{role}.format", "unknown.v2")
    setter = mocker.spy(mlflow.tracking.MlflowClient, "set_registered_model_alias")
    with promotion, pytest.raises(ValueError):
        promotion.run()
    setter.assert_not_called()


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "deleted"])
def test_promotion_requires_completed_source(
    promotion: jobs.PromotionJob, status: str, mocker: MockerFixture
) -> None:
    """A valid evaluation cannot authorize an incomplete or deleted source run."""
    client = promotion.mlflow_service.client()
    candidate = client.get_model_version(promotion.mlflow_service.registry_name, "2")
    assert candidate.run_id is not None
    if status == "deleted":
        client.delete_run(candidate.run_id)
    else:
        client.set_terminated(candidate.run_id, status=status)
    setter = mocker.spy(mlflow.tracking.MlflowClient, "set_registered_model_alias")
    with promotion, pytest.raises(ValueError, match="source run"):
        promotion.run()
    setter.assert_not_called()


@pytest.mark.parametrize("value", ["", "a" * 63, "g" * 64, "A" * 64, "a" * 65])
def test_hash_configuration_rejects_invalid_sha256(value: str) -> None:
    """Hashes must use the canonical lowercase SHA-256 representation."""
    with pytest.raises(pdt.ValidationError):
        jobs.PromotionJob.DatasetHashes(inputs=value, targets="a" * 64, reference_inputs="b" * 64)
