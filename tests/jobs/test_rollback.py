"""Test explicit rollback and refusal to overwrite changed registry state."""

from pathlib import Path

import mlflow
import pydantic as pdt
import pytest
from pytest_mock import MockerFixture

from bikes import jobs, scripts
from bikes.io import services


@pytest.fixture
def rollback(
    mlflow_service: services.MlflowService, alerts_service: services.AlertsService
) -> jobs.RollbackJob:
    """Create registry history without loading model artifacts."""
    client = mlflow_service.client()
    name = mlflow_service.registry_name
    client.create_registered_model(name)
    with mlflow_service.run_context(mlflow_service.RunConfig(name="Source")) as source:
        for _ in range(3):
            client.create_model_version(
                name, source=f"runs:/{source.info.run_id}/model", run_id=source.info.run_id
            )
    client.set_registered_model_alias(name, "Champion", "2")
    with mlflow_service.run_context(mlflow_service.RunConfig(name="Promotion")) as promotion:
        mlflow.set_tags(
            {
                "promotion.status": "applied",
                "promotion.model_name": name,
                "promotion.alias": "Champion",
                "promotion.version": "2",
                "promotion.previous_version": "1",
            }
        )
    return jobs.RollbackJob(
        mlflow_service=mlflow_service,
        alerts_service=alerts_service,
        promotion_run_id=promotion.info.run_id,
        expected_version=2,
        target_version=1,
        reason="Restore the previous routing after an incident.",
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
        "pending",
        "name",
        "alias",
        "version",
        "previous",
        "unset_previous",
        "missing_previous",
        "missing_target",
        "changed_alias",
        "missing_alias",
        "same_version",
        "repeated",
    ],
)
def test_rollback_guard(rollback: jobs.RollbackJob, case: str, mocker: MockerFixture) -> None:
    """Invalid or stale history must never reach the alias setter."""
    client = rollback.mlflow_service.client()
    name = rollback.mlflow_service.registry_name
    run_id = rollback.promotion_run_id
    data = rollback.model_dump()
    changes = {
        "pending": ("promotion.status", "pending"),
        "name": ("promotion.model_name", "Other"),
        "alias": ("promotion.alias", "Other"),
        "version": ("promotion.version", "3"),
        "previous": ("promotion.previous_version", "3"),
        "unset_previous": ("promotion.previous_version", ""),
    }
    if case in changes:
        client.set_tag(run_id, *changes[case])
    elif case in {"failed", "running"}:
        client.set_terminated(run_id, status=case.upper())
    elif case == "deleted":
        client.delete_run(run_id)
    elif case == "experiment":
        data["mlflow_service"]["experiment_name"] = "Other"
    elif case == "missing_run":
        data["promotion_run_id"] = "0" * 32
    elif case == "missing_previous":
        client.delete_tag(run_id, "promotion.previous_version")
    elif case == "missing_target":
        client.delete_model_version(name, "1")
    elif case == "changed_alias":
        client.set_registered_model_alias(name, "Champion", "3")
    elif case == "missing_alias":
        client.delete_registered_model_alias(name, "Champion")
    elif case == "same_version":
        data["target_version"] = 2
    elif case == "repeated":
        with rollback:
            rollback.run()
    before = client.get_registered_model(name).aliases.copy()
    job = jobs.RollbackJob.model_validate(data)
    setter = mocker.spy(mlflow.tracking.MlflowClient, "set_registered_model_alias")
    notify = mocker.spy(services.AlertsService, "notify")
    with job:
        if case == "valid":
            result = job.run()
            setter.assert_called_once()
            notify.assert_called_once()
            audit = client.get_run(result["run"].info.run_id)
            assert audit.info.status == "FINISHED"
            assert audit.data.tags["rollback.status"] == "applied"
            assert audit.data.tags["rollback.promotion_run_id"] == run_id
            assert audit.data.tags["rollback.reason"] == job.reason
            assert audit.data.tags["rollback.from_version"] == "2"
            assert audit.data.tags["rollback.to_version"] == "1"
            assert int(client.get_model_version_by_alias(name, "Champion").version) == 1
        else:
            with pytest.raises((ValueError, mlflow.exceptions.MlflowException)):
                job.run()
            setter.assert_not_called()
            notify.assert_not_called()
            assert client.get_registered_model(name).aliases == before
            experiment = client.get_experiment_by_name(job.mlflow_service.experiment_name)
            assert experiment is not None
            audits = client.search_runs(
                [experiment.experiment_id], filter_string="tags.`mlflow.runName` = 'Rollback'"
            )
            assert any(
                run.info.status == "FAILED" and run.data.tags["rollback.status"] == "pending"
                for run in audits
            )


@pytest.mark.parametrize(
    "change",
    [{"target_version": 0}, {"expected_version": 0}, {"reason": " "}, {"promotion_run_id": ""}],
)
def test_rollback_requires_explicit_configuration(
    rollback: jobs.RollbackJob, change: dict[str, object]
) -> None:
    with pytest.raises(pdt.ValidationError):
        jobs.RollbackJob.model_validate(rollback.model_dump() | change)


def test_rollback_cli(rollback: jobs.RollbackJob, tmp_path: Path) -> None:
    """The public job discriminator and CLI execute a configured rollback."""
    path = tmp_path / "rollback.json"
    path.write_text('{"job":' + rollback.model_dump_json() + "}", encoding="utf-8")
    assert scripts.main([str(path)]) == 0
    assert (
        int(
            rollback.mlflow_service.client()
            .get_model_version_by_alias(rollback.mlflow_service.registry_name, "Champion")
            .version
        )
        == 1
    )
