# %% IMPORTS

from contextlib import ExitStack, nullcontext
from pathlib import Path

import _pytest.capture as pc
import mlflow
import pytest
from pytest_mock import MockerFixture

from bikes import jobs, scripts
from bikes.core import metrics, schemas
from bikes.io import datasets, provenance, registries, services

# %% JOBS


@pytest.mark.parametrize(
    "alias_or_version, thresholds",
    [
        (
            1,
            {"mean_squared_error": metrics.Threshold(threshold=1e12, greater_is_better=False)},
        ),
        (
            "Promotion",
            {"r2_score": metrics.Threshold(threshold=-1, greater_is_better=True)},
        ),
        pytest.param(
            "Promotion",
            {"r2_score": metrics.Threshold(threshold=100, greater_is_better=True)},
            marks=pytest.mark.xfail(
                reason="Invalid threshold for metric.",
                raises=metrics.MlflowModelValidationFailedException,
                strict=True,
            ),
        ),
    ],
)
def test_evaluations_job(
    alias_or_version: str | int,
    thresholds: dict[str, metrics.Threshold],
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.ParquetReader,
    targets_reader: datasets.ParquetReader,
    model_alias: registries.Version,
    metric: metrics.SklearnMetric,
    capsys: pc.CaptureFixture[str],
    mocker: MockerFixture,
) -> None:
    # given
    if isinstance(alias_or_version, int):
        assert alias_or_version == model_alias.version, "Model version should be the same!"
    else:
        assert alias_or_version == model_alias.aliases[0], "Model alias should be the same!"
    run_config = mlflow_service.RunConfig(
        name="EvaluationsTest",
        tags={"context": "evaluations"},
        description="Evaluations job.",
    )
    client = mlflow_service.client()
    original_load = registries.CustomLoader.load
    loaded_uris: list[str] = []

    def load_after_alias_change(
        loader: registries.CustomLoader, uri: str
    ) -> registries.CustomLoader.Adapter:
        """Move the alias after resolution, immediately before real model loading."""
        loaded_uris.append(uri)
        if isinstance(alias_or_version, str):
            replacement = client.create_model_version(
                name=mlflow_service.registry_name,
                source=model_alias.source,
                run_id=model_alias.run_id,
            )
            client.set_registered_model_alias(
                name=mlflow_service.registry_name,
                alias=alias_or_version,
                version=replacement.version,
            )
        return original_load(loader, uri)

    mocker.patch.object(
        registries.CustomLoader, "load", side_effect=load_after_alias_change, autospec=True
    )
    # when
    job = jobs.EvaluationsJob(
        logger_service=logger_service,
        alerts_service=alerts_service,
        mlflow_service=mlflow_service,
        run_config=run_config,
        inputs=inputs_reader,
        targets=targets_reader,
        alias_or_version=alias_or_version,
        metrics=[metric],
        thresholds=thresholds,
    )
    with job as runner:
        out = runner.run()
    # then
    fingerprint_tags = mlflow_service.client().get_run(out["run"].info.run_id).data.tags
    assert fingerprint_tags["data.inputs.sha256"] == provenance.fingerprint(out["inputs"])
    assert fingerprint_tags["data.targets.sha256"] == provenance.fingerprint(out["targets"])
    assert (
        out["client"].get_run(out["run"].info.run_id).data.tags["evaluation.boundary"]
        == "unchecked"
    )
    # - vars
    assert set(out) == {
        "self",
        "logger",
        "client",
        "run",
        "inputs",
        "inputs_",
        "inputs_lineage",
        "targets",
        "targets_",
        "targets_lineage",
        "outputs",
        "model",
        "model_uri",
        "dataset",
        "dataset_",
        "extra_metrics",
        "validation_thresholds",
        "evaluations",
    }
    # - run
    assert run_config.tags is not None, "Run config tags should be set!"
    assert out["run"].info.run_name == run_config.name, "Run name should be the same!"
    assert run_config.description in out["run"].data.tags.values(), "Run desc. should be tags!"
    assert out["run"].data.tags.items() > run_config.tags.items(), (
        "Run tags should be a subset of tags!"
    )
    # - data
    assert out["inputs"].ndim == out["inputs_"].ndim == 2, "Inputs should be a dataframe!"
    assert out["targets"].ndim == out["targets_"].ndim == 2, "Targets should be a dataframe!"
    # - lineage
    assert out["inputs_lineage"].name == "inputs", "Inputs lineage name should be inputs!"
    assert out["inputs_lineage"].source.uri == inputs_reader.path, (
        "Inputs lineage source should be the inputs reader path!"
    )
    assert out["targets_lineage"].name == "targets", "Targets lineage name should be targets!"
    assert out["targets_lineage"].source.uri == targets_reader.path, (
        "Targets lineage source should be the targets reader path!"
    )
    assert out["targets_lineage"].targets == schemas.TargetsSchema.cnt, (
        "Targets lineage target should be cnt!"
    )
    # - outputs
    assert out["outputs"].ndim == 2, "Outputs should be a dataframe!"
    # - model uri
    expected_uri = registries.uri_for_model_version(
        mlflow_service.registry_name, int(model_alias.version)
    )
    assert loaded_uris == [expected_uri]
    assert out["model_uri"] == expected_uri
    tags = client.get_run(out["run"].info.run_id).data.tags
    assert tags["evaluation.thresholds"] == "passed"
    assert tags["evaluation.model_name"] == mlflow_service.registry_name
    assert tags["evaluation.model_version"] == str(model_alias.version)
    assert tags["evaluation.model_uri"] == expected_uri
    assert tags["evaluation.model_source_run_id"] == model_alias.run_id
    assert tags["evaluation.model_requested"] == str(alias_or_version)
    if isinstance(alias_or_version, str):
        moved = client.get_model_version_by_alias(
            name=mlflow_service.registry_name, alias=alias_or_version
        )
        assert moved.version != model_alias.version
    assert mlflow_service.registry_name in out["model_uri"], (
        "Model URI should contain the registry name!"
    )
    # - model
    assert out["model"].model.metadata.run_id == model_alias.run_id, (
        "Model run id should be the same!"
    )
    assert out["model"].model.metadata.signature is not None, "Model should have a signature!"
    assert out["model"].model.metadata.flavors.get("python_function"), (
        "Model should have a pyfunc flavor!"
    )
    # - dataset
    assert out["dataset"].name == "evaluation", "Dataset name should be evaluation!"
    assert out["dataset"].targets == schemas.TargetsSchema.cnt, (
        "Dataset targets should be the target column!"
    )
    assert out["dataset"].predictions == schemas.OutputsSchema.prediction, (
        "Dataset predictions should be the prediction column!"
    )
    assert out["dataset"].source.to_dict().keys() == {"tags"}, "Dataset source should have tags!"
    # - extra metrics
    assert len(out["extra_metrics"]) == len(job.metrics), (
        "Extra metrics should have the same length as metrics!"
    )
    assert out["extra_metrics"][0].name == job.metrics[0].name, (
        "Extra metrics name should be the same!"
    )
    assert out["extra_metrics"][0].greater_is_better == job.metrics[0].greater_is_better, (
        "Extra metrics greatter is better should be the same!"
    )
    # - validation thresholds
    assert out["validation_thresholds"].keys() == thresholds.keys(), (
        "Validation thresholds should have the same keys as thresholds!"
    )
    # - evaluations
    assert out["evaluations"].metrics["example_count"] == inputs_reader.limit, (
        "Evaluations should have the same number of examples as the inputs!"
    )
    assert job.metrics[0].name in out["evaluations"].metrics, "Metric should be logged in Mlflow!"
    # - mlflow tracking
    experiment = mlflow_service.client().get_experiment_by_name(name=mlflow_service.experiment_name)
    assert experiment is not None, "Mlflow Experiment should exist!"
    assert experiment.name == mlflow_service.experiment_name, (
        "Mlflow Experiment name should be the same!"
    )
    runs = mlflow_service.client().search_runs(experiment_ids=experiment.experiment_id)
    assert len(runs) == 2, "There should be a two Mlflow run for training and evaluations!"
    assert metric.name in runs[0].data.metrics, "Metric should be logged in Mlflow!"
    assert runs[0].info.status == "FINISHED", "Mlflow run status should be set as FINISHED!"
    # - alerting service
    assert "Evaluations" in capsys.readouterr().out, "Alerting service should be called!"


@pytest.mark.parametrize("overlap", [False, True])
def test_evaluation_reference_boundary(
    overlap: bool,
    tmp_path: Path,
    train_test_sets: tuple[schemas.Inputs, schemas.Targets, schemas.Inputs, schemas.Targets],
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    model_alias: registries.Version,
    mocker: MockerFixture,
) -> None:
    reference, _, evaluation, targets = train_test_sets
    reference_path = tmp_path / "reference.parquet"
    inputs_path = tmp_path / "inputs.parquet"
    targets_path = tmp_path / "targets.parquet"
    (evaluation if overlap else reference).to_parquet(reference_path)
    evaluation.to_parquet(inputs_path)
    targets.to_parquet(targets_path)
    loader = mocker.spy(registries.CustomLoader, "load")
    job = jobs.EvaluationsJob(
        mlflow_service=mlflow_service,
        alerts_service=alerts_service,
        logger_service=logger_service,
        inputs=datasets.ParquetReader(path=str(inputs_path)),
        targets=datasets.ParquetReader(path=str(targets_path)),
        reference_inputs=datasets.ParquetReader(path=str(reference_path)),
        run_config=mlflow_service.RunConfig(name="ReferenceBoundaryTest"),
        alias_or_version=model_alias.aliases[0],
        thresholds={
            "mean_squared_error": metrics.Threshold(threshold=1e12, greater_is_better=False)
        },
    )
    with job as runner:
        if overlap:
            with pytest.raises(ValueError, match="disjoint"):
                runner.run()
            loader.assert_not_called()
            client = mlflow_service.client()
            experiment = client.get_experiment_by_name(mlflow_service.experiment_name)
            assert experiment is not None
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.`mlflow.runName` = 'ReferenceBoundaryTest'",
            )
            assert len(runs) == 1
            assert runs[0].info.status == "FAILED"
            assert runs[0].data.tags["evaluation.boundary"] == "rejected"
        else:
            out = runner.run()
            loader.assert_called_once()
            run = out["client"].get_run(out["run"].info.run_id)
            assert run.data.tags["evaluation.thresholds"] == "passed"
            assert run.data.tags["evaluation.boundary"] == "passed_against_reference"
            assert run.data.tags["data.reference_inputs.sha256"] == provenance.fingerprint(
                reference
            )
            assert any(
                item.dataset.name == "reference_inputs" for item in run.inputs.dataset_inputs
            )
            promotion = jobs.PromotionJob(
                mlflow_service=mlflow_service,
                alerts_service=alerts_service,
                version=int(model_alias.version),
                evaluation_run_id=run.info.run_id,
                dataset_digests=jobs.PromotionJob.DatasetDigests.model_validate(
                    {
                        item.dataset.name: item.dataset.digest
                        for item in run.inputs.dataset_inputs
                        if item.dataset.name in {"inputs", "targets", "reference_inputs"}
                    }
                ),
                thresholds=job.thresholds,
            )
            config_path = tmp_path / "promotion.json"
            config_path.write_text('{"job":' + promotion.model_dump_json() + "}", encoding="utf-8")
            assert scripts.main([str(config_path)]) == 0
            assert int(
                mlflow_service.client()
                .get_model_version_by_alias(mlflow_service.registry_name, "Champion")
                .version
            ) == int(model_alias.version)


@pytest.mark.parametrize(
    "values, configured, expected",
    [
        ({"r2_score": 0.8}, True, "passed"),
        ({"r2_score": 0.2}, True, "rejected"),
        ({}, True, "rejected"),
        ({"r2_score": float("nan")}, True, "rejected"),
        ({"r2_score": float("inf")}, True, "rejected"),
        ({"r2_score": float("-inf")}, True, "rejected"),
        ({"r2_score": 0.8, "other": float("nan")}, True, "rejected"),
        ({"r2_score": 0.8}, False, "unchecked"),
        ({"r2_score": float("nan")}, False, "rejected"),
    ],
)
def test_recorded_metric_acceptance(
    values: dict[str, float],
    configured: bool,
    expected: str,
    mlflow_service: services.MlflowService,
    inputs_reader: datasets.ParquetReader,
    targets_reader: datasets.ParquetReader,
) -> None:
    """Exercise the real validator and tracking store without retraining a model."""
    policy = {"r2_score": metrics.Threshold(threshold=0.5, greater_is_better=True)}
    job = jobs.EvaluationsJob(
        mlflow_service=mlflow_service,
        inputs=inputs_reader,
        targets=targets_reader,
        thresholds=policy if configured else {},
    )
    result = mlflow.models.EvaluationResult(metrics=values, artifacts={})
    thresholds = {name: value.to_mlflow() for name, value in job.thresholds.items()}
    with job, ExitStack() as setup:
        run = setup.enter_context(mlflow_service.run_context(job.run_config))
        job._record_policy()
        assert (
            mlflow_service.client().get_run(run.info.run_id).data.tags["evaluation.thresholds"]
            == "pending"
        )
        expectation = (
            pytest.raises((ValueError, metrics.MlflowModelValidationFailedException))
            if expected == "rejected"
            else nullcontext()
        )
        # Transfer run cleanup only after setup succeeds. Validation exceptions
        # reach MLflow before pytest catches them, so rejected runs become FAILED.
        with expectation, setup.pop_all():
            job._validate_results(result, thresholds)
        recorded = mlflow_service.client().get_run(run.info.run_id)
        assert recorded.data.tags["evaluation.thresholds"] == expected
        assert recorded.info.status == ("FAILED" if expected == "rejected" else "FINISHED")
        saved = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/evaluation/thresholds.json")
        assert saved == {name: value.model_dump() for name, value in job.thresholds.items()}
