"""Define a job for evaluating registered models with data."""

# %% IMPORTS

import math
import typing as T

import mlflow
import pandas as pd
import pydantic as pdt
from mlflow.tracking import MlflowClient

from bikes.core import metrics as metrics_
from bikes.core import schemas
from bikes.io import datasets, provenance, registries, services
from bikes.jobs import base
from bikes.utils import splitters

# %% JOBS


class EvaluationsJob(base.Job):
    """Generate evaluations from a registered model and a dataset.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        targets (datasets.ReaderKind): reader for the targets data.
        reference_inputs (datasets.ReaderKind | None): declared development inputs for boundary checks.
        model_type (str): model type (e.g. "regressor", "classifier").
        alias_or_version (str | int): alias or version for the  model.
        metrics (metrics_.MetricsKind): metric list to compute.
        evaluators (list[str]): list of evaluators to use.
        thresholds (dict[str, metrics_.Threshold] | None): metric thresholds.
    """

    KIND: T.Literal["EvaluationsJob"] = "EvaluationsJob"

    # Run
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Evaluations"
    )
    # Data
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    reference_inputs: datasets.ReaderKind | None = pdt.Field(None, discriminator="KIND")
    # Model
    model_type: str = "regressor"
    alias_or_version: str | int = "Champion"
    # Loader
    loader: registries.LoaderKind = pdt.Field(registries.CustomLoader(), discriminator="KIND")
    # Metrics
    metrics: metrics_.MetricsKind = [metrics_.SklearnMetric()]
    # Evaluators
    evaluators: list[str] = ["default"]
    # Thresholds
    thresholds: dict[str, metrics_.Threshold] = {
        "r2_score": metrics_.Threshold(threshold=0.5, greater_is_better=True)
    }

    def _validate_reference(self, inputs: schemas.Inputs) -> None:
        """Record whether the supplied evaluation/reference boundary passed.

        Missing references retain diagnostic evaluation compatibility and are
        explicitly marked unchecked. A supplied invalid reference always fails.
        """
        mlflow.set_tag("evaluation.boundary", "unchecked")
        if self.reference_inputs is None:
            return
        mlflow.set_tag("evaluation.boundary", "rejected")
        reference = schemas.InputsSchema.check(self.reference_inputs.read())
        splitters.check_temporal_boundary(reference, inputs)
        provenance.log_frames({"reference_inputs": reference})
        mlflow.log_input(
            self.reference_inputs.lineage(data=reference, name="reference_inputs"),
            context="evaluation_reference",
        )
        mlflow.set_tag("evaluation.boundary", "passed_against_reference")

    def _resolve_model_uri(self, client: MlflowClient) -> str:
        """Resolve once and record the version that the loader will actually receive.

        Source run IDs may be absent for externally registered models. Recording
        that absence does not establish any training-data provenance.
        """
        name = self.mlflow_service.registry_name
        requested = self.alias_or_version
        if isinstance(requested, int):
            version = client.get_model_version(name=name, version=str(requested))
        else:
            version = client.get_model_version_by_alias(name=name, alias=requested)
        uri = registries.uri_for_model_version(name=name, version=int(version.version))
        mlflow.set_tags(
            {
                "evaluation.model_name": name,
                "evaluation.model_version": str(version.version),
                "evaluation.model_uri": uri,
                "evaluation.model_source_run_id": version.run_id or "",
                "evaluation.model_requested": str(requested),
            }
        )
        return uri

    def _record_policy(self) -> None:
        """Persist the configured policy before any data or model operation."""
        mlflow.set_tag("evaluation.thresholds", "pending")
        mlflow.log_dict(
            {name: threshold.model_dump() for name, threshold in self.thresholds.items()},
            "evaluation/thresholds.json",
        )

    def _validate_results(
        self,
        result: mlflow.models.EvaluationResult,
        thresholds: dict[str, metrics_.MlflowThreshold],
    ) -> None:
        """Reject invalid metrics and record whether a nonempty policy passed.

        This is metric acceptance only. Consumers must also inspect run status,
        model identity, dataset evidence and the reference boundary separately.
        """
        mlflow.set_tag("evaluation.thresholds", "rejected")
        for name, value in result.metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"Evaluation metric {name!r} must be finite.")
        mlflow.validate_evaluation_results(
            validation_thresholds=thresholds, candidate_result=result
        )
        mlflow.set_tag("evaluation.thresholds", "passed" if thresholds else "unchecked")

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client.tracking_uri)
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            self._record_policy()
            logger.info("With run context: {}", run.info)
            # data
            # - inputs
            logger.info("Read inputs: {}", self.inputs)
            inputs_ = self.inputs.read()  # unchecked!
            inputs = schemas.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
            logger.info("Read targets: {}", self.targets)
            targets_ = self.targets.read()  # unchecked!
            targets = schemas.TargetsSchema.check(targets_)
            schemas.check_row_alignment(inputs, targets)
            self._validate_reference(inputs)
            provenance.log_frames({"inputs": inputs, "targets": targets})
            logger.debug("- Targets shape: {}", targets.shape)
            # lineage
            # - inputs
            logger.info("Log lineage: inputs")
            inputs_lineage = self.inputs.lineage(data=inputs, name="inputs")
            mlflow.log_input(dataset=inputs_lineage, context=self.run_config.name)
            logger.debug("- Inputs lineage: {}", inputs_lineage.to_dict())
            # - targets
            logger.info("Log lineage: targets")
            targets_lineage = self.targets.lineage(
                data=targets, name="targets", targets=schemas.TargetsSchema.cnt
            )
            mlflow.log_input(dataset=targets_lineage, context=self.run_config.name)
            logger.debug("- Targets lineage: {}", targets_lineage.to_dict())
            # model
            logger.info("With model: {}", self.mlflow_service.registry_name)
            model_uri = self._resolve_model_uri(client)
            logger.debug("- Model URI: {}", model_uri)
            # loader
            logger.info("Load model: {}", self.loader)
            model = self.loader.load(uri=model_uri)
            logger.debug("- Model: {}", model)
            # outputs
            logger.info("Predict outputs: {}", len(inputs))
            outputs = model.predict(inputs=inputs)  # checked
            schemas.check_row_alignment(targets, outputs)
            logger.debug("- Outputs shape: {}", outputs.shape)
            # dataset
            logger.info("Create dataset: inputs & targets & outputs")
            dataset_ = pd.concat([inputs, targets, outputs], axis="columns")
            dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
                df=dataset_,
                name="evaluation",
                targets=schemas.TargetsSchema.cnt,
                predictions=schemas.OutputsSchema.prediction,
            )
            logger.debug("- Dataset: {}", dataset.to_dict())
            # metrics
            logger.debug("Convert metrics: {}", self.metrics)
            extra_metrics = [metric.to_mlflow() for metric in self.metrics]
            logger.debug("- Extra metrics: {}", extra_metrics)
            # thresholds
            logger.info("Convert thresholds: {}", self.thresholds)
            validation_thresholds = {
                name: threshold.to_mlflow() for name, threshold in self.thresholds.items()
            }
            logger.debug("- Validation thresholds: {}", validation_thresholds)
            # evaluations
            logger.info("Compute evaluations: {}", self.model_type)
            evaluations = mlflow.evaluate(
                data=dataset,
                model_type=self.model_type,
                evaluators=self.evaluators,
                extra_metrics=extra_metrics,
            )
            self._validate_results(evaluations, validation_thresholds)
            logger.debug("- Evaluations metrics: {}", evaluations.metrics)
            # notify
            self.alerts_service.notify(
                title="Evaluations Job Finished",
                message=f"Evaluation metrics: {evaluations.metrics}",
            )
        return locals()
