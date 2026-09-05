"""Promote an explicit model version using recorded evaluation evidence."""

import math
import typing as T

import mlflow
import pydantic as pdt
from mlflow.tracking import MlflowClient

from bikes.core import metrics
from bikes.io import registries, services
from bikes.jobs import base


class PromotionJob(base.Job):
    """Validate evidence before changing a registered model alias.

    Parameters:
        version (int): explicit candidate version; latest-version selection is forbidden.
        evaluation_run_id (str): completed evaluation of this exact candidate.
        dataset_digests (DatasetDigests): operator-selected MLflow dataset identities.
        thresholds (dict[str, Threshold]): nonempty promotion acceptance policy.
        alias (str): alias to update after validation.
        run_config (RunConfig): tracking configuration for the promotion audit.

    The tracking and registry stores are trusted and mutable. Dataset digests do
    not prove training provenance, and concurrent alias writes are not serialized.
    """

    class DatasetDigests(pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
        """Expected MLflow lineage digests, not cryptographic full-data hashes."""

        inputs: str = pdt.Field(min_length=1)
        targets: str = pdt.Field(min_length=1)
        reference_inputs: str = pdt.Field(min_length=1)

    KIND: T.Literal["PromotionJob"] = "PromotionJob"
    alias: str = pdt.Field(default="Champion", min_length=1)
    version: int = pdt.Field(gt=0)
    evaluation_run_id: str = pdt.Field(min_length=1)
    dataset_digests: DatasetDigests
    thresholds: dict[str, metrics.Threshold] = pdt.Field(
        default={"r2_score": metrics.Threshold(threshold=0.5, greater_is_better=True)},
        min_length=1,
    )
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Promotion"
    )

    def _validate_evidence(self, client: MlflowClient) -> None:
        """Fail before the alias mutation if any required evidence disagrees."""
        candidate = client.get_model_version(
            name=self.mlflow_service.registry_name, version=str(self.version)
        )
        run = client.get_run(self.evaluation_run_id)
        experiment = client.get_experiment_by_name(self.mlflow_service.experiment_name)
        if (
            experiment is None
            or run.info.experiment_id != experiment.experiment_id
            or run.info.lifecycle_stage != "active"
            or run.info.status != "FINISHED"
        ):
            raise ValueError("Evaluation must be active and FINISHED in the configured experiment.")
        if not candidate.run_id:
            raise ValueError("Candidate must have a source run ID.")
        required = {
            "evaluation.model_name": candidate.name,
            "evaluation.model_version": str(self.version),
            "evaluation.model_uri": registries.uri_for_model_version(candidate.name, self.version),
            "evaluation.model_source_run_id": candidate.run_id,
            "evaluation.thresholds": "passed",
            "evaluation.boundary": "passed_against_reference",
        }
        for key, expected in required.items():
            if run.data.tags.get(key) != expected:
                raise ValueError(f"Evaluation evidence mismatch: {key}.")
        for name, expected_digest in self.dataset_digests.model_dump().items():
            matches = [item for item in run.inputs.dataset_inputs if item.dataset.name == name]
            if len(matches) != 1 or matches[0].dataset.digest != expected_digest:
                raise ValueError(f"Evaluation dataset identity mismatch: {name}.")
            if name == "reference_inputs" and not any(
                tag.key == "mlflow.data.context" and tag.value == "evaluation_reference"
                for tag in matches[0].tags
            ):
                raise ValueError("Reference dataset must have evaluation_reference context.")
        # Recheck the operator's policy; a passing tag alone is insufficient.
        for name, value in run.data.metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"Evaluation metric {name!r} must be finite.")
        mlflow.validate_evaluation_results(
            validation_thresholds={
                name: bound.to_mlflow() for name, bound in self.thresholds.items()
            },
            candidate_result=mlflow.models.EvaluationResult(metrics=run.data.metrics, artifacts={}),
        )

    @T.override
    def run(self) -> base.Locals:
        client = self.mlflow_service.client()
        name = self.mlflow_service.registry_name
        with self.mlflow_service.run_context(self.run_config) as run:
            mlflow.set_tags(
                {
                    "promotion.status": "pending",
                    "promotion.model_name": name,
                    "promotion.version": str(self.version),
                    "promotion.alias": self.alias,
                    "promotion.evaluation_run_id": self.evaluation_run_id,
                }
            )
            mlflow.log_dict(self.dataset_digests.model_dump(), "promotion/dataset_digests.json")
            mlflow.log_dict(
                {name: bound.model_dump() for name, bound in self.thresholds.items()},
                "promotion/thresholds.json",
            )
            self._validate_evidence(client)
            previous_version = client.get_registered_model(name).aliases.get(self.alias, "")
            mlflow.set_tag("promotion.previous_version", previous_version)
            client.set_registered_model_alias(
                name=name, alias=self.alias, version=str(self.version)
            )
            mlflow.set_tag("promotion.status", "applied")
            self.alerts_service.notify(
                title="Promotion Job Finished",
                message=f"Version: {self.version} @ {self.alias}",
            )
        return locals()
