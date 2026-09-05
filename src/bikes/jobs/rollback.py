"""Restore a recorded previous alias version through an explicit audit record."""

import typing as T

import mlflow
import pydantic as pdt
from mlflow.tracking import MlflowClient

from bikes.io import services
from bikes.jobs import base


class RollbackJob(base.Job):
    """Restore the previous version from a successful promotion.

    Parameters:
        promotion_run_id (str): completed promotion to reverse.
        expected_version (int): version that must still hold the alias.
        target_version (int): explicit version matching the recorded previous version.
        reason (str): operator's explanation for restoring that version.
        alias (str): alias named in the promotion record.
        run_config (RunConfig): tracking configuration for the rollback audit.

    Rollback restores registry routing; it does not re-evaluate the old model.
    Records are trusted and mutable. The final alias check is not an atomic
    compare-and-swap, so external serialization is required for concurrent writers.
    """

    KIND: T.Literal["RollbackJob"] = "RollbackJob"
    promotion_run_id: str = pdt.Field(min_length=1)
    expected_version: int = pdt.Field(gt=0)
    target_version: int = pdt.Field(gt=0)
    reason: str = pdt.Field(min_length=1, pattern=r"\S")
    alias: str = pdt.Field(default="Champion", min_length=1)
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(name="Rollback")

    def _validate_promotion(self, client: MlflowClient) -> None:
        """Require a matching successful promotion and an existing restore target."""
        if self.target_version == self.expected_version:
            raise ValueError("Rollback target must differ from the expected current version.")
        promotion = client.get_run(self.promotion_run_id)
        experiment = client.get_experiment_by_name(self.mlflow_service.experiment_name)
        if (
            experiment is None
            or promotion.info.experiment_id != experiment.experiment_id
            or promotion.info.lifecycle_stage != "active"
            or promotion.info.status != "FINISHED"
        ):
            raise ValueError("Promotion must be active and FINISHED in the configured experiment.")
        required = {
            "promotion.status": "applied",
            "promotion.model_name": self.mlflow_service.registry_name,
            "promotion.alias": self.alias,
            "promotion.version": str(self.expected_version),
            "promotion.previous_version": str(self.target_version),
        }
        for key, expected in required.items():
            if promotion.data.tags.get(key) != expected:
                raise ValueError(f"Promotion evidence mismatch: {key}.")
        client.get_model_version(
            name=self.mlflow_service.registry_name, version=str(self.target_version)
        )

    @T.override
    def run(self) -> base.Locals:
        client = self.mlflow_service.client()
        name = self.mlflow_service.registry_name
        with self.mlflow_service.run_context(self.run_config) as run:
            mlflow.set_tags(
                {
                    "rollback.status": "pending",
                    "rollback.model_name": name,
                    "rollback.alias": self.alias,
                    "rollback.promotion_run_id": self.promotion_run_id,
                    "rollback.from_version": str(self.expected_version),
                    "rollback.to_version": str(self.target_version),
                    "rollback.reason": self.reason,
                }
            )
            self._validate_promotion(client)
            # Keep this check immediately before the write. It rejects stale
            # records but cannot exclude another writer between these two calls.
            current = client.get_model_version_by_alias(name=name, alias=self.alias)
            if int(current.version) != self.expected_version:
                raise ValueError("Alias no longer points to the expected promoted version.")
            client.set_registered_model_alias(
                name=name, alias=self.alias, version=str(self.target_version)
            )
            mlflow.set_tag("rollback.status", "applied")
            self.alerts_service.notify(
                title="Rollback Job Finished",
                message=f"Version: {self.target_version} @ {self.alias}",
            )
        return locals()
