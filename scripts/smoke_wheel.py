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
        for role, table in {"inputs": inputs, "targets": targets}.items():
            if (
                run.data.tags.get(f"data.{role}.sha256") != provenance.fingerprint(table)
                or run.data.tags.get(f"data.{role}.format") != provenance.FORMAT
            ):
                raise RuntimeError(f"Installed training fingerprint mismatch: {role}")
    print("Installed training, registration, reload and prediction roundtrip verified")


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
