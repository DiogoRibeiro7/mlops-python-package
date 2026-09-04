"""Verify an installed wheel from outside the checkout using only the standard library.

Invoke with the clean environment's Python and -I to disable source-path injection.
The checkout supplies expected metadata only; bikes must load from the environment.
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
        print(f"Installed bikes {installed_version} verified at {origin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
