"""Fingerprint validated numeric/datetime dataframes without sampling rows."""

import hashlib
import json
import typing as T
from collections.abc import Mapping

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd

FORMAT = "bikes.dataframe.v1"


def _frame(payload: bytes) -> bytes:
    """Length-prefix each part to prevent ambiguous concatenation."""
    return len(payload).to_bytes(8, "big") + payload


def _array_payload(values: npt.NDArray[T.Any]) -> bytes:
    """Encode finite numeric or datetime values using canonical little-endian bytes."""
    dtype = values.dtype.newbyteorder("<")
    if values.ndim != 1 or dtype.kind not in "biufM":
        raise TypeError("Fingerprint arrays must be one-dimensional numeric or datetime values.")
    if dtype.itemsize not in {1, 2, 4, 8}:
        raise TypeError("Fingerprint arrays require numeric widths of at most 64 bits.")
    if not np.isfinite(values).all():
        raise ValueError("Fingerprint values must be finite and nonmissing.")
    # Preserve exact IEEE floating-point values, including signed zero. Never
    # round through decimal text or reduce the table to 64-bit per-row hashes.
    canonical = values.astype(dtype, copy=False)
    return _frame(dtype.str.encode("ascii")) + _frame(canonical.tobytes(order="C"))


def fingerprint(data: pd.DataFrame) -> str:
    """Return the SHA-256 of every value, row ID and the ordered schema.

    Version 1 supports single-level numeric/datetime indices and columns with
    string labels and NumPy dtypes. Nulls, objects and extension dtypes fail.
    Row/column order, index name, dtype width, datetime unit and exact float bits
    are significant. Byte order is normalized. This hashes validated model data,
    not source-file bytes or values discarded by schema conversion.
    """
    if data.empty or isinstance(data.index, pd.MultiIndex) or not data.columns.is_unique:
        raise ValueError(
            "Fingerprint data must be nonempty with a single index and unique columns."
        )
    if not all(isinstance(name, str) for name in data.columns):
        raise TypeError("Fingerprint column names must be strings.")
    if data.index.name is not None and not isinstance(data.index.name, str):
        raise TypeError("Fingerprint index names must be strings or None.")
    if not isinstance(data.index.dtype, np.dtype) or not all(
        isinstance(dtype, np.dtype) for dtype in data.dtypes
    ):
        raise TypeError("Fingerprint data must use NumPy dtypes, not extension dtypes.")
    header = {
        "format": FORMAT,
        "columns": list(data.columns),
        "index_name": data.index.name,
        "rows": len(data),
    }
    digest = hashlib.sha256()
    digest.update(
        _frame(json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    )
    digest.update(_frame(_array_payload(data.index.to_numpy())))
    for column in data.columns:
        digest.update(_frame(_array_payload(data[column].to_numpy())))
    return digest.hexdigest()


def log_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    """Write fingerprints and shape metadata to the active MLflow run.

    Each role has a separate artifact so optional reference data does not
    overwrite previously recorded input/target metadata. No raw rows are logged.
    """
    for role, data in frames.items():
        sha256 = fingerprint(data)
        mlflow.log_dict(
            {
                "format": FORMAT,
                "sha256": sha256,
                "rows": len(data),
                "columns": list(data.columns),
                "dtypes": [str(dtype) for dtype in data.dtypes],
                "index_name": data.index.name,
                "index_dtype": str(data.index.dtype),
            },
            f"provenance/{role}.json",
        )
        mlflow.set_tags({f"data.{role}.sha256": sha256, f"data.{role}.format": FORMAT})
