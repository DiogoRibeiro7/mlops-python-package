"""Verify full-table sensitivity, canonical encoding and persisted metadata."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from bikes.io import provenance, services


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small table covering index, exact floating point, boolean and datetime values."""
    return pd.DataFrame(
        {
            "value": [0.0, 0.1],
            "flag": [True, False],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        },
        index=pd.Index(np.array([7, 8], dtype="uint32"), name="instant"),
    )


def test_fingerprint_round_trip(frame: pd.DataFrame, tmp_path: Path) -> None:
    before = frame.copy(deep=True)
    path = tmp_path / "data.parquet"
    frame.to_parquet(path)
    assert provenance.fingerprint(frame) == provenance.fingerprint(pd.read_parquet(path))
    pd.testing.assert_frame_equal(frame, before)
    other_endian = frame.copy()
    other_endian["value"] = other_endian["value"].to_numpy().astype(">f8")
    assert provenance.fingerprint(frame) == provenance.fingerprint(other_endian)


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "ulp",
        "signed_zero",
        "row_order",
        "row_id",
        "index_name",
        "column_order",
        "column_name",
        "dtype",
        "timestamp",
        "bool",
    ],
)
def test_fingerprint_sensitivity(frame: pd.DataFrame, change: str) -> None:
    changed = frame.copy(deep=True)
    if change == "value":
        changed.loc[8, "value"] = 0.2
    elif change == "ulp":
        changed.loc[8, "value"] = np.nextafter(0.1, 1.0)
    elif change == "signed_zero":
        changed.loc[7, "value"] = -0.0
    elif change == "row_order":
        changed = changed.iloc[::-1]
    elif change == "row_id":
        changed.index = pd.Index(np.array([7, 9], dtype="uint32"), name="instant")
    elif change == "index_name":
        changed.index.name = "other"
    elif change == "column_order":
        changed = changed[changed.columns[::-1]]
    elif change == "column_name":
        changed = changed.rename(columns={"value": "other"})
    elif change == "dtype":
        changed["value"] = changed["value"].astype("float32")
    elif change == "timestamp":
        changed.loc[8, "time"] = pd.Timestamp("2024-01-02") + pd.Timedelta(1, unit="ns")
    elif change == "bool":
        changed.loc[8, "flag"] = True
    assert provenance.fingerprint(frame) != provenance.fingerprint(changed)


def test_fingerprint_includes_late_rows() -> None:
    data = pd.DataFrame({"value": np.arange(25000, dtype="int64")})
    original = provenance.fingerprint(data)
    data.loc[24000, "value"] = 24001
    assert provenance.fingerprint(data) != original


@pytest.mark.parametrize(
    "case", ["empty", "nan", "inf", "nat", "object", "nullable", "duplicate_columns", "multiindex"]
)
def test_fingerprint_rejects_unsupported_data(frame: pd.DataFrame, case: str) -> None:
    if case == "empty":
        frame = frame.iloc[:0]
    elif case in {"nan", "inf"}:
        frame.loc[7, "value"] = float(case)
    elif case == "nat":
        frame.loc[7, "time"] = pd.NaT
    elif case == "object":
        frame["value"] = ["a", "b"]
    elif case == "nullable":
        frame["value"] = pd.Series([1, 2], index=frame.index, dtype="Int64")
    elif case == "duplicate_columns":
        frame.columns = pd.Index(["same", "same", "time"])
    elif case == "multiindex":
        frame.index = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    with pytest.raises((TypeError, ValueError)):
        provenance.fingerprint(frame)


def test_logged_fingerprint(frame: pd.DataFrame, mlflow_service: services.MlflowService) -> None:
    with mlflow_service.run_context(mlflow_service.RunConfig(name="Fingerprint")) as run:
        provenance.log_frames({"inputs": frame})
        provenance.log_frames({"reference_inputs": frame.iloc[:1]})
    saved = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/provenance/inputs.json")
    assert saved["format"] == provenance.FORMAT
    assert saved["sha256"] == provenance.fingerprint(frame)
    assert saved["rows"] == len(frame)
    assert saved["columns"] == list(frame.columns)
    assert saved["dtypes"] == [str(dtype) for dtype in frame.dtypes]
    assert saved["index_dtype"] == str(frame.index.dtype)
    tags = mlflow_service.client().get_run(run.info.run_id).data.tags
    assert tags["data.inputs.sha256"] == saved["sha256"]
    assert tags["data.reference_inputs.sha256"] == provenance.fingerprint(frame.iloc[:1])


def test_fingerprint_v1_known_vector() -> None:
    """Freeze the public format so changes require a new format identifier."""
    data = pd.DataFrame(
        {"x": np.array([1, 2], dtype="<i8")},
        index=pd.Index(np.array([10, 20], dtype="<u4"), name="row"),
    )
    assert (
        provenance.fingerprint(data)
        == "f8c739f543929fd818345e7da2464454c6c4a842705b9045e7fc071d3cf63e03"
    )
